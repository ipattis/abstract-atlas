import json

import pandas as pd
import pytest

from abstract_atlas import audit, embed, enrich, trends
from abstract_atlas.backends import make_backend
from abstract_atlas.backends.claude import BedrockBatchBackend, parse_response, tool_body
from abstract_atlas.backends.mock import MockBackend
from abstract_atlas.backends.openai_compat import parse_json
from abstract_atlas.data import load_enrichment, load_papers
from abstract_atlas.rules import derive, matches
from abstract_atlas.schema import Field, object_schema, validate_record


def test_object_schema_and_validation():
    fields = [Field("a", "text"), Field("b", "enum", options=["x", "y"]), Field("c", "enum_list", options=["p", "q"]),
              Field("n", "int"), Field("l", "list")]
    schema = object_schema(fields)
    assert schema["required"] == ["a", "b", "c", "n", "l"] and schema["properties"]["b"]["enum"] == ["x", "y"]
    ok = validate_record({"a": " hi ", "b": "x", "c": ["p"], "n": "7", "l": "USA, Canada"}, fields)
    assert ok == {"a": "hi", "b": "x", "c": ["p"], "n": 7, "l": ["USA", "Canada"]}
    for bad in ({"a": "", "b": "x", "c": [], "n": 1, "l": []}, {"a": "t", "b": "z", "c": [], "n": 1, "l": []},
                {"a": "t", "b": "x", "c": ["nope"], "n": 1, "l": []}, {"a": "t", "b": "x", "c": [], "n": "many", "l": []}):
        with pytest.raises(ValueError):
            validate_record(bad, fields)


def test_audit_role_defaults():
    assert Field("a", "text").audit == "text" and Field("b", "int").audit == "skip"
    with pytest.raises(ValueError):
        Field("c", "enum")  # no options


SPEC = {"default": "Unclear", "rules": [
    {"if": {"study": {"startswith": ["Randomized", "Quasi"]}, "outcomes": {"contains": "Learning"}}, "then": "Causal"},
    {"if": {"study": {"startswith": ["Randomized", "Quasi"]}}, "then": "Causal, other"},
    {"if": {"n": {"gte": 100}}, "then": "Big"}]}


def test_rules_first_match_wins():
    assert derive({"study": "Randomized trial", "outcomes": ["Learning"]}, SPEC) == "Causal"
    assert derive({"study": "Quasi-experimental", "outcomes": ["Perceptions"]}, SPEC) == "Causal, other"
    assert derive({"study": "Survey", "n": 500}, SPEC) == "Big"
    assert derive({}, SPEC) == "Unclear"
    with pytest.raises(ValueError):
        matches({"a": {"bogus": 1}}, {"a": 1})


def test_parse_json_tolerates_think_and_fences():
    assert parse_json('<think>hmm</think>\n```json\n{"a": 1}\n```') == {"a": 1}


def test_claude_tool_body_forced_and_fallback():
    from abstract_atlas.backends import Request

    req = Request("p", {"type": "object", "properties": {}}, name="rec", system="sys")
    forced, auto = tool_body(req, True), tool_body(req, False)
    assert forced["tool_choice"] == {"type": "tool", "name": "rec"} and forced["system"] == "sys"
    assert auto["tool_choice"] == {"type": "auto"} and "calling the rec tool" in auto["system"]
    resp = {"content": [{"type": "tool_use", "input": {"a": 1}}], "usage": {"input_tokens": 3, "output_tokens": 4}}
    assert parse_response(resp)["output"] == {"a": 1} and parse_response({"error": "x"}) == {"error": "x"}


def test_batch_fingerprint_changes_with_prompt():
    a = {"r1": {"messages": ["hello"]}}
    b = {"r1": {"messages": ["hello!"]}}
    assert BedrockBatchBackend.fingerprint(a) == BedrockBatchBackend.fingerprint(dict(a))
    assert BedrockBatchBackend.fingerprint(a) != BedrockBatchBackend.fingerprint(b)


def test_make_backend_types(tmp_path):
    assert isinstance(make_backend({"type": "mock"}, tmp_path), MockBackend)
    with pytest.raises(SystemExit):
        make_backend({"type": "nope"}, tmp_path)


def test_enrich_is_incremental_and_refresh_keeps_old_fields(cfg):
    backend = MockBackend()
    enrich.run(cfg, backend, limit=20)
    assert len(load_enrichment(cfg)) == 20
    enrich.run(cfg, backend)            # only the remaining papers
    done = load_enrichment(cfg)
    assert len(done) == len(load_papers(cfg))
    enrich.run(cfg, backend)            # nothing left: file doesn't grow
    assert sum(1 for _ in open(cfg.workdir / "enrichment.jsonl")) == len(done)

    cfg.raw["extraction"]["prompt_version"] = 2
    before = dict(done["syn-000"])
    enrich.run(cfg, backend, refresh_fields=["summary"])
    after = load_enrichment(cfg)["syn-000"]
    assert after["prompt_version"] == 2 and after["method"] == before["method"] and "refresh_model" in after


def test_enrich_drops_invalid_records(cfg):
    class Bad(MockBackend):
        def run(self, requests, job="job"):
            res = super().run(requests, job)
            first = next(iter(res))
            res[first]["output"]["contribution"] = "not an option"
            return res

    enrich.run(cfg, Bad(), limit=5)
    assert len(load_enrichment(cfg)) == 4


def test_estimate_runs(cfg, capsys):
    enrich.estimate(cfg, price_in=3, price_out=15)
    out = capsys.readouterr().out
    assert "108 papers" in out and "batch at 50%" in out


def test_embed_cache_keys_on_text(cfg, tmp_path):
    calls = []

    def encode(batch):
        calls.append(len(batch))
        return [[float(len(t)), 1.0] for t in batch]

    path = tmp_path / "e.npz"
    v1 = embed.embed_cached(path, ["aa", "bbb"], encode, "m", "p")
    embed.embed_cached(path, ["aa", "bbb"], encode, "m", "p")
    v3 = embed.embed_cached(path, ["aa", "cccc"], encode, "m", "p")   # text changed -> only that one recomputed
    assert calls == [2, 1] and v1[0].tolist() == v3[0].tolist() and v3[1][0] == 4.0


def test_trends_share_math_and_labels():
    rows = ([{"period": "2025", "L0_id": 0, "L0_name": "A"}] * 10 + [{"period": "2025", "L0_id": 1, "L0_name": "B"}] * 10
            + [{"period": "2026", "L0_id": 0, "L0_name": "A"}] * 18 + [{"period": "2026", "L0_id": 1, "L0_name": "B"}] * 2)
    out = trends.topic_trends(pd.DataFrame(rows), ["2025", "2026"], min_latest=5)[0]
    assert out[0]["label"] == "Surging" and out[0]["growth"] > 0
    assert out[1]["label"] == "Small" and out[1]["growth"] < 0     # too few latest papers to call it shrinking


def test_period_ordering(cfg):
    cfg.raw["trends"] = {"pool_through": 2023}
    papers = load_papers(cfg)
    assert trends.ordered_periods(papers) == ["≤2023", "2024", "2025", "2026"]


def test_audit_end_to_end_with_mock(cfg):
    enrich.run(cfg, MockBackend())
    audit.run(cfg, MockBackend())
    a = pd.read_parquet(cfg.workdir / "audit.parquet")
    assert len(a) == 108 and {"summary", "contribution", "issues"} <= set(a.columns)
    assert (cfg.workdir / "audit_report.md").read_text().startswith("# Extraction audit")
    audit.run(cfg, MockBackend())       # nothing new to audit
    assert len(pd.read_parquet(cfg.workdir / "audit.parquet")) == 108
