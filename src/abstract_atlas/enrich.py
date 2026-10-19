"""Step 1: structured extraction from each abstract.

The model sees only the title and abstract, answers through a schema (so output is machine-checkable), and may
always say "not stated". Records are appended to enrichment.jsonl; re-runs only process new or incomplete papers.
`--refresh-fields` regenerates chosen fields for records written under an older `prompt_version` (the fix-after-audit loop).
"""
import json
from collections import defaultdict

from .backends import Request
from .config import Config
from .data import enrichment_path, load_enrichment, load_papers
from .schema import object_schema, subset, validate_record


def build_requests(cfg: Config, papers, fields):
    schema = object_schema(fields)
    out = {}
    for row in papers.to_dict("records"):
        prompt = cfg.prompt.format_map(defaultdict(lambda: "Not stated", row))
        out[row["pid"]] = Request(prompt=prompt, schema=schema, name="record_paper_analysis",
                                  description="Record a faithful, structured analysis of a paper, using only its abstract.",
                                  system=cfg.system, max_tokens=int(cfg.extraction.get("max_tokens", 1500)))
    return out


def select(cfg: Config, papers, done, refresh_fields=None):
    if refresh_fields:
        return papers[papers["pid"].map(lambda p: p in done and done[p].get("prompt_version", 1) < cfg.prompt_version)]
    all_fields = cfg.fields
    return papers[papers["pid"].map(lambda p: p not in done or not _complete(done[p], all_fields))]


def _complete(rec, fields):
    try:
        validate_record(rec, fields)
        return True
    except ValueError:
        return False


def run(cfg: Config, backend, limit=None, refresh_fields=None):
    papers, done = load_papers(cfg), load_enrichment(cfg)
    fields = subset(cfg.fields, refresh_fields) if refresh_fields else cfg.fields
    todo = select(cfg, papers, done, refresh_fields)
    if limit:
        todo = todo.head(limit)
    print(f"{len(done)} enriched, {len(todo)} to process with {backend.name}" + (f" (refreshing {refresh_fields})" if refresh_fields else ""))
    if todo.empty:
        return
    results = backend.run(build_requests(cfg, todo, fields), job="refresh" if refresh_fields else "enrich")

    saved = failed = tokens_in = tokens_out = 0
    with open(enrichment_path(cfg), "a") as f:
        for pid, res in results.items():
            try:
                if "error" in res:
                    raise ValueError(res["error"])
                clean = validate_record(res["output"], fields)
            except ValueError as e:
                failed += 1
                print(f"  skip {pid}: {str(e)[:100]}")
                continue
            if refresh_fields:  # keep the old record, overwrite only the refreshed fields
                rec = {**done[pid], **clean, "prompt_version": cfg.prompt_version, "refresh_model": backend.name}
            else:
                rec = {"pid": pid, **clean, "prompt_version": cfg.prompt_version, "model": backend.name}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            saved += 1
            tokens_in += res["usage"]["input_tokens"]
            tokens_out += res["usage"]["output_tokens"]
    print(f"Saved {saved} records. Tokens: {tokens_in:,} in / {tokens_out:,} out. Failed: {failed} (re-run to retry)")


def estimate(cfg: Config, avg_output_tokens=450, price_in=None, price_out=None, batch_discount=0.5):
    """Rough token count and cost before spending anything. Prices are per million tokens and yours to supply."""
    papers, done = load_papers(cfg), load_enrichment(cfg)
    todo = select(cfg, papers, done)
    reqs = build_requests(cfg, todo, cfg.fields)
    schema_chars = len(json.dumps(object_schema(cfg.fields)))
    chars = sum(len(r.prompt) + len(r.system or "") + schema_chars for r in reqs.values())
    tin, tout = chars // 4, avg_output_tokens * len(reqs)   # ~4 characters per token
    print(f"{len(reqs)} papers to extract: ~{tin:,} input tokens, ~{tout:,} output tokens (rough: chars/4, {avg_output_tokens} out/paper)")
    if price_in is not None and price_out is not None:
        cost = tin / 1e6 * price_in + tout / 1e6 * price_out
        print(f"On-demand ~${cost:,.2f}; batch at {batch_discount:.0%} of that ~${cost * batch_discount:,.2f}")
    else:
        print("Pass --price-in / --price-out (USD per million tokens, from your provider's price list) for a cost estimate.")
