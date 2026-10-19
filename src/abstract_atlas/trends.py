"""Step 4: which topics are gaining or losing ground.

Growth is computed from counts, never by the LLM: a topic's share of the latest period's papers against its share
of the previous period's, with +1 smoothing so tiny topics don't swing wildly. Shares, not raw counts, because the
latest period is usually a partial year. Topics with too few latest-period papers are labelled "Small" rather than
"Surging" or "Shrinking". The LLM only writes the optional "what's new" note, from a sample of each period's papers.
"""
import json
import math
import random

import pandas as pd

from .backends import Request
from .config import Config
from .data import load_mapped

NOTE_SCHEMA = {
    "type": "object",
    "properties": {"whats_new": {"type": "string", "description": "1-2 plain-language sentences on what is new or different in the latest period's papers."}},
    "required": ["whats_new"], "additionalProperties": False,
}
NOTE_PROMPT = """Topic on a map of {corpus}: "{name}".
Its share of papers went from {prev_share:.1%} in {prev} to {last_share:.1%} in {last}.

Sample of {last} papers in this topic:
{last_papers}

Sample of earlier papers in this topic:
{prev_papers}

In 1-2 plain sentences, say what is new or shifting in {last} compared with earlier papers. Be specific and use only what the samples show."""


def trend_label(ratio, n_last, min_latest):
    if n_last < min_latest:
        return "Small"
    if ratio >= 1.6:
        return "Surging"
    if ratio >= 1.15:
        return "Growing"
    if ratio <= 0.65:
        return "Shrinking"
    if ratio <= 0.87:
        return "Cooling"
    return "Steady"


def ordered_periods(papers):
    """Period labels in time order (pooled "≤YYYY" first, "Undated" excluded)."""
    ps = [p for p in papers["period"].unique() if p != "Undated"]
    return sorted(ps, key=lambda p: int(p.lstrip("≤")) + (-0.5 if p.startswith("≤") else 0))


def topic_trends(m, periods, min_latest):
    """m: map frame joined with papers' `period`. Returns {layer: {topic_id: info}}."""
    last, prev = periods[-1], periods[-2]
    totals = m["period"].value_counts()
    layers = sorted(int(c[1:-3]) for c in m.columns if c.startswith("L") and c.endswith("_id"))
    out = {}
    for layer in layers:
        out[layer] = {}
        for tid, sub in m[m[f"L{layer}_id"] >= 0].groupby(f"L{layer}_id"):
            counts = sub["period"].value_counts()
            share = {p: (counts.get(p, 0) + 1) / (totals.get(p, 0) + 1) for p in periods}
            ratio = share[last] / share[prev]
            out[layer][int(tid)] = {
                "name": sub[f"L{layer}_name"].iloc[0],
                "counts": {p: int(counts.get(p, 0)) for p in periods},
                "shares": {p: round(share[p], 4) for p in periods},
                "growth": round(math.log2(ratio), 3),   # log2 change in share: +1 = doubled
                "label": trend_label(ratio, int(counts.get(last, 0)), min_latest),
            }
    return out


def sample_lines(sub, enrich, k, rng, field):
    pids = list(sub["pid"])
    rng.shuffle(pids)
    titles = dict(zip(sub["pid"], sub["title"]))
    return "\n".join(f"- {titles[p]}: {enrich[p].get(field, '')}" for p in pids[:k])


def run(cfg: Config, backend, views=None):
    papers, enrich = load_mapped(cfg)
    periods = ordered_periods(papers)
    if len(periods) < 2:
        print("Fewer than two dated periods; skipping trends (set input.year and trends.pool_through)")
        return
    last, prev = periods[-1], periods[-2]
    t = cfg.trends
    summary_field = t.get("note_field", "summary")
    rng = random.Random(0)
    for view in views or list(cfg.views):
        m = pd.read_parquet(cfg.workdir / f"map_{view}.parquet").merge(papers[["pid", "period", "title"]], on="pid")
        trends = topic_trends(m, periods, t["min_latest"])
        layers = sorted(trends)
        note_layers = layers[1:-1] if len(layers) > 2 else layers   # specific enough to be interesting, few enough to be cheap
        requests = {}
        if t["note_model_calls"]:
            for layer in note_layers:
                for tid, info in trends[layer].items():
                    sub = m[m[f"L{layer}_id"] == tid]
                    latest, earlier = sub[sub["period"] == last], sub[sub["period"].isin(periods[:-1])]
                    if len(latest) >= 3 and len(earlier) >= 3:
                        prompt = NOTE_PROMPT.format(
                            corpus=cfg.raw.get("corpus_description", "research papers"), name=info["name"], prev=prev, last=last,
                            prev_share=info["shares"][prev], last_share=info["shares"][last],
                            last_papers=sample_lines(latest, enrich, 10, rng, summary_field),
                            prev_papers=sample_lines(earlier, enrich, 10, rng, summary_field))
                        requests[f"L{layer}-T{tid}"] = Request(prompt, NOTE_SCHEMA, "record_trend_note", "Describe how a topic changed.", max_tokens=300)
        results = backend.run(requests, job=f"trends-{view}") if requests else {}
        for key, res in results.items():
            layer, tid = key.split("-")
            if "output" in res:
                trends[int(layer[1:])][int(tid[1:])]["whats_new"] = res["output"].get("whats_new", "")
        (cfg.workdir / f"trends_{view}.json").write_text(json.dumps({"periods": periods, "layers": trends}, indent=1, ensure_ascii=False))
        mid = trends[layers[1 if len(layers) > 1 else 0]].values()
        top = sorted((v for v in mid if v["label"] != "Small"), key=lambda v: -v["growth"])[:5]
        print(f"{view}: {sum(map(len, trends.values()))} topics, {len(results)} notes. Fastest growing:",
              [f"{v['name']} ({2 ** v['growth']:.1f}x)" for v in top])
