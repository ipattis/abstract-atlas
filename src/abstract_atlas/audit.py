"""Step 5: audit every paper's extraction with an LLM judge.

The judge sees the abstract and the generated fields and rates each one: free text as faithful / minor_issue /
unsupported, categorical labels as correct / debatable / wrong. Abstracts are short, so auditing ALL papers is
cheap; sampling is only needed for very large corpora (`audit.sample: N`). The audit is what makes the fix loop
work: a bad field shows up as a rate, you tighten the prompt, bump `prompt_version`, run
`atlas enrich --refresh-fields <field>` and re-audit.

Caveat: if the judge is the same model family as the extractor, this is a consistency check, not independent
verification. Use a different (ideally stronger) model as judge when you can.
"""
import pandas as pd

from .backends import Request
from .config import Config
from .data import load_mapped
from .schema import Field

TEXT_RATINGS = ["faithful", "minor_issue", "unsupported"]
LABEL_RATINGS = ["correct", "debatable", "wrong"]

PROMPT = """Check generated notes about a research paper against its abstract.
Text fields:
- faithful: everything stated is supported by (or a fair simplification of) the abstract
- minor_issue: slight overstatement or vague imprecision, not misleading
- unsupported: states a result, method, number, recommendation or claim that the abstract does not support
Labels: correct / debatable (a reasonable reader could choose otherwise) / wrong.
{guidance}
Title: {title}
Abstract: {abstract}

Generated:
{generated}"""


def audited_fields(fields):
    text = [f for f in fields if f.audit == "text"]
    label = [f for f in fields if f.audit == "label"]
    return text, label


def judge_schema(text: list[Field], label: list[Field]):
    props = {f.name: {"type": "string", "enum": TEXT_RATINGS} for f in text}
    props |= {f.name: {"type": "string", "enum": LABEL_RATINGS} for f in label}
    props["issues"] = {"type": "array", "items": {"type": "string"},
                       "description": "Each specific claim or label the abstract does not support (empty if none)."}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def run(cfg: Config, backend):
    papers, enrich = load_mapped(cfg)
    text, label = audited_fields(cfg.fields)
    schema = judge_schema(text, label)
    path = cfg.workdir / "audit.parquet"
    previous = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["pid", "enrich_version"])
    seen = dict(zip(previous["pid"], previous["enrich_version"]))
    version = {p: enrich[p].get("prompt_version", 1) for p in papers["pid"]}
    todo = papers[papers["pid"].map(lambda p: seen.get(p) != version[p])]   # re-audit anything refreshed since
    if cfg.audit.get("sample"):
        todo = todo.sample(min(int(cfg.audit["sample"]), len(todo)), random_state=0)

    requests = {}
    for row in todo.to_dict("records"):
        gen = "\n".join(f"{f.name}: {enrich[row['pid']].get(f.name, 'Not stated')}" for f in text + label)
        requests[row["pid"]] = Request(
            PROMPT.format(guidance=cfg.audit.get("guidance", ""), title=row["title"], abstract=row["abstract"], generated=gen),
            schema, "record_audit", "Record whether each generated field is faithful to the abstract.", max_tokens=800)
    print(f"Auditing {len(requests)} papers with {backend.name}")
    rows = [{"pid": pid, **res["output"], "enrich_version": version[pid]}
            for pid, res in (backend.run(requests, job="audit") if requests else {}).items() if "output" in res]
    audit = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("pid", keep="last")
    audit["issues"] = audit["issues"].apply(lambda x: list(x) if x is not None else [])
    audit.to_parquet(path)
    report(cfg, audit, papers, text, label, backend.name)


def report(cfg, audit, papers, text, label, judge):
    audit = audit.merge(papers[["pid", "title"]], on="pid")
    n = max(len(audit), 1)
    lines = [f"# Extraction audit: {cfg.name}\n",
             f"Judge: `{judge}`. {len(audit)} of {len(papers)} papers audited.\n",
             "If the judge is the same model family as the extractor, treat this as a consistency check, not independent verification.\n",
             "| Field | Faithful | Minor issue | Unsupported |", "|---|---|---|---|"]
    for f in text:
        vc = audit[f.name].value_counts(normalize=True)
        lines.append(f"| {f.name} | {vc.get('faithful', 0):.1%} | {vc.get('minor_issue', 0):.1%} | {vc.get('unsupported', 0):.1%} |")
    if label:
        lines += ["", "| Label | Correct | Debatable | Wrong |", "|---|---|---|---|"]
        for f in label:
            vc = audit[f.name].value_counts(normalize=True)
            lines.append(f"| {f.name} | {vc.get('correct', 0):.1%} | {vc.get('debatable', 0):.1%} | {vc.get('wrong', 0):.1%} |")
    bad = pd.Series(False, index=audit.index)
    for f in text:
        bad |= audit[f.name] == "unsupported"
    for f in label:
        bad |= audit[f.name] == "wrong"
    lines.append(f"\n**Papers with at least one unsupported field or wrong label: {bad.sum()} / {len(audit)} ({bad.sum() / n:.1%}).**\n")
    lines.append("## Examples of flagged issues\n")
    for row in audit[bad].head(20).itertuples():
        lines.append(f"- *{row.title}*: " + "; ".join(row.issues[:2]))
    (cfg.workdir / "audit_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
