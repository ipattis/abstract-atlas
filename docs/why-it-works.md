# Why this works for academic abstracts

This is what the NeurIPS and SCALE projects learned. Numbers are from those runs and will differ on your corpus.

## 1. The input is unusually well-suited

An abstract is 150-300 words, self-contained, and written to state the problem, method and result. That means:
extraction is a reading task rather than a research task, per-paper cost is small (a few hundred tokens in and out),
and you can check every output against a source short enough for a person, or a second model, to read in seconds.

## 2. Ask for facts in a schema, not prose

Each field has a type, a description, and (for categories) a closed list of options. The model answers through a
JSON schema (a forced tool call on Claude, `response_format: json_schema` on local servers), so:

- output parses every time or the record is rejected and retried, never silently stored;
- "not stated" is explicitly allowed, which removes the pressure to fill gaps;
- definitions go in the field description. Most of the SCALE accuracy gains came from sharpening definitions
  ("pre-post needs measurement before *and* after; a post-only survey is correlational"), not from a bigger model.

## 3. Separate extraction from judgement

The model extracts `study_type`, `outcomes`, `direction`. A short rule in the config turns those into an evidence
tier. Reasons: it is auditable, it is reproducible across models, you can change your mind without re-running the LLM,
and nobody has to argue about what the model "thinks" of a paper's quality.

## 4. Audit everything, then repair narrowly

`atlas audit` has a judge compare each paper's fields with its abstract. Free text is rated faithful / minor issue /
unsupported; categories are rated correct / debatable / wrong. The loop:

1. Run the audit; read `audit_report.md`.
2. Find a field with a high unsupported rate (NeurIPS "applications": 20.5%).
3. Tighten that field's prompt, bump `extraction.prompt_version`.
4. `atlas enrich --refresh-fields applications`, then `atlas audit` again (only changed records are re-judged).

After the fix, 0.2% of applications were unsupported, and across 600 audited NeurIPS papers 0.3% had any unsupported
field. Because the refresh touches one field, you pay for one field, and the other fields stay stable.

## 5. Multiple views from one model

Embedding "title + abstract" gives a topic map. Embedding only the extracted *problem* sentence (with the instruction
"identify the problem this paper addresses") gives a problem map; the extracted *method* sentence gives a method map.
The extraction step is what makes the second and third views possible: you embed the LLM's distilled field, not the
whole abstract.

## 6. Trends need care

- All periods share one layout, so regions are comparable over time.
- Compare **shares**, not counts, because the newest period is usually partial.
- Smooth with +1 so a topic going from 0 to 2 papers is not "infinite growth", and label topics with too few
  recent papers "Small" instead of ranking them.
- Let the LLM describe a change (from sampled papers) but never compute it.

## 7. Operate it like a data pipeline

Everything is cached by content: extraction by paper id and prompt version, embeddings by hash of model + instruction +
text, batch jobs by a fingerprint of the records and prompts. A monthly re-run touches only new papers.

## Limits

- Abstracts under-describe studies. In SCALE, study type was "debatable" for 42% of papers. Treat labels as a way to sort, not a verdict.
- Same-family judge: a Claude judge of Claude extraction shares blind spots. Prefer a different or stronger judge, and hand-check a sample.
- Reported results lean positive because abstracts do. Say so in anything you publish from a map.
- Operational notes from the original runs (account-specific, may have changed): some models were slow to start batch jobs
  (Haiku 4.5 sat "Scheduled" for 3+ hours while Sonnet 4.6 started in minutes); some newer models do not support Bedrock
  batch inference; some newer Claude models reject forced `tool_choice`, which the Claude backend probes for and handles.
  Check your own account and region.
