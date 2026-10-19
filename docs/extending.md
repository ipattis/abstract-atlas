# Extending abstract-atlas

A use case is one YAML file. Nothing in the Python is specific to education or machine learning.

## 1. Input

```yaml
input: {path: data/papers.parquet, id: id, title: title, abstract: abstract, year: year}
```

csv / json / jsonl / parquet. Extra columns (venue, authors, ...) are available to prompts and view templates as
`{column}`. Abstracts shorter than `min_abstract_chars` (default 100) are skipped.

## 2. Fields

```yaml
extraction:
  prompt_version: 1
  instructions: "System prompt: role, grounding rules, how to say 'not stated'."
  prompt: "Title: {title}\nAbstract: {abstract}"      # optional; this is the default shape
  fields:
    - {name: summary, type: text, description: "..."}
    - {name: design,  type: enum, options: [a, b, unclear], audit: label}
    - {name: outcomes, type: enum_list, options: [x, y]}
    - {name: n, type: int}   # also: bool, list
```

Types: `text`, `enum`, `enum_list`, `list`, `int`, `bool`. `audit:` is `text` (default for text), `label` (judged
correct / debatable / wrong) or `skip`. Put definitions and edge cases in `description`; that is where accuracy comes from.

## 3. Derived labels

```yaml
derived:
  evidence_tier:
    default: Unclear
    rules:
      - {if: {design: {startswith: [Randomized]}, outcomes: {contains: x}}, then: Strong}
```

Operators: `equals`, `in`, `startswith`, `contains`, `contains_any`, `gte`, `lte`, `not_empty`. First match wins.
Derived labels appear in the map's hover text via `map.meta_fields`; they are recomputed on read, so editing a rule needs no LLM calls.

## 4. Views

```yaml
views:
  topic:  {title: By topic,  text: "{title}. {abstract}", instruction: "Identify the research topic of this paper"}
  method: {title: By method, text: "{method}",            instruction: "Identify the technique this paper uses"}
```

`text` can use input columns and extracted fields. `instruction` steers instruction-tuned embedding models (Qwen3-Embedding
style). For models without instructions set `embedding.instruct: false`. The first view becomes `index.html`.

## 5. Ideas that fit the same shape

- **Clinical or social-science evidence maps**: add `population`, `intervention`, `comparator`, `outcome`, `risk_of_bias_signals` enums; derive an evidence grade by rule.
- **Grant portfolios or thesis abstracts**: views by funder priority and by method.
- **Literature review triage**: add a `relevant` enum against your inclusion criteria, audit it hard, and screen by rule.
- **Preprint monitoring**: schedule `atlas run` monthly; only new papers cost anything.

## 6. Adding a backend

Subclass `abstract_atlas.backends.base.Backend`: implement `run({id: Request}) -> {id: {"output", "usage"} | {"error"}}` and
`chat_text(system, user, max_tokens)`, then register it in `backends/__init__.py::make_backend`.
