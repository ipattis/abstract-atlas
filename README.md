# abstract-atlas

Turn a pile of research abstracts into an **audited, explorable map**: a zoomable 2D landscape of the papers, named
topics, trends over time, and a structured record for every paper that says *what kind of evidence it reports*.

It is a generalised, runnable version of the method behind two projects:

- **NeurIPS maps**: all 19,493 accepted NeurIPS 2024-2026 papers, mapped three ways (topic, problem, method).
- **SCALE maps**: 1,652 papers on AI in education, each tagged with an evidence profile.

The same pipeline runs **locally** (Ollama, llama.cpp, vLLM, LM Studio: anything OpenAI-compatible) or as a
**batch job on AWS Bedrock** (Claude, about half the on-demand price). Swap with one line of config.

> **Credit.** This began as a rebuild of Jay Alammar's
> [*The Illustrated NeurIPS 2025*](https://newsletter.languagemodels.co/p/the-illustrated-neurips-2025-a-visual)
> (embed abstracts, have an LLM annotate them, UMAP + cluster, name the clusters, render with
> [datamapplot](https://github.com/TutteInstitute/datamapplot)). It uses the Tutte Institute's
> [datamapplot](https://github.com/TutteInstitute/datamapplot), [EVoC](https://github.com/TutteInstitute/evoc)
> and [Toponymy](https://github.com/TutteInstitute/toponymy), and [UMAP](https://github.com/lmcinnes/umap).

## Why it works for abstracts

Abstracts are a nearly ideal LLM input. Short, self-contained, written to be summarised, and available for almost
everything. The method leans on that:

| Idea | What it buys you |
|---|---|
| **Extract with a schema, from the abstract only** | Every paper gets the same typed fields (summary, problem, method, study type, sample size...). Output is machine-checkable, and "not stated" is always a legal answer, so the model has no reason to guess. |
| **LLM extracts facts, a published rule makes the category** | The model never decides "evidence tier" or "is this strong?". A short YAML rule maps extracted facts to a tier. You can read it, argue with it, and change it, and every paper is re-labelled for free. |
| **Audit every paper, then fix the prompt** | Abstracts are cheap enough to judge all of them. A field that fails shows up as a rate, you tighten that field's prompt and regenerate *only that field*. In the NeurIPS run, an "applications" field was 20.5% unsupported (the prompt invited invented use cases); one prompt fix and a refresh brought it to 0.2%. |
| **Several maps of the same papers** | One embedding model plus a different *task instruction* per view gives a map by topic, by problem, and by method. In the problem map, different techniques for the same goal sit together; in the method map, one technique applied to different problems does. |
| **One shared layout across years** | All periods are laid out together, so a region means the same thing every year and the year filter shows the field moving. |
| **Trends from counts, not vibes** | Growth is a topic's *share* of papers in the latest period vs the previous (smoothed; partial years handled; tiny topics labelled "Small"). The LLM only writes the "what's new" note. |
| **Incremental and cached** | Every step skips what is already done. A monthly refresh pays only for new papers. Interrupted batch jobs re-attach instead of resubmitting. |
| **Cheap** | In the NeurIPS runs, enriching 9,099 papers with Claude Sonnet 4.6 on Bedrock batch cost about $52. Use `atlas estimate` to size your own corpus before spending anything. |

More detail, including the limits: [docs/why-it-works.md](docs/why-it-works.md).

## Quickstart (about 10 minutes, no cloud needed)

```bash
git clone <this repo> && cd abstract-atlas
python -m venv .venv && source .venv/bin/activate
pip install -e ".[embed,map]"            # add ,aws for Bedrock

# 1. Smoke test with no LLM at all (placeholder text; checks your install end to end)
atlas -c configs/quickstart.yaml --backend mock run

# 2. For real, with a local model: start Ollama and pull any 7-14B instruct model
ollama pull qwen3:14b                    # then set that tag under backends.local.model in the config
atlas -c configs/quickstart.yaml estimate
atlas -c configs/quickstart.yaml run     # enrich, embed, topics, trends, audit, map
open runs/quickstart/site/index.html
```

The bundled `examples/synthetic_abstracts.csv` is **fake, templated text** for smoke tests. Point `input.path` at
your own csv / json / jsonl / parquet with an id, title and abstract (and a year for trends).

## Choose where the LLM runs

| | Local | AWS Bedrock batch |
|---|---|---|
| Setup | any OpenAI-compatible server | `scripts/aws_setup.sh`, Bedrock model access |
| Cost | your hardware and time | per token, ~50% off on-demand |
| Best for | privacy, tinkering, small to mid corpora | thousands of papers, strongest extraction |
| Guide | [docs/local-inference.md](docs/local-inference.md) | [docs/aws-batch.md](docs/aws-batch.md) |

Embedding always runs locally (sentence-transformers, default Qwen3-Embedding; a 0.6B model works on a laptop).
Mix backends per role in the config, e.g. batch Claude for `extract` and `audit`, a stronger on-demand model for `naming`.

## Pipeline

```
atlas enrich    structured extraction from each abstract            -> enrichment.jsonl
atlas embed     one embedding per paper per view (task-instructed)  -> emb_<view>.npz
atlas topics    shared UMAP layout + EVoC clusters + Toponymy names -> map_<view>.parquet
atlas trends    topic share by period + "what's new" notes          -> trends_<view>.json
atlas audit     LLM judge checks every paper's fields               -> audit.parquet, audit_report.md
atlas map       interactive HTML per view (datamapplot)             -> site/*.html
atlas estimate  token and cost estimate for `enrich`, before spending
```

All outputs go under `workdir` (default `runs/<name>/`). `atlas run --skip audit,trends` runs a subset.

## Make it yours

A use case is one YAML file: the **fields** to extract, the **rules** that derive categories, the **views** to
map, and the **backends**. See [docs/extending.md](docs/extending.md). Ready-made configs:

- `configs/quickstart.yaml`: generic summary / problem / method / contribution.
- `configs/education_evidence.yaml`: evidence profile (study type, outcomes, direction, sample size) and a rule-based evidence tier. The SCALE schema.
- `configs/ml_conference.yaml`: proceedings-style maps. The NeurIPS schema.

## Honest limits

- The LLM reads **abstracts only**. It records what an abstract *says*, not whether the study is good. Evidence tiers are a sorting aid, not a quality score.
- In the SCALE run the judge flagged a field on 5.9% of papers, and study type was "debatable" for 42%, because abstracts under-describe designs. Expect the same.
- If the audit judge is the same model family as the extractor, the audit is a consistency check, not independent verification.
- Small local models make more extraction mistakes. The audit is how you find out; run it before trusting a map.
- The built-in map is datamapplot's standard interface (search, topic tree, hover). The NeurIPS and SCALE sites add a custom toolbar, agenda and in-browser "where does my paper fit?"; those are not part of this package.

## Development

```bash
pip install -e ".[dev]" && pytest        # unit tests need no models, network or AWS
```

## License

MIT. Papers, titles and abstracts belong to their authors; respect the terms of whichever source you fetch from.
AI-written fields can be wrong: read the paper before citing it.
