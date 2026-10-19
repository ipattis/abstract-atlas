# Reproduce it with local inference

Any server that exposes `POST /v1/chat/completions` and honours `response_format: {type: json_schema}` works:
[Ollama](https://ollama.com), [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server`, [vLLM](https://github.com/vllm-project/vllm),
LM Studio, SGLang. Schema-constrained decoding is what makes small models usable here: the server can only emit valid JSON
with allowed enum values.

## Steps

```bash
pip install -e ".[embed,map]"
ollama pull qwen3:14b          # or any instruct model your hardware can run
```

In your config:

```yaml
backends:
  local:
    type: local
    base_url: http://localhost:11434/v1      # Ollama; llama-server is usually :8080/v1, vLLM :8000/v1
    model: qwen3:14b                          # the name your server uses
    workers: 2                                # parallel requests; raise it if the server batches well
use: {default: local}
```

```bash
atlas -c my.yaml enrich --limit 20      # try a handful first and read the output
atlas -c my.yaml audit                  # then judge them
atlas -c my.yaml run                    # everything
```

## Practical advice

- **Start with `--limit 20`** and read `runs/<name>/enrichment.jsonl`. Fix field descriptions before processing the whole corpus.
- **Model size matters most for categorical fields** (study design, direction). Free-text summaries degrade gracefully; closed-list labels need a capable model. The audit tells you which you have.
- **Reasoning models** emit `<think>` blocks; the backend strips them, but they cost time. Turn thinking off in the server if you can.
- **Speed.** Roughly: `papers x (a few hundred output tokens) / tokens-per-second`. Measure on 20 papers and multiply.
- **Naming clusters** (`topics`) benefits from a stronger model than extraction. You can route it elsewhere: `use: {default: local, naming: bedrock}`.
- **Embeddings** run in-process with sentence-transformers on MPS/CUDA/CPU. On a laptop set `embedding.model: Qwen/Qwen3-Embedding-0.6B`.
- **Privacy.** With the local backend and local embeddings, no text leaves your machine.

Not covered by automated tests: the code is tested against a fake OpenAI-compatible server, not against a real model.
If your server rejects `strict: true` or `json_schema`, open an issue with its name and version.
