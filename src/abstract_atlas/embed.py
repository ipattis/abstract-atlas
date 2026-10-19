"""Step 2: one embedding per paper *per view*.

A view is a text template plus a task instruction, e.g. "Identify the problem this paper addresses". The same
embedding model, steered by a different instruction, puts papers with the same goal next to each other (problem
view) or papers that use the same technique next to each other (method view). Embeddings are cached by a hash of
(model, instruction, text), so re-runs embed only what changed.
"""
import hashlib
from collections import defaultdict

import numpy as np

from .config import Config
from .data import load_mapped


def instruct(task):
    return f"Instruct: {task}\nQuery: "


def view_texts(cfg: Config, papers, enrich, view):
    template = cfg.views[view]["text"]
    out = []
    for row in papers.to_dict("records"):
        ctx = defaultdict(str, {**row, **enrich.get(row["pid"], {})})
        out.append(template.format_map(ctx).strip())
    return out


def cache_key(model_name, prompt, text):
    return hashlib.sha1(f"{model_name}\x00{prompt}\x00{text}".encode()).hexdigest()


def embed_cached(path, texts, encode, model_name, prompt):
    """encode(list[str]) -> array. Returns vectors in `texts` order, computing only cache misses."""
    cache = {}
    if path.exists():
        z = np.load(path, allow_pickle=False)
        cache = dict(zip(z["keys"], z["vectors"]))
    keys = [cache_key(model_name, prompt, t) for t in texts]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    if todo:
        if any(not texts[i] for i in todo):
            raise SystemExit(f"Some papers have empty text for {path.name}; run `atlas enrich` first")
        vecs = np.asarray(encode([texts[i] for i in todo]), dtype=np.float32)
        for i, v in zip(todo, vecs):
            cache[keys[i]] = v
        np.savez(path, keys=np.array(list(cache)), vectors=np.stack(list(cache.values())))
    print(f"{path.name}: {len(todo)} new, {len(texts)} total")
    return np.stack([cache[k] for k in keys])


def load_model(name):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(name, device=device, model_kwargs={"torch_dtype": torch.float16 if device != "cpu" else torch.float32})
    model.max_seq_length = 512
    return model


def view_path(cfg, view):
    return cfg.workdir / f"emb_{view}.npz"


def run(cfg: Config, views=None, model=None):
    papers, enrich = load_mapped(cfg)
    emb = cfg.embedding
    model = model or load_model(emb["model"])
    for view in views or list(cfg.views):
        prompt = instruct(cfg.views[view]["instruction"]) if emb["instruct"] else ""
        texts = view_texts(cfg, papers, enrich, view)
        embed_cached(view_path(cfg, view), texts,
                     lambda batch: model.encode(batch, prompt=prompt or None, batch_size=emb["batch_size"],
                                                normalize_embeddings=True, show_progress_bar=True),
                     emb["model"], prompt)


def load_view(cfg, view, papers, enrich):
    """Vectors for `papers` in order (looked up by text hash, so stale cache entries are never used)."""
    emb = cfg.embedding
    prompt = instruct(cfg.views[view]["instruction"]) if emb["instruct"] else ""
    z = np.load(view_path(cfg, view), allow_pickle=False)
    lookup = dict(zip(z["keys"], z["vectors"]))
    keys = [cache_key(emb["model"], prompt, t) for t in view_texts(cfg, papers, enrich, view)]
    missing = sum(k not in lookup for k in keys)
    if missing:
        raise SystemExit(f"{missing} papers lack current {view} embeddings; run `atlas embed`")
    return np.stack([lookup[k] for k in keys])
