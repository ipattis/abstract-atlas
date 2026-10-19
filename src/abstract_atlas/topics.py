"""Step 3: a shared 2D layout, a data-driven topic hierarchy and LLM topic names, per view.

1. UMAP lays out ALL papers once, so the same region means the same thing in every year or period.
2. EVoC clusters the full embeddings at several resolutions; the data decides how many clusters.
3. Toponymy names each cluster from exemplar papers, contrastive keyphrases and subtopic names, then
   de-duplicates names across the hierarchy.
Output: map_<view>.parquet with x, y and an id/name column per layer (L0 = finest).
"""
import numpy as np
import pandas as pd

from .config import Config
from .data import load_mapped
from .embed import instruct, load_model, load_view


def layout(emb, seed=42):
    import umap

    return umap.UMAP(n_neighbors=15, min_dist=0.05, metric="cosine", random_state=seed).fit_transform(emb)


def name_objects(cfg, view, papers, enrich):
    from .embed import view_texts

    return view_texts(cfg, papers, enrich, view)


def run(cfg: Config, naming_backend, views=None, st_model=None):
    from toponymy import Toponymy

    from .toponymy_glue import BackendNamer, EVoCClusterer, InstructedEmbedder

    papers, enrich = load_mapped(cfg)
    emb_cfg, t = cfg.embedding, cfg.topics
    st_model = st_model or load_model(emb_cfg["model"])
    for view in views or list(cfg.views):
        print(f"\n=== {view} view ===")
        emb = load_view(cfg, view, papers, enrich)
        coords = layout(emb)
        prompt = instruct(cfg.views[view]["instruction"]) if emb_cfg["instruct"] else ""
        embedder = InstructedEmbedder(
            lambda batch: st_model.encode(batch, prompt=prompt or None, batch_size=32, normalize_embeddings=True),
            prompt, cache_path=cfg.workdir / f"keyphrase_cache_{view}.npz")
        names = cfg.views[view].get("naming", {})
        model = Toponymy(
            llm_wrapper=BackendNamer(naming_backend),
            text_embedding_model=embedder,
            clusterer=EVoCClusterer(t["min_clusters"], t["max_clusters"], t["max_layers"], t["base_min_cluster_size"]),
            object_description=names.get("objects", "paper texts"),
            corpus_description=names.get("corpus", cfg.raw.get("corpus_description", "research papers")),
            lowest_detail_level=0.5,   # finest layer: 3-6 word names; coarsest: 1-4 words. Toponymy's default is too long for labels.
            highest_detail_level=0.8,
            verbose=True,
        )
        model.fit(name_objects(cfg, view, papers, enrich), embedding_vectors=emb, clusterable_vectors=coords)

        out = pd.DataFrame({"pid": papers["pid"], "x": coords[:, 0], "y": coords[:, 1]})
        for i, layer in enumerate(model.cluster_layers_):
            labels = np.asarray(layer.cluster_labels)
            topic_names = np.array(list(layer.topic_names) + ["Unlabelled"], dtype=object)
            out[f"L{i}_id"] = labels
            out[f"L{i}_name"] = topic_names[np.where(labels >= 0, labels, len(layer.topic_names))]
            print(f"  layer {i}: {labels.max() + 1} topics, e.g. {list(layer.topic_names[:4])}")
        out.to_parquet(cfg.workdir / f"map_{view}.parquet")
        print(f"Saved map_{view}.parquet")
