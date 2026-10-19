"""Glue for naming clusters with Toponymy, EVoC clusters and any Backend (Claude or a local model)."""
import asyncio
import hashlib

import numpy as np
from toponymy.cluster_layer import ClusterLayerText
from toponymy.clustering import Clusterer, build_cluster_tree, centroids_from_labels
from toponymy.llm_wrappers import AsyncLLMWrapper


class BackendNamer(AsyncLLMWrapper):
    """Toponymy LLM wrapper that calls Backend.chat_text in worker threads."""

    def __init__(self, backend, max_concurrent_requests=8):
        self.backend = backend
        self.callback = None
        self.max_concurrent_requests = max_concurrent_requests
        self._loop = None

    def _bind_loop(self):
        # Toponymy runs each layer in a fresh event loop; asyncio primitives must be created inside it.
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            self._loop = loop
            self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)

    async def _create(self, user, max_tokens, system=None):
        self._bind_loop()
        async with self.semaphore:
            return await asyncio.to_thread(self.backend.chat_text, system, user, max_tokens)

    async def _call_single_llm(self, prompt, temperature, max_tokens):
        return await self._create(prompt["combined"], max_tokens)

    async def _call_single_llm_with_system(self, prompt, temperature, max_tokens):
        return await self._create(prompt["user"], max_tokens, prompt["system"])


class EVoCClusterer(Clusterer):
    """Data-driven multi-resolution clusters from EVoC on the full embedding vectors.

    Keeps up to `max_layers` layers with between `min_clusters` and `max_clusters` clusters (finest first), so
    the number of topics per zoom level comes from the data. EVoC leaves many points as noise; they are assigned
    to the majority cluster of their nearest labelled neighbours so every paper lands in a topic.
    """

    def __init__(self, min_clusters=5, max_clusters=120, max_layers=3, base_min_cluster_size=5, random_state=42,
                 assign_noise_k=15):
        self.min_clusters, self.max_clusters, self.max_layers = min_clusters, max_clusters, max_layers
        self.base_min_cluster_size, self.random_state, self.assign_noise_k = base_min_cluster_size, random_state, assign_noise_k

    def fit(self, clusterable_vectors, embedding_vectors, layer_class=ClusterLayerText, verbose=None,
            show_progress_bar=None, **layer_kwargs):
        import evoc

        model = evoc.EVoC(base_min_cluster_size=self.base_min_cluster_size, random_state=self.random_state)
        model.fit(embedding_vectors)
        n_of = lambda labels: len(set(labels.tolist()) - {-1})  # noqa: E731
        layers = [l for l in model.cluster_layers_ if self.min_clusters <= n_of(l) <= self.max_clusters]
        if not layers:
            raise SystemExit("EVoC found no layers in the allowed cluster range; lower topics.min_clusters or add papers")
        if len(layers) > self.max_layers:
            idx = np.unique(np.linspace(0, len(layers) - 1, self.max_layers).round().astype(int))
            layers = [layers[i] for i in idx]
        print("EVoC layers (clusters per layer, finest first):", [n_of(l) for l in layers])
        if self.assign_noise_k:
            layers = [self._assign_noise(l, embedding_vectors) for l in layers]
        self.cluster_tree_ = build_cluster_tree(layers)
        self.cluster_layers_ = [layer_class(labels, centroids_from_labels(labels, embedding_vectors), layer_id=i, **layer_kwargs)
                                for i, labels in enumerate(layers)]
        return self

    def _assign_noise(self, labels, vectors):
        from sklearn.neighbors import NearestNeighbors

        labels = labels.copy()
        noise = labels < 0
        if not noise.any() or noise.all():
            return labels
        k = min(self.assign_noise_k, int((~noise).sum()))
        _, idx = NearestNeighbors(n_neighbors=k, metric="cosine").fit(vectors[~noise]).kneighbors(vectors[noise])
        labels[noise] = [np.bincount(row).argmax() for row in labels[~noise][idx]]
        return labels

    def fit_predict(self, clusterable_vectors, embedding_vectors, layer_class=ClusterLayerText, verbose=None,
                    show_progress_bar=None, **layer_kwargs):
        self.fit(clusterable_vectors, embedding_vectors, layer_class, verbose, show_progress_bar, **layer_kwargs)
        return self.cluster_layers_, self.cluster_tree_


class InstructedEmbedder:
    """Embeds Toponymy's keyphrases with the same task instruction as the papers, with an on-disk cache."""

    def __init__(self, encode, prompt, cache_path=None):
        self.encode_fn, self.prompt, self.cache_path = encode, prompt, cache_path
        self.cache = {}
        if cache_path and cache_path.exists():
            z = np.load(cache_path, allow_pickle=False)
            self.cache = dict(zip(z["keys"], z["vectors"]))

    def _key(self, text):
        return hashlib.sha1((self.prompt + "\x00" + text).encode()).hexdigest()

    def encode(self, texts, show_progress_bar=False, **kwargs):
        keys = [self._key(t) for t in texts]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        if todo:
            vecs = self.encode_fn([texts[i] for i in todo])
            for i, v in zip(todo, np.asarray(vecs, dtype=np.float32)):
                self.cache[keys[i]] = v
            if self.cache_path and len(todo) > 100:
                np.savez(self.cache_path, keys=np.array(list(self.cache)), vectors=np.stack(list(self.cache.values())))
        return np.stack([self.cache[k] for k in keys]).astype(np.float32)
