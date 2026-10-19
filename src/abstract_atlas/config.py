"""Run configuration (one YAML file per use case) and the on-disk layout of a run."""
from dataclasses import dataclass
from pathlib import Path

import yaml

from .schema import Field, fields_from_config

DEFAULT_PROMPT = """Read the paper below and record your analysis with the tool. Be concrete and faithful to the
abstract: do not invent results, numbers or names, and prefer "not stated" over a guess.

Title: {title}

Abstract: {abstract}"""


@dataclass
class Config:
    raw: dict
    path: Path

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        return cls(yaml.safe_load(path.read_text()), path)

    # --- layout -------------------------------------------------------------------------------------------
    def _rel(self, p):
        p = Path(p).expanduser()
        return p if p.is_absolute() else (self.path.parent / p).resolve()

    @property
    def name(self):
        return self.raw.get("name", self.path.stem)

    @property
    def workdir(self):
        d = self._rel(self.raw.get("workdir", f"runs/{self.name}"))
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def input(self):
        return self.raw["input"]

    @property
    def input_path(self):
        return self._rel(self.input["path"])

    # --- extraction ---------------------------------------------------------------------------------------
    @property
    def extraction(self):
        return self.raw["extraction"]

    @property
    def fields(self) -> list[Field]:
        return fields_from_config(self.extraction["fields"])

    @property
    def prompt(self):
        return self.extraction.get("prompt", DEFAULT_PROMPT)

    @property
    def system(self):
        return self.extraction.get("instructions")

    @property
    def prompt_version(self):
        return int(self.extraction.get("prompt_version", 1))

    @property
    def derived(self):
        return self.raw.get("derived", {})

    # --- views / topics / trends --------------------------------------------------------------------------
    @property
    def views(self):
        return self.raw["views"]

    @property
    def embedding(self):
        return {"model": "Qwen/Qwen3-Embedding-4B", "instruct": True, "batch_size": 16, **self.raw.get("embedding", {})}

    @property
    def topics(self):
        return {"min_clusters": 5, "max_clusters": 120, "max_layers": 3, "base_min_cluster_size": 5, **self.raw.get("topics", {})}

    @property
    def trends(self):
        return {"pool_through": None, "min_latest": 6, "note_model_calls": True, **self.raw.get("trends", {})}

    @property
    def audit(self):
        return self.raw.get("audit", {})

    @property
    def map(self):
        return {"title": self.name, "subtitle": "Zoom for detail, hover to preview.", **self.raw.get("map", {})}

    # --- backends -----------------------------------------------------------------------------------------
    def backend_spec(self, role, override=None):
        """The backend config for a role ('extract', 'audit', 'naming', 'trends')."""
        backends = self.raw.get("backends", {})
        name = override or self.raw.get("use", {}).get(role) or self.raw.get("use", {}).get("default")
        if name is None and len(backends) == 1:
            name = next(iter(backends))
        if name not in backends:
            if override in {"mock"}:
                return {"type": "mock"}
            raise SystemExit(f"No backend {name!r} for role {role!r}. Define it under `backends:` and select it under `use:`.")
        return {"name": name, **backends[name]}
