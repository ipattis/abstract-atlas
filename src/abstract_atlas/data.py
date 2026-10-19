"""Reading papers and the append-only enrichment store."""
import json

import pandas as pd

from .config import Config
from .schema import validate_record


def read_table(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise SystemExit(f"Unsupported input format {suffix!r}: use csv, json, jsonl or parquet")


def period_of(year, pool_through):
    if year is None or pd.isna(year):
        return "Undated"
    year = int(year)
    return f"≤{pool_through}" if pool_through is not None and year <= pool_through else str(year)


def load_papers(cfg: Config) -> pd.DataFrame:
    """Input table -> columns pid, title, abstract, period, plus any other columns the template can use."""
    spec = cfg.input
    df = read_table(cfg.input_path)
    cols = {spec.get("id", "id"): "pid", spec.get("title", "title"): "title", spec.get("abstract", "abstract"): "abstract"}
    if spec.get("year"):
        cols[spec["year"]] = "year"
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{cfg.input_path.name} lacks columns {missing}; set them under `input:` in the config")
    df = df.rename(columns=cols)
    df["pid"] = df["pid"].astype(str)
    if df["pid"].duplicated().any():
        raise SystemExit("Paper ids must be unique")
    df["abstract"] = df["abstract"].fillna("").astype(str)
    df = df[df["abstract"].str.len() >= int(spec.get("min_abstract_chars", 100))].reset_index(drop=True)
    pool = cfg.trends["pool_through"]
    df["period"] = df["year"].map(lambda y: period_of(y, pool)) if "year" in df else "Undated"
    return df


def enrichment_path(cfg):
    return cfg.workdir / "enrichment.jsonl"


def load_enrichment(cfg: Config) -> dict:
    """{pid: record}. The file is append-only; later lines (refreshes) win."""
    out, path = {}, enrichment_path(cfg)
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    out[rec["pid"]] = rec
    return out


def load_mapped(cfg: Config):
    """(papers, enrichment) for papers that have been enriched. Every step after `enrich` uses this so all outputs
    share one paper order."""
    enrich = load_enrichment(cfg)
    df = load_papers(cfg)
    return df[df["pid"].isin(enrich)].reset_index(drop=True), enrich


def is_complete(rec, cfg: Config):
    try:
        validate_record(rec, cfg.fields)
        return True
    except ValueError:
        return False
