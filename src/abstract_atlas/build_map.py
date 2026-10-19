"""Step 6: render each view as a self-contained interactive HTML map with datamapplot."""
import html

import pandas as pd

from .config import Config
from .data import load_mapped
from .rules import derive_all

HOVER = """<div style="max-width:340px;white-space:normal;font-size:12px">
<b>{hover_text}</b><br><span style="color:#666">{meta}</span><br>{blurb}</div>"""


def run(cfg: Config, views=None):
    import datamapplot

    papers, enrich = load_mapped(cfg)
    site = cfg.workdir / "site"
    site.mkdir(exist_ok=True)
    audit_path = cfg.workdir / "audit.parquet"
    flagged = set()
    if audit_path.exists():
        a = pd.read_parquet(audit_path)
        cols = [c for c in a.columns if c not in {"pid", "issues", "enrich_version"}]
        flagged = set(a.loc[a[cols].isin(["unsupported", "wrong"]).any(axis=1), "pid"])
    shown = cfg.map.get("hover_fields", ["summary"])
    meta_fields = cfg.map.get("meta_fields", [])
    for view in views or list(cfg.views):
        m = pd.read_parquet(cfg.workdir / f"map_{view}.parquet").merge(papers, on="pid")
        layer_cols = sorted((c for c in m.columns if c.startswith("L") and c.endswith("_name")), key=lambda c: -int(c[1:-5]))
        names = [m[c].to_numpy() for c in layer_cols]   # datamapplot wants coarsest first
        meta, blurb = [], []
        for pid, year in zip(m["pid"], m.get("year", pd.Series([""] * len(m)))):
            rec = {**enrich[pid], **derive_all(enrich[pid], cfg.derived)}
            parts = [str(year)] if str(year) not in {"", "nan"} else []
            for f in meta_fields:
                v = rec.get(f)
                parts.append(", ".join(v) if isinstance(v, list) else str(v))
            if pid in flagged:
                parts.append("⚠ audit flagged a field")
            meta.append(html.escape(" · ".join(p for p in parts if p)))
            blurb.append(html.escape(" ".join(str(rec.get(f, "")) for f in shown)))
        extra = pd.DataFrame({"meta": meta, "blurb": blurb})
        plot = datamapplot.create_interactive_plot(
            m[["x", "y"]].to_numpy(), *names,
            hover_text=m["title"].to_numpy(),
            hover_text_html_template=HOVER,
            extra_point_data=extra,
            title=f"{cfg.map['title']}: {cfg.views[view].get('title', view)}",
            sub_title=f"{len(m):,} papers. {cfg.map['subtitle']}",
            enable_search=True,
            enable_topic_tree=True,
            font_family="Roboto",
            cluster_boundary_polygons=True,
        )
        out = site / ("index.html" if view == list(cfg.views)[0] else f"{view}.html")
        plot.save(str(out))
        out.write_text(out.read_text().replace("<head>", '<head>\n<meta name="viewport" content="width=device-width, initial-scale=1" />', 1))
        print(f"{view}: saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
