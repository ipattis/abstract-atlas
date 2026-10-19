"""Generate examples/synthetic_abstracts.csv: templated, obviously fake abstracts for smoke tests and demos.

They are NOT real papers. They exist so you can run the whole pipeline in minutes without downloading anything.
"""
import csv
import itertools
import random
from pathlib import Path

rng = random.Random(7)
TASKS = ["rainfall nowcasting", "hurricane track forecasting", "sea-ice concentration mapping", "crop yield prediction",
         "wildfire risk estimation", "air-quality forecasting"]
METHODS = ["a graph neural network", "a diffusion model", "a transformer with sparse attention", "gradient-boosted trees",
           "a physics-informed neural network", "a foundation model fine-tuned with LoRA"]
CONTRIB = [("We propose {m} for {t} and show a {n}% reduction in error over operational baselines.", "new method"),
           ("We evaluate {m} for {t} across {k} regions and find gains concentrated in extreme events.", "empirical study"),
           ("We release a benchmark of {k} labelled cases for {t} and report baselines including {m}.", "benchmark or dataset")]


def make(n=150):
    combos = list(itertools.product(TASKS, METHODS, CONTRIB))
    rng.shuffle(combos)
    rows = []
    for i, (t, m, (tmpl, _)) in enumerate(combos[:n]):
        year = rng.choices([2023, 2024, 2025, 2026], weights=[1, 2, 3, 3])[0]
        # make the later years drift towards two of the methods, so the trends step has something to find
        if year >= 2025 and rng.random() < 0.4:
            m = rng.choice(["a diffusion model", "a foundation model fine-tuned with LoRA"])
        body = tmpl.format(m=m, t=t, n=rng.randint(5, 40), k=rng.randint(3, 30))
        rows.append({
            "id": f"syn-{i:03d}", "year": year, "title": f"Synthetic example {i}: {m} for {t}",
            "abstract": f"Accurate {t} matters for planning and early warning, but existing approaches struggle with rare, "
                        f"high-impact events. {body} All data in this example are synthetic.",
        })
    return rows


if __name__ == "__main__":
    out = Path(__file__).parent / "synthetic_abstracts.csv"
    rows = make()
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")
