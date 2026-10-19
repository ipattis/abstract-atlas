"""atlas: run the pipeline step by step or all at once."""
import argparse

from .backends import make_backend
from .config import Config

STEPS = ["enrich", "embed", "topics", "trends", "audit", "map"]


def backend_for(cfg, role, override):
    return make_backend(cfg.backend_spec(role, override), cfg.workdir)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="atlas", description=__doc__)
    ap.add_argument("-c", "--config", required=True, help="path to the YAML config")
    ap.add_argument("--backend", help="use this backend (a name under `backends:`, or 'mock') for every LLM step")
    sub = ap.add_subparsers(dest="step", required=True)
    e = sub.add_parser("enrich", help="structured extraction from each abstract")
    e.add_argument("--limit", type=int)
    e.add_argument("--refresh-fields", help="comma-separated fields to regenerate for records from an older prompt_version")
    est = sub.add_parser("estimate", help="token and cost estimate for `enrich`, before spending anything")
    est.add_argument("--price-in", type=float, help="USD per million input tokens")
    est.add_argument("--price-out", type=float, help="USD per million output tokens")
    est.add_argument("--batch-discount", type=float, default=0.5)
    for s in ["embed", "topics", "trends", "map"]:
        p = sub.add_parser(s)
        p.add_argument("--views", help="comma-separated subset of views")
    sub.add_parser("audit", help="LLM-judge every paper's extraction")
    run = sub.add_parser("run", help="enrich, embed, topics, trends, audit, map")
    run.add_argument("--skip", default="", help="comma-separated steps to skip")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    views = args.views.split(",") if getattr(args, "views", None) else None

    def do(step):
        if step == "enrich":
            from . import enrich
            refresh = args.refresh_fields.split(",") if getattr(args, "refresh_fields", None) else None
            enrich.run(cfg, backend_for(cfg, "extract", args.backend), getattr(args, "limit", None), refresh)
        elif step == "embed":
            from . import embed
            embed.run(cfg, views)
        elif step == "topics":
            from . import topics
            topics.run(cfg, backend_for(cfg, "naming", args.backend), views)
        elif step == "trends":
            from . import trends
            trends.run(cfg, backend_for(cfg, "trends", args.backend), views)
        elif step == "audit":
            from . import audit
            audit.run(cfg, backend_for(cfg, "audit", args.backend))
        elif step == "map":
            from . import build_map
            build_map.run(cfg, views)

    if args.step == "estimate":
        from . import enrich
        enrich.estimate(cfg, price_in=args.price_in, price_out=args.price_out, batch_discount=args.batch_discount)
    elif args.step == "run":
        for step in STEPS:
            if step not in args.skip.split(","):
                print(f"\n##### {step}")
                do(step)
    else:
        do(args.step)


if __name__ == "__main__":
    main()
