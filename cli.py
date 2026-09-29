"""Command-line interface.

    premortem run examples/saas_startup.yaml
    premortem run my_plan.yaml --out reports/my_plan --sims 20000
    premortem init my_plan.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path

from . import __version__
from . import formatting as fmt
from .analysis import summarize
from .config import load_plan
from .engine import run as run_simulation
from .report import write_report

TEMPLATE = """\
# Premortem business plan. Every assumption can be a number or a range.
# Ranges use {dist: pert, low: worst, mode: most likely, high: best-or-worst}.
company: "My Company"
description: "One line on what the business does."
horizon_months: 36
simulations: 10000
seed: 42
survival_target: 0.80

assumptions:
  starting_cash: 500000
  starting_customers: 20
  arpu:            {dist: pert, low: 40, mode: 55, high: 65}       # $ per customer per month
  gross_margin:    {dist: pert, low: 0.55, mode: 0.65, high: 0.72}
  cac:             {dist: pert, low: 120, mode: 200, high: 400}     # $ to win one customer
  marketing_spend: 15000                                            # $ per month
  marketing_growth: 0.02                                            # budget grows 2% a month
  organic_signups: {dist: pert, low: 5, mode: 20, high: 40}
  monthly_churn:   {dist: pert, low: 0.03, mode: 0.05, high: 0.09}
  fixed_costs:     {dist: normal, mean: 30000, sd: 3000, low: 0}
  fixed_cost_growth: 0.01
  demand_volatility: 0.2

# Optional: a funding round that only closes if revenue hits a milestone.
fundraise:
  month: 15
  amount: 1000000
  mrr_milestone: 30000
  close_probability: 0.6

# Optional: decisions to test against the same futures.
levers:
  - name: "Raise prices 10%"
    changes: {arpu: {scale: 1.10}, monthly_churn: {scale: 1.08}}
  - name: "Cut marketing 30%"
    changes: {marketing_spend: {scale: 0.7}}
"""


def _bar(share: float, width: int = 30) -> str:
    filled = int(round(share * width))
    return "\u2588" * filled + "\u2591" * (width - filled)


def _print_summary(s: dict, paths: dict, seconds: float) -> None:
    w = 64
    print()
    print(f"  PREMORTEM  {s['company']}".ljust(w))
    print("  " + "\u2500" * (w - 2))
    print(f"  Survival over {s['horizon_months']} months   {_bar(s['survival_rate'])}  "
          f"{fmt.pct(s['survival_rate'])}  ({s['verdict']})")
    if s["median_death_month"]:
        print(f"  Median time of death       month {s['median_death_month']:.0f}")
    killers = sorted([d for d in s["drivers"] if d["share_of_deaths"] > 0],
                     key=lambda d: -d["share_of_deaths"])[:3]
    if killers:
        print("\n  Top causes of death")
        for d in killers:
            print(f"    {d['cause_label']:<34} {_bar(d['share_of_deaths'], 16)} {fmt.pct(d['share_of_deaths'])}")
    reqs = [r for r in s["requirements"] if r["achievable"]][:3]
    if reqs:
        print(f"\n  What must be true for {fmt.pct(s['survival_target'])} survival")
        for r in reqs:
            print(f"    {r['label']:<34} {r['direction']} {fmt.value(r['threshold'], r['unit'])}")
    if s["levers"]:
        print("\n  Decisions tested")
        for l in s["levers"]:
            print(f"    {l['name'][:40]:<40} {l['delta'] * 100:+5.0f} pts  -> {fmt.pct(l['survival'])}")
    print("\n  Findings")
    for line in s["narrative"]:
        print(f"    - {line}")
    print(f"\n  Report   {paths['html']}")
    print(f"  Data     {paths['json']}  |  {paths['csv']}")
    print(f"  Done in {seconds:.1f}s\n")


def cmd_run(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    if args.sims:
        plan.simulations = args.sims
    if args.seed is not None:
        plan.seed = args.seed
    out = Path(args.out) if args.out else Path("reports") / Path(args.plan).stem
    t0 = time.perf_counter()
    result = run_simulation(plan)
    summary = summarize(result)
    paths = write_report(result, summary, out, save_pngs=not args.no_png)
    _print_summary(summary, paths, time.perf_counter() - t0)
    if args.open:
        webbrowser.open(paths["html"].resolve().as_uri())
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if path.exists() and not args.force:
        print(f"{path} already exists. Use --force to overwrite.", file=sys.stderr)
        return 1
    path.write_text(TEMPLATE, encoding="utf-8")
    print(f"Wrote {path}. Edit the assumptions, then run:  premortem run {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="premortem",
                                description="Stress-test a business plan across thousands of possible futures.")
    p.add_argument("--version", action="version", version=f"premortem {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="simulate a plan and write the report")
    r.add_argument("plan", help="path to a YAML business plan")
    r.add_argument("-o", "--out", help="output folder (default: reports/<plan name>)")
    r.add_argument("--sims", type=int, help="number of simulated futures")
    r.add_argument("--seed", type=int, help="random seed for reproducible results")
    r.add_argument("--no-png", action="store_true", help="skip writing separate chart images")
    r.add_argument("--open", action="store_true", help="open the report in a browser")
    r.set_defaults(func=cmd_run)

    i = sub.add_parser("init", help="write a starter plan to edit")
    i.add_argument("path", nargs="?", default="my_plan.yaml")
    i.add_argument("--force", action="store_true")
    i.set_defaults(func=cmd_init)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"premortem: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
