"""Write the report: a self-contained HTML memo, a JSON summary and a CSV of every run."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import charts
from . import formatting as fmt
from .analysis import FUNDRAISE_KEY
from .config import ASSUMPTIONS
from .engine import SimulationResult

CSS = """
:root{--paper:#F3F5F7;--sheet:#FFFFFF;--ink:#1B2A3A;--muted:#5E6B7A;--rule:#DCE2E8;
--live:#2E7D7A;--dead:#B23A48;--warn:#C98A12;
--serif:"Iowan Old Style","Charter","Palatino Linotype",Georgia,serif;
--sans:"Avenir Next","Segoe UI",system-ui,-apple-system,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 var(--sans)}
main{max-width:880px;margin:0 auto;padding:48px 24px 80px}
.sheet{background:var(--sheet);border:1px solid var(--rule);border-radius:4px;padding:40px 44px;margin-bottom:28px}
.case{color:var(--muted);font-size:14px;margin:0 0 20px}
h1{font:600 clamp(30px,5vw,46px)/1.12 var(--serif);margin:0 0 14px;letter-spacing:-.01em}
h2{font:600 24px/1.25 var(--serif);margin:0 0 6px}
h2+p.lede{margin-top:0}
p.lede{color:var(--muted);max-width:64ch}
.verdict{display:inline-block;padding:3px 12px;border-radius:999px;font-weight:600;font-size:14px;color:#fff}
.green{background:var(--live)}.amber{background:var(--warn)}.red{background:var(--dead)}
.futures{display:grid;grid-template-columns:repeat(20,1fr);gap:5px;margin:28px 0 10px;max-width:520px}
.futures span{aspect-ratio:1;border-radius:50%;background:var(--live)}
.futures span.d{background:transparent;border:2px solid var(--dead)}
.key{font-size:13px;color:var(--muted)}
.key i{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 6px 0 14px;vertical-align:-1px}
.key i:first-child{margin-left:0}
ul.findings{padding-left:20px;max-width:68ch;font-family:var(--serif);font-size:17px;line-height:1.65}
ul.findings li{margin-bottom:8px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:0;border-top:1px solid var(--rule);margin-top:24px}
.stats div{padding:14px 16px 12px 0;border-bottom:1px solid var(--rule)}
.stats b{display:block;font:600 24px/1.2 var(--serif)}
.stats small{color:var(--muted);font-size:13px}
img{max-width:100%;display:block;margin:18px 0 6px}
table{width:100%;border-collapse:collapse;font-size:14.5px;margin-top:14px}
th{text-align:left;font-weight:600;color:var(--muted);border-bottom:2px solid var(--rule);padding:8px 10px 8px 0}
td{border-bottom:1px solid var(--rule);padding:9px 10px 9px 0;vertical-align:top}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.bar{height:6px;background:var(--rule);border-radius:3px;margin-top:6px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--live)}
.bar.low i{background:var(--dead)}
.method{font-size:14.5px;color:var(--muted);max-width:68ch}
footer{color:var(--muted);font-size:13px;text-align:center}
@media (max-width:600px){.sheet{padding:26px 20px}.futures{gap:3px}}
"""


def _futures_grid(survival: float) -> str:
    alive = int(round(survival * 100))
    dots = "".join('<span></span>' if i < alive else '<span class="d"></span>' for i in range(100))
    return (f'<div class="futures" role="img" aria-label="{alive} of 100 futures survive">{dots}</div>'
            f'<div class="key"><i style="background:var(--live)"></i>survives'
            f'<i style="border:2px solid var(--dead)"></i>runs out of cash</div>')


def _img(b64: str, alt: str) -> str:
    return f'<img alt="{escape(alt)}" src="data:image/png;base64,{b64}">'


def render_html(summary: dict[str, Any], images: dict[str, str]) -> str:
    s = summary
    alive = int(round(s["survival_rate"] * 100))
    company = escape(s["company"])
    findings = "".join(f"<li>{escape(line)}</li>" for line in s["narrative"])

    stats = [
        (fmt.pct(s["survival_rate"]), f"survive {s['horizon_months']} months"),
        (f"month {s['median_death_month']:.0f}" if s["median_death_month"] else "none",
         "median time of death"),
        (fmt.pct(s["reach_profitability"]), "ever reach monthly profit"),
        (fmt.money(s["final_mrr"]["p50"]), "median final monthly revenue of survivors"),
    ]
    stats_html = "".join(f"<div><b>{escape(v)}</b><small>{escape(k)}</small></div>" for v, k in stats)

    fr_html = ""
    if s["fundraise"]:
        f = s["fundraise"]
        fr_html = (f"<p class='lede'>The plan depends on raising {fmt.money(f['amount'])} in month {f['month']}, "
                   f"which needs {fmt.money(f['mrr_milestone'])} of monthly revenue. "
                   f"{fmt.pct(f['alive_to_attempt'])} of futures are still alive to try, "
                   f"{fmt.pct(f['hit_milestone_share'])} of those hit the milestone, and "
                   f"{fmt.pct(f['closed_share_of_attempts'])} actually close the round.</p>")

    req_rows = ""
    for r in s["requirements"]:
        if r["achievable"]:
            need = f"{r['direction']} {fmt.value(r['threshold'], r['unit'])}"
            share = r["share_of_range_ok"]
            bar = f"<div class='bar{' low' if share < 0.5 else ''}'><i style='width:{share * 100:.0f}%'></i></div>"
            odds = f"{fmt.pct(share)} of your range{bar}"
        else:
            need = "not reachable alone"
            odds = f"best case {fmt.pct(r['best_possible_survival'])} survival"
        req_rows += (f"<tr><td>{escape(r['label'])}</td><td class='num'>{fmt.value(r['assumed_median'], r['unit'])}</td>"
                     f"<td class='num'><b>{escape(need)}</b></td><td>{odds}</td></tr>")
    req_html = ""
    if req_rows:
        req_html = f"""
<section class="sheet"><h2>What must be true</h2>
<p class="lede">A reverse stress test. For each of the biggest drivers, the value it has to reach for
{fmt.pct(s['survival_target'])} of futures to survive, with every other assumption still uncertain.
The last column shows how much of your own assumed range already clears the bar.</p>
<table><thead><tr><th>Assumption</th><th>You assumed</th><th>Needs to be</th><th>Already there</th></tr></thead>
<tbody>{req_rows}</tbody></table></section>"""

    lever_html = ""
    if "levers" in images:
        lever_html = f"""
<section class="sheet"><h2>Decisions tested</h2>
<p class="lede">Each decision is replayed against the exact same {s['simulations']:,} futures, so any
difference in survival comes from the decision, not from luck.</p>
{_img(images['levers'], 'Change in survival for each decision')}</section>"""

    ue = s["unit_economics"]
    ue_rows = "".join(
        f"<tr><td>{name}</td><td class='num'>{f(ue[k]['p10'])}</td><td class='num'><b>{f(ue[k]['p50'])}</b></td>"
        f"<td class='num'>{f(ue[k]['p90'])}</td></tr>"
        for name, k, f in [
            ("Customer lifetime value", "ltv", fmt.money),
            ("LTV to CAC", "ltv_to_cac", lambda v: f"{v:.1f}x"),
            ("CAC payback", "payback_months", lambda v: f"{v:.1f} mo"),
        ])

    assumption_rows = "".join(
        f"<tr><td>{escape(a['label'])}</td><td>{escape(a['description'])}</td></tr>"
        for a in s["assumptions"])

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Premortem: {company}</title><style>{CSS}</style></head><body><main>

<section class="sheet">
<p class="case">Premortem report &nbsp;/&nbsp; {s['simulations']:,} simulated futures over {s['horizon_months']} months</p>
<h1>{company} survives {alive} of every 100 futures.</h1>
<span class="verdict {s['verdict_color']}">{escape(s['verdict'])}</span>
{f"<p class='lede'>{escape(s['description'])}</p>" if s['description'] else ""}
{_futures_grid(s['survival_rate'])}
<div class="stats">{stats_html}</div>
</section>

<section class="sheet"><h2>Findings</h2><ul class="findings">{findings}</ul></section>

<section class="sheet"><h2>How long the company lives</h2>
<p class="lede">The share of futures still solvent each month, and the spread of cash and revenue behind it.</p>
{_img(images['survival'], 'Survival curve')}{_img(images['cash'], 'Cash fan chart')}
{fr_html}{_img(images['mrr'], 'Monthly revenue fan chart')}</section>

<section class="sheet"><h2>How it dies</h2>
<p class="lede">Every failed future is attributed to the assumption that hurt it most, weighted by how
strongly that assumption drives survival overall.</p>
{_img(images['causes'], 'Causes of death')}{_img(images['timing'], 'Timing of failures')}</section>

<section class="sheet"><h2>What swings survival</h2>
<p class="lede">Survival when each assumption lands in its worst tenth (red) versus its best tenth (green).
The labels show the values at those points. Longer bars deserve more of management's attention.</p>
{_img(images['tornado'], 'Survival tornado chart')}</section>
{req_html}{lever_html}

<section class="sheet"><h2>Unit economics</h2>
<table><thead><tr><th>Metric</th><th>Pessimistic (P10)</th><th>Median</th><th>Optimistic (P90)</th></tr></thead>
<tbody>{ue_rows}</tbody></table></section>

<section class="sheet"><h2>Assumptions used</h2>
<table><thead><tr><th>Assumption</th><th>Range</th></tr></thead><tbody>{assumption_rows}</tbody></table>
<p class="method" style="margin-top:22px">Method: Monte Carlo simulation with Beta-PERT and other distributions,
monthly cohort-style customer model with log-normal demand shocks, cause-of-death attribution by
Spearman-weighted adverse percentiles, and reverse stress testing by bisection with common random numbers.</p>
</section>

<footer>Generated by Premortem. A simulation is a thinking tool, not a forecast.</footer>
</main></body></html>"""


def runs_frame(res: SimulationResult, causes: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({k: res.inputs[k] for k in ASSUMPTIONS})
    df.insert(0, "run", np.arange(1, res.n + 1))
    df["survived"] = res.survived
    df["death_month"] = res.death_month
    df["cause_of_death"] = causes
    df["breakeven_month"] = res.breakeven_month
    df["final_cash"] = res.final_cash
    df["final_mrr"] = res.mrr[:, -1]
    if res.plan.fundraise is not None:
        df["raise_closed"] = res.raise_closed
    return df


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if np.isnan(v) or np.isinf(v) else round(v, 6)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_report(res: SimulationResult, summary: dict[str, Any], out_dir: str | Path,
                 save_pngs: bool = True) -> dict[str, Path]:
    """Write report.html, summary.json, simulations.csv and chart PNGs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs = charts.build_all(res, summary)
    images = {name: charts.to_base64(fig) for name, fig in figs.items()}
    paths: dict[str, Path] = {}

    if save_pngs:
        (out / "charts").mkdir(exist_ok=True)
        for name, fig in figs.items():
            p = out / "charts" / f"{name}.png"
            fig.savefig(p, bbox_inches="tight")
        dash = charts.dashboard(res, summary)
        dash.savefig(out / "charts" / "dashboard.png", bbox_inches="tight")
        paths["charts"] = out / "charts"
    charts.close_all()

    paths["html"] = out / "report.html"
    paths["html"].write_text(render_html(summary, images), encoding="utf-8")
    paths["json"] = out / "summary.json"
    paths["json"].write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    paths["csv"] = out / "simulations.csv"
    runs_frame(res, summary["_causes"]).to_csv(paths["csv"], index=False)
    return paths


__all__ = ["write_report", "render_html", "runs_frame", "FUNDRAISE_KEY"]
