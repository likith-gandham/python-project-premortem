"""Turning 10,000 simulated futures into decisions.

The analysis answers five questions a founder, investor or board would ask:

1. How likely is the company to survive?               -> survival curve
2. When it dies, what kills it?                          -> cause-of-death attribution
3. Which assumptions matter most?                        -> survival tornado
4. What has to be true for the plan to work?             -> reverse stress test
5. Which management decision helps the most?             -> lever testing
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import formatting as fmt
from .config import ASSUMPTIONS, Plan
from .engine import SimulationResult, apply_changes, simulate

FUNDRAISE_KEY = "fundraise"


@dataclass
class Driver:
    key: str
    label: str
    unit: str
    correlation: float       # Spearman rank correlation with the outcome
    survival_worst: float    # survival when this input lands in its worst 10%
    survival_best: float     # survival when this input lands in its best 10%
    worst_is_high: bool      # True if higher values are bad (e.g. churn, CAC)
    worst_value: float
    best_value: float
    median_value: float
    share_of_deaths: float = 0.0
    cause_label: str = ""

    def __post_init__(self) -> None:
        if not self.cause_label:
            self.cause_label = self.label

    @property
    def swing(self) -> float:
        return self.survival_best - self.survival_worst


@dataclass
class Requirement:
    """Result of a reverse stress test for one assumption."""

    key: str
    label: str
    unit: str
    target: float
    achievable: bool
    direction: str                 # "at most" or "at least"
    threshold: float | None
    assumed_median: float
    share_of_range_ok: float       # how much of the assumed range meets the bar
    best_possible_survival: float


@dataclass
class LeverResult:
    name: str
    survival: float
    delta: float
    median_final_cash: float
    changes: dict[str, dict[str, float]]


# --------------------------------------------------------------------------- helpers

def _ranks(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").to_numpy()


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def outcome_score(res: SimulationResult) -> np.ndarray:
    """A single number per run where higher is always better.

    Dead companies score the month they died; survivors score above every
    dead company, ranked by the cash they end with.
    """
    scale = max(float(np.median(np.abs(res.inputs["starting_cash"]))), 1.0)
    return np.where(res.survived, res.horizon + 1 + res.final_cash / scale, res.death_month)


def stochastic_keys(res: SimulationResult) -> list[str]:
    return [k for k in ASSUMPTIONS if np.ptp(res.inputs[k]) > 0]


# --------------------------------------------------------------------------- survival

def survival_curve(res: SimulationResult) -> np.ndarray:
    """Share of companies still alive at the end of each month (index 0 = today)."""
    months = np.arange(res.horizon + 1)
    death = np.where(res.survived, np.inf, res.death_month)
    return (death[None, :] > months[:, None]).mean(axis=1)


def bands(values: np.ndarray, qs=(5, 25, 50, 75, 95)) -> dict[int, np.ndarray]:
    return {q: np.percentile(values, q, axis=0) for q in qs}


# --------------------------------------------------------------------------- drivers

def driver_analysis(res: SimulationResult) -> list[Driver]:
    """Rank every uncertain assumption by how much it swings survival."""
    score = outcome_score(res)
    survived = res.survived
    drivers: list[Driver] = []

    for key in stochastic_keys(res):
        x = res.inputs[key]
        info = ASSUMPTIONS[key]
        lo_q, hi_q = np.quantile(x, [0.1, 0.9])
        s_low = float(survived[x <= lo_q].mean())
        s_high = float(survived[x >= hi_q].mean())
        rho = _spearman(x, score)
        worst_is_high = s_high < s_low if s_high != s_low else rho < 0
        drivers.append(Driver(
            key=key, label=info.label, unit=info.unit, correlation=rho,
            survival_worst=min(s_low, s_high), survival_best=max(s_low, s_high),
            worst_is_high=bool(worst_is_high),
            worst_value=float(hi_q if worst_is_high else lo_q),
            best_value=float(lo_q if worst_is_high else hi_q),
            median_value=float(np.median(x)),
        ))

    if res.plan.fundraise is not None and res.raise_attempted.any():
        closed = res.raise_closed.astype(float)
        att = res.raise_attempted
        if 0 < closed[att].mean() < 1:
            drivers.append(Driver(
                key=FUNDRAISE_KEY, label="Fundraise closes", unit="bool",
                cause_label="Failed fundraise",
                correlation=_spearman(closed, score),
                survival_worst=float(survived[att & ~res.raise_closed].mean()),
                survival_best=float(survived[res.raise_closed].mean()),
                worst_is_high=False, worst_value=0.0, best_value=1.0,
                median_value=float(closed[att].mean()),
            ))

    drivers.sort(key=lambda d: d.swing, reverse=True)
    return drivers


def cause_of_death(res: SimulationResult, drivers: list[Driver]) -> np.ndarray:
    """Attribute each failed run to the assumption that hurt it most.

    For every dead company we ask: which input landed furthest on its bad
    side, weighted by how strongly that input drives survival overall?
    The winner is recorded as the cause of death. Survivors get "".
    Also fills in ``share_of_deaths`` on each driver.
    """
    causes = np.full(res.n, "", dtype=object)
    dead = ~res.survived
    if not dead.any() or not drivers:
        return causes

    contributions = np.zeros((res.n, len(drivers)))
    for j, d in enumerate(drivers):
        weight = abs(d.correlation)
        if d.key == FUNDRAISE_KEY:
            adverse = np.where(res.raise_attempted, 1.0 - res.raise_closed, 0.5)
        else:
            pct_rank = pd.Series(res.inputs[d.key]).rank(pct=True).to_numpy()
            adverse = pct_rank if d.worst_is_high else 1.0 - pct_rank
        contributions[:, j] = weight * (adverse - 0.5)

    winner = contributions.argmax(axis=1)
    labels = np.array([d.cause_label for d in drivers], dtype=object)
    causes[dead] = labels[winner[dead]]
    for j, d in enumerate(drivers):
        d.share_of_deaths = float((winner[dead] == j).mean())
    return causes


# --------------------------------------------------------------------------- reverse stress test

def what_must_be_true(plan: Plan, inputs: dict[str, np.ndarray], key: str,
                      target: float | None = None, n_eval: int = 2000) -> Requirement:
    """Find the value an assumption must hit for survival to reach ``target``.

    Everything else stays uncertain. Because the same random draws are reused
    at every step, survival changes smoothly and a bisection search works.
    """
    target = plan.survival_target if target is None else target
    info = ASSUMPTIONS[key]
    sub = {k: v[:n_eval] for k, v in inputs.items()}
    x = inputs[key]
    lo, hi = (float(v) for v in np.quantile(x, [0.005, 0.995]))

    def survival_at(v: float) -> float:
        return simulate(plan, apply_changes(sub, {key: {"set": v}})).survival_rate

    s_lo, s_hi = survival_at(lo), survival_at(hi)
    good, bad = (lo, hi) if s_lo >= s_hi else (hi, lo)
    direction = "at most" if good < bad else "at least"
    best = max(s_lo, s_hi)

    if best < target:
        return Requirement(key, info.label, info.unit, target, False, direction, None,
                           float(np.median(x)), 0.0, best)

    if survival_at(bad) >= target:
        threshold = bad
    else:
        for _ in range(22):
            mid = (good + bad) / 2
            if survival_at(mid) >= target:
                good = mid
            else:
                bad = mid
        threshold = good

    ok = x <= threshold if direction == "at most" else x >= threshold
    return Requirement(key, info.label, info.unit, target, True, direction, float(threshold),
                       float(np.median(x)), float(ok.mean()), best)


# --------------------------------------------------------------------------- levers

def evaluate_levers(res: SimulationResult) -> list[LeverResult]:
    """Re-run the exact same futures under each management decision."""
    out = []
    for lever in res.plan.levers:
        alt = simulate(res.plan, apply_changes(res.inputs, lever.changes))
        out.append(LeverResult(lever.name, alt.survival_rate, alt.survival_rate - res.survival_rate,
                               float(np.median(alt.final_cash)), lever.changes))
    out.sort(key=lambda r: r.delta, reverse=True)
    return out


# --------------------------------------------------------------------------- unit economics

def unit_economics(res: SimulationResult) -> dict[str, dict[str, float]]:
    g = res.inputs
    monthly_gp = g["arpu"] * g["gross_margin"]
    ltv = monthly_gp / np.maximum(g["monthly_churn"], 1e-4)
    metrics = {
        "ltv": ltv,
        "ltv_to_cac": ltv / g["cac"],
        "payback_months": np.where(monthly_gp > 0, g["cac"] / np.maximum(monthly_gp, 1e-9), np.inf),
    }
    return {k: {f"p{q}": float(np.percentile(v, q)) for q in (10, 50, 90)} for k, v in metrics.items()}


# --------------------------------------------------------------------------- verdict & narrative

def verdict(survival: float) -> tuple[str, str]:
    if survival >= 0.80:
        return "Resilient", "green"
    if survival >= 0.50:
        return "Fragile", "amber"
    return "Likely to fail", "red"


def narrative(s: dict[str, Any]) -> list[str]:
    """Write the executive summary in plain English."""
    lines = []
    surv = s["survival_rate"]
    lines.append(f"Across {s['simulations']:,} simulated futures, {s['company']} survives the full "
                 f"{s['horizon_months']}-month horizon in {fmt.pct(surv)} of them.")
    if s["deaths"]:
        early = s["early_death_share"]
        timing = ("almost none happen in the first year" if early < 0.02
                  else f"{fmt.pct(early)} of failures happen in the first year")
        lines.append(f"When it fails, the median time of death is month {s['median_death_month']:.0f}, "
                     f"and {timing}.")
    killers = [d for d in s["drivers"] if d["share_of_deaths"] > 0]
    if killers:
        k = max(killers, key=lambda d: d["share_of_deaths"])
        name = k["cause_label"] if k["unit"] == "bool" else k["label"]
        lines.append(f"The number one killer is {name.lower()}: it is the decisive factor "
                     f"in {fmt.pct(k['share_of_deaths'])} of failures.")
    if s["drivers"]:
        d = s["drivers"][0]
        if d["unit"] == "bool":
            lines.append(f"The funding round is the swing factor: survival is {fmt.pct(d['survival_best'])} "
                         f"when it closes and {fmt.pct(d['survival_worst'])} when it does not.")
        else:
            lines.append(f"If {d['label'].lower()} lands in its worst 10% ({fmt.value(d['worst_value'], d['unit'])}), "
                         f"survival falls to {fmt.pct(d['survival_worst'])}; in its best 10% "
                         f"({fmt.value(d['best_value'], d['unit'])}) it rises to {fmt.pct(d['survival_best'])}.")
    for r in s["requirements"][:1]:
        if r["achievable"]:
            share = r["share_of_range_ok"]
            tail = (f"Only {fmt.pct(share)} of the range you assumed clears that bar." if share < 0.5
                    else f"{fmt.pct(share)} of the range you assumed already clears that bar.")
            lines.append(f"To reach {fmt.pct(r['target'])} survival, {r['label'].lower()} must be "
                         f"{r['direction']} {fmt.value(r['threshold'], r['unit'])}. {tail}")
        else:
            lines.append(f"No plausible value of {r['label'].lower()} alone gets survival to "
                         f"{fmt.pct(r['target'])}; the best case is {fmt.pct(r['best_possible_survival'])}.")
    if s["levers"]:
        b = s["levers"][0]
        if b["delta"] > 0:
            lines.append(f"The strongest decision tested is \u201c{b['name']}\u201d, which moves survival from "
                         f"{fmt.pct(surv)} to {fmt.pct(b['survival'])} ({b['delta'] * 100:+.0f} points).")
        else:
            lines.append("None of the decisions tested improve survival; the plan needs a different fix.")
    ue = s["unit_economics"]
    lines.append(f"Median LTV to CAC is {ue['ltv_to_cac']['p50']:.1f}x (a common benchmark is 3x or more) "
                 f"and CAC is paid back in about {ue['payback_months']['p50']:.0f} months.")
    return lines


# --------------------------------------------------------------------------- one-shot summary

def summarize(res: SimulationResult, requirements_for: int = 3) -> dict[str, Any]:
    """Run every analysis and return a JSON-friendly dictionary."""
    plan = res.plan
    drivers = driver_analysis(res)
    causes = cause_of_death(res, drivers)
    dead = ~res.survived

    reqs = []
    for d in [d for d in drivers if d.key != FUNDRAISE_KEY][:requirements_for]:
        reqs.append(asdict(what_must_be_true(plan, res.inputs, d.key)))

    levers = [asdict(l) for l in evaluate_levers(res)]
    label, color = verdict(res.survival_rate)
    be = res.breakeven_month

    summary: dict[str, Any] = {
        "company": plan.company,
        "description": plan.description,
        "simulations": res.n,
        "horizon_months": res.horizon,
        "survival_target": plan.survival_target,
        "survival_rate": res.survival_rate,
        "verdict": label,
        "verdict_color": color,
        "deaths": int(dead.sum()),
        "median_death_month": float(np.median(res.death_month[dead])) if dead.any() else None,
        "early_death_share": float((res.death_month[dead] <= 12).mean()) if dead.any() else 0.0,
        "reach_profitability": float((~np.isnan(be)).mean()),
        "median_breakeven_month": float(np.nanmedian(be)) if (~np.isnan(be)).any() else None,
        "final_cash": {f"p{q}": float(np.percentile(res.final_cash, q)) for q in (10, 50, 90)},
        "final_mrr": {f"p{q}": float(np.percentile(res.mrr[res.survived, -1], q)) if res.survived.any()
                      else 0.0 for q in (10, 50, 90)},
        "fundraise": None,
        "drivers": [asdict(d) | {"swing": d.swing} for d in drivers],
        "requirements": reqs,
        "levers": levers,
        "unit_economics": unit_economics(res),
        "assumptions": [
            {"key": k, "label": info.label, "unit": info.unit,
             "description": plan.assumptions[k].describe(),
             "median": float(np.median(res.inputs[k]))}
            for k, info in ASSUMPTIONS.items()
        ],
    }
    if plan.fundraise is not None:
        att = res.raise_attempted
        summary["fundraise"] = {
            "month": plan.fundraise.month,
            "amount": plan.fundraise.amount,
            "mrr_milestone": plan.fundraise.mrr_milestone,
            "close_probability": plan.fundraise.close_probability,
            "alive_to_attempt": float(att.mean()),
            "closed_share_of_attempts": float(res.raise_closed[att].mean()) if att.any() else 0.0,
            "hit_milestone_share": float((res.mrr[att, plan.fundraise.month - 1]
                                          >= plan.fundraise.mrr_milestone).mean()) if att.any() else 0.0,
        }
    summary["narrative"] = narrative(summary)
    summary["_causes"] = causes  # per-run, dropped before JSON export
    return summary
