"""Charts for the report, drawn with matplotlib (no display needed)."""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

from . import formatting as fmt  # noqa: E402
from .analysis import bands, survival_curve  # noqa: E402
from .engine import SimulationResult  # noqa: E402

INK = "#1B2A3A"
MUTED = "#7A8797"
GRID = "#DCE2E8"
LIVE = "#2E7D7A"
DEAD = "#B23A48"
WARN = "#C98A12"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 160,
})

_money_axis = FuncFormatter(lambda v, _: fmt.money(v))


def _fundraise_line(ax, res: SimulationResult) -> None:
    fr = res.plan.fundraise
    if fr is not None:
        ax.axvline(fr.month, color=WARN, lw=1.2, ls="--")
        ax.annotate(f"Raise {fmt.money(fr.amount)}", (fr.month, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -12), textcoords="offset points", color=WARN, fontsize=8.5)


def draw_survival(ax, res: SimulationResult) -> None:
    curve = survival_curve(res)
    months = np.arange(len(curve))
    ax.fill_between(months, curve, step="post", color=LIVE, alpha=0.15)
    ax.step(months, curve, where="post", color=LIVE, lw=2.2)
    ax.set_ylim(0, 1.04)
    ax.set_xlim(0, res.horizon)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of futures still alive")
    ax.set_title("Survival curve")
    ax.annotate(fmt.pct(curve[-1]), (months[-1], curve[-1]), xytext=(-6, 8),
                textcoords="offset points", ha="right", color=LIVE, fontweight="bold", fontsize=12)
    _fundraise_line(ax, res)


def draw_cash_fan(ax, res: SimulationResult) -> None:
    b = bands(res.cash)
    months = np.arange(res.cash.shape[1])
    ax.fill_between(months, b[5], b[95], color=INK, alpha=0.08, lw=0, label="90% of futures")
    ax.fill_between(months, b[25], b[75], color=INK, alpha=0.18, lw=0, label="50% of futures")
    ax.plot(months, b[50], color=INK, lw=2, label="Median")
    ax.axhline(0, color=DEAD, lw=1.2)
    ax.text(0.5, 0, " out of cash", color=DEAD, fontsize=8.5, va="bottom")
    ax.yaxis.set_major_formatter(_money_axis)
    ax.set_xlim(0, res.horizon)
    ax.set_xlabel("Month")
    ax.set_title("Cash balance across futures")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    _fundraise_line(ax, res)


def draw_tornado(ax, summary: dict[str, Any], top: int = 7) -> None:
    drivers = summary["drivers"][:top][::-1]
    base = summary["survival_rate"]
    y = np.arange(len(drivers))
    for i, d in enumerate(drivers):
        ax.barh(i, d["survival_worst"] - base, left=base, color=DEAD, height=0.6)
        ax.barh(i, d["survival_best"] - base, left=base, color=LIVE, height=0.6)
        ax.text(min(d["survival_worst"], base) - 0.015, i, fmt.value(d["worst_value"], d["unit"]),
                ha="right", va="center", fontsize=8, color=DEAD)
        ax.text(max(d["survival_best"], base) + 0.015, i, fmt.value(d["best_value"], d["unit"]),
                ha="left", va="center", fontsize=8, color=LIVE)
    ax.axvline(base, color=INK, lw=1)
    ax.set_yticks(y, [d["label"] for d in drivers])
    ax.set_xlim(-0.1, 1.1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Survival when this assumption lands in its worst / best 10%")
    ax.set_title("What swings survival")


def draw_causes(ax, summary: dict[str, Any]) -> None:
    killers = sorted([d for d in summary["drivers"] if d["share_of_deaths"] >= 0.005],
                     key=lambda d: d["share_of_deaths"])[-7:]
    if not killers:
        ax.text(0.5, 0.5, "No simulated future failed", ha="center", va="center", color=MUTED)
        ax.set_axis_off()
        return
    y = np.arange(len(killers))
    shares = [d["share_of_deaths"] for d in killers]
    ax.barh(y, shares, color=DEAD, height=0.6)
    for i, s in enumerate(shares):
        ax.text(s + 0.01, i, fmt.pct(s), va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [d["cause_label"] for d in killers])
    ax.set_xlim(0, max(shares) * 1.25)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_title("Cause of death (share of failures)")


def draw_death_timing(ax, res: SimulationResult) -> None:
    dead = res.death_month[~res.survived]
    if dead.size == 0:
        ax.text(0.5, 0.5, "No simulated future failed", ha="center", va="center", color=MUTED)
        ax.set_axis_off()
        return
    ax.hist(dead, bins=np.arange(0.5, res.horizon + 1.5, 1), color=DEAD, alpha=0.85)
    ax.set_xlim(0, res.horizon + 1)
    ax.set_xlabel("Month the cash ran out")
    ax.set_ylabel("Futures")
    ax.set_title("When failures happen")
    _fundraise_line(ax, res)


def draw_levers(ax, summary: dict[str, Any]) -> None:
    levers = summary["levers"][::-1]
    y = np.arange(len(levers))
    deltas = [l["delta"] for l in levers]
    ax.barh(y, deltas, color=[LIVE if d >= 0 else DEAD for d in deltas], height=0.6)
    span = max(max(abs(d) for d in deltas), 0.01)
    for i, l in enumerate(levers):
        off = span * 0.03 * (1 if l["delta"] >= 0 else -1)
        ax.text(l["delta"] + off, i, f"{l['delta'] * 100:+.0f} pts  \u2192 {fmt.pct(l['survival'])}",
                va="center", ha="left" if l["delta"] >= 0 else "right", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, [l["name"] for l in levers])
    ax.set_xlim(-span * 1.6, span * 1.6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:+.0f}"))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in survival (percentage points), same futures")
    ax.set_title("Decisions tested")


def draw_mrr_fan(ax, res: SimulationResult) -> None:
    months = np.arange(1, res.horizon + 1)
    death = np.where(res.survived, np.inf, res.death_month)
    alive_mrr = np.where(death[:, None] > months[None, :], res.mrr, np.nan)
    keep = ~np.all(np.isnan(alive_mrr), axis=0)
    months, alive_mrr = months[keep], alive_mrr[:, keep]
    b = {q: np.nanpercentile(alive_mrr, q, axis=0) for q in (5, 25, 50, 75, 95)}
    ax.fill_between(months, b[5], b[95], color=LIVE, alpha=0.10, lw=0)
    ax.fill_between(months, b[25], b[75], color=LIVE, alpha=0.22, lw=0)
    ax.plot(months, b[50], color=LIVE, lw=2)
    fr = res.plan.fundraise
    if fr is not None and fr.mrr_milestone > 0:
        ax.axhline(fr.mrr_milestone, color=WARN, lw=1.2, ls="--")
        ax.text(1, fr.mrr_milestone, " revenue needed to raise", color=WARN, fontsize=8.5, va="bottom")
    ax.yaxis.set_major_formatter(_money_axis)
    ax.set_xlim(1, res.horizon)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Month")
    ax.set_title("Monthly revenue of companies still alive")


def _single(draw, *args, size=(7.2, 4.2)):
    fig, ax = plt.subplots(figsize=size)
    draw(ax, *args)
    fig.tight_layout()
    return fig


def build_all(res: SimulationResult, summary: dict[str, Any]) -> dict[str, plt.Figure]:
    n_drivers = min(len(summary["drivers"]), 7)
    figs = {
        "survival": _single(draw_survival, res),
        "cash": _single(draw_cash_fan, res),
        "mrr": _single(draw_mrr_fan, res),
        "tornado": _single(draw_tornado, summary, size=(7.2, 1.2 + 0.5 * max(n_drivers, 2))),
        "causes": _single(draw_causes, summary, size=(7.2, 3.6)),
        "timing": _single(draw_death_timing, res, size=(7.2, 3.4)),
    }
    if summary["levers"]:
        figs["levers"] = _single(draw_levers, summary, size=(7.2, 1.2 + 0.55 * len(summary["levers"])))
    return figs


def dashboard(res: SimulationResult, summary: dict[str, Any]) -> plt.Figure:
    """A four-panel overview used as the README hero image."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    draw_survival(axes[0, 0], res)
    draw_cash_fan(axes[0, 1], res)
    draw_tornado(axes[1, 0], summary, top=5)
    draw_causes(axes[1, 1], summary)
    fig.suptitle(f"Premortem: {summary['company']}  |  survives {fmt.pct(summary['survival_rate'])} "
                 f"of {summary['simulations']:,} futures", x=0.01, ha="left",
                 fontsize=15, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def close_all() -> None:
    plt.close("all")
