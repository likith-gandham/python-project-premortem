"""Loading and validating a business plan written in YAML."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .distributions import Distribution, Fixed, from_spec


@dataclass(frozen=True)
class AssumptionInfo:
    label: str
    unit: str  # "$", "%", "#"
    required: bool
    default: float | None = None
    lower: float | None = None
    upper: float | None = None


#: Every lever of the operating model, in the order it is sampled.
ASSUMPTIONS: dict[str, AssumptionInfo] = {
    "starting_cash": AssumptionInfo("Starting cash", "$", True, lower=0),
    "starting_customers": AssumptionInfo("Starting customers", "#", False, 0.0, lower=0),
    "arpu": AssumptionInfo("Revenue per customer / month", "$", True, lower=0),
    "arpu_growth": AssumptionInfo("Monthly price growth", "%", False, 0.0, -0.5, 0.5),
    "gross_margin": AssumptionInfo("Gross margin", "%", True, lower=-1, upper=1),
    "cac": AssumptionInfo("Customer acquisition cost", "$", True, lower=1e-6),
    "marketing_spend": AssumptionInfo("Marketing spend / month", "$", True, lower=0),
    "marketing_growth": AssumptionInfo("Monthly marketing growth", "%", False, 0.0, -0.5, 0.5),
    "organic_signups": AssumptionInfo("Organic sign-ups / month", "#", False, 0.0, lower=0),
    "monthly_churn": AssumptionInfo("Monthly churn", "%", True, lower=0, upper=1),
    "fixed_costs": AssumptionInfo("Fixed costs / month", "$", True, lower=0),
    "fixed_cost_growth": AssumptionInfo("Monthly fixed-cost growth", "%", False, 0.0, -0.5, 0.5),
    "demand_volatility": AssumptionInfo("Demand volatility", "%", False, 0.15, 0.0, 2.0),
}

LEVER_OPS = ("scale", "set", "add")


@dataclass(frozen=True)
class Fundraise:
    """A funding round that only closes if the company hits a revenue milestone."""

    month: int
    amount: float
    mrr_milestone: float = 0.0
    close_probability: float = 1.0


@dataclass(frozen=True)
class Lever:
    """A management decision to test, e.g. 'raise prices 15%'."""

    name: str
    changes: dict[str, dict[str, float]]


@dataclass
class Plan:
    company: str
    assumptions: dict[str, Distribution]
    horizon_months: int = 36
    simulations: int = 10_000
    seed: int = 42
    survival_target: float = 0.80
    description: str = ""
    fundraise: Fundraise | None = None
    levers: list[Lever] = field(default_factory=list)

    def label(self, key: str) -> str:
        return ASSUMPTIONS[key].label if key in ASSUMPTIONS else key


def _suggest(key: str, options: list[str]) -> str:
    match = difflib.get_close_matches(key, options, n=1)
    return f" Did you mean '{match[0]}'?" if match else ""


def _parse_lever(raw: dict[str, Any], index: int) -> Lever:
    if not isinstance(raw, dict) or "name" not in raw or "changes" not in raw:
        raise ValueError(f"levers[{index}] needs a 'name' and 'changes'")
    changes: dict[str, dict[str, float]] = {}
    for key, op in dict(raw["changes"]).items():
        if key not in ASSUMPTIONS:
            raise ValueError(f"lever '{raw['name']}': unknown assumption '{key}'."
                             + _suggest(key, list(ASSUMPTIONS)))
        if not isinstance(op, dict) or len(op) != 1 or next(iter(op)) not in LEVER_OPS:
            raise ValueError(f"lever '{raw['name']}': change to '{key}' must be one of "
                             f"{{scale: x}}, {{set: x}} or {{add: x}}")
        (name, value), = op.items()
        changes[key] = {name: float(value)}
    return Lever(str(raw["name"]), changes)


def plan_from_dict(data: dict[str, Any]) -> Plan:
    """Build and validate a :class:`Plan` from a plain dictionary."""
    if not isinstance(data, dict):
        raise ValueError("The plan file must be a YAML mapping")
    raw = data.get("assumptions") or {}
    if not isinstance(raw, dict):
        raise ValueError("'assumptions' must be a mapping")

    unknown = [k for k in raw if k not in ASSUMPTIONS]
    if unknown:
        k = unknown[0]
        raise ValueError(f"Unknown assumption '{k}'." + _suggest(k, list(ASSUMPTIONS)))

    assumptions: dict[str, Distribution] = {}
    for key, info in ASSUMPTIONS.items():
        if key in raw:
            assumptions[key] = from_spec(raw[key], key)
        elif info.required:
            raise ValueError(f"Missing required assumption '{key}' ({info.label})")
        else:
            assumptions[key] = Fixed(float(info.default or 0.0))

    fundraise = None
    if data.get("fundraise"):
        fr = dict(data["fundraise"])
        try:
            fundraise = Fundraise(
                month=int(fr["month"]),
                amount=float(fr["amount"]),
                mrr_milestone=float(fr.get("mrr_milestone", 0.0)),
                close_probability=float(fr.get("close_probability", 1.0)),
            )
        except KeyError as exc:
            raise ValueError(f"fundraise is missing {exc}") from None
        if not 0 <= fundraise.close_probability <= 1:
            raise ValueError("fundraise.close_probability must be between 0 and 1")

    plan = Plan(
        company=str(data.get("company", "Unnamed company")),
        description=str(data.get("description", "")).strip(),
        assumptions=assumptions,
        horizon_months=int(data.get("horizon_months", 36)),
        simulations=int(data.get("simulations", 10_000)),
        seed=int(data.get("seed", 42)),
        survival_target=float(data.get("survival_target", 0.80)),
        fundraise=fundraise,
        levers=[_parse_lever(l, i) for i, l in enumerate(data.get("levers") or [])],
    )
    if not 1 <= plan.horizon_months <= 120:
        raise ValueError("horizon_months must be between 1 and 120")
    if not 100 <= plan.simulations <= 1_000_000:
        raise ValueError("simulations must be between 100 and 1,000,000")
    if not 0 < plan.survival_target < 1:
        raise ValueError("survival_target must be between 0 and 1 (e.g. 0.8)")
    if fundraise and not 1 <= fundraise.month <= plan.horizon_months:
        raise ValueError("fundraise.month must fall inside the horizon")
    return plan


def load_plan(path: str | Path) -> Plan:
    """Read a YAML business plan from disk."""
    with open(path, encoding="utf-8") as fh:
        return plan_from_dict(yaml.safe_load(fh))
