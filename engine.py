"""The Monte Carlo engine: thousands of possible futures, simulated month by month.

Every simulated future ("run") draws one value for each uncertain assumption,
then plays the company forward:

    new customers  = (marketing spend / CAC + organic sign-ups) x demand shock
    customers      = last month's customers x (1 - churn) + new customers
    revenue (MRR)  = customers x revenue per customer
    net income     = revenue x gross margin - fixed costs - marketing spend
    cash           = last month's cash + net income (+ fundraise, if it closes)

A company dies the first month its cash goes below zero. It cannot come back.

All runs are computed together as NumPy arrays, so 10,000 futures over
three years take well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ASSUMPTIONS, Plan

NOISE_KEY = "_demand_noise"
RAISE_KEY = "_raise_draw"


@dataclass
class SimulationResult:
    plan: Plan
    inputs: dict[str, np.ndarray]
    cash: np.ndarray            # (runs, months + 1), month 0 = starting cash
    mrr: np.ndarray             # (runs, months)
    customers: np.ndarray       # (runs, months)
    net_income: np.ndarray      # (runs, months)
    death_month: np.ndarray     # (runs,), NaN if the company survived
    breakeven_month: np.ndarray  # (runs,), first profitable month, NaN if never
    raise_attempted: np.ndarray  # (runs,) bool
    raise_closed: np.ndarray     # (runs,) bool

    @property
    def n(self) -> int:
        return self.cash.shape[0]

    @property
    def horizon(self) -> int:
        return self.cash.shape[1] - 1

    @property
    def survived(self) -> np.ndarray:
        return np.isnan(self.death_month)

    @property
    def survival_rate(self) -> float:
        return float(self.survived.mean())

    @property
    def final_cash(self) -> np.ndarray:
        return self.cash[:, -1]


def sample_inputs(plan: Plan, n: int | None = None, seed: int | None = None) -> dict[str, np.ndarray]:
    """Draw every random number the simulation needs, up front.

    Drawing once and re-using the same draws when testing a decision
    ("common random numbers") means differences between scenarios come from
    the decision, not from luck.
    """
    n = int(n or plan.simulations)
    rng = np.random.default_rng(plan.seed if seed is None else seed)
    inputs: dict[str, np.ndarray] = {}
    for key, info in ASSUMPTIONS.items():
        values = plan.assumptions[key].sample(rng, n).astype(float)
        lo = -np.inf if info.lower is None else info.lower
        hi = np.inf if info.upper is None else info.upper
        inputs[key] = np.clip(values, lo, hi)
    inputs[NOISE_KEY] = rng.standard_normal((n, plan.horizon_months))
    inputs[RAISE_KEY] = rng.random(n)
    return inputs


def apply_changes(inputs: dict[str, np.ndarray], changes: dict[str, dict[str, float]]) -> dict[str, np.ndarray]:
    """Return a copy of ``inputs`` with a lever's changes applied."""
    out = dict(inputs)
    for key, op in changes.items():
        (kind, value), = op.items()
        base = inputs[key]
        if kind == "scale":
            new = base * value
        elif kind == "add":
            new = base + value
        elif kind == "set":
            new = np.full_like(base, value)
        else:  # pragma: no cover - validated in config
            raise ValueError(kind)
        info = ASSUMPTIONS[key]
        lo = -np.inf if info.lower is None else info.lower
        hi = np.inf if info.upper is None else info.upper
        out[key] = np.clip(new, lo, hi)
    return out


def simulate(plan: Plan, inputs: dict[str, np.ndarray]) -> SimulationResult:
    """Play every run forward month by month."""
    g = inputs
    n = len(g["starting_cash"])
    horizon = plan.horizon_months
    noise = g[NOISE_KEY]
    if noise.shape[1] < horizon:
        raise ValueError("demand noise is shorter than the horizon")

    cash = np.empty((n, horizon + 1))
    cash[:, 0] = g["starting_cash"]
    mrr = np.zeros((n, horizon))
    customers = np.zeros((n, horizon))
    net_income = np.zeros((n, horizon))
    death = np.full(n, np.nan)
    breakeven = np.full(n, np.nan)
    attempted = np.zeros(n, dtype=bool)
    closed = np.zeros(n, dtype=bool)

    alive = np.ones(n, dtype=bool)
    cust = g["starting_customers"].copy()
    sigma = g["demand_volatility"]
    fr = plan.fundraise

    for t in range(horizon):
        marketing = g["marketing_spend"] * (1 + g["marketing_growth"]) ** t
        fixed = g["fixed_costs"] * (1 + g["fixed_cost_growth"]) ** t
        price = g["arpu"] * (1 + g["arpu_growth"]) ** t
        # Log-normal shock with mean 1: some months are simply better than others.
        shock = np.exp(sigma * noise[:, t] - 0.5 * sigma**2)
        new_customers = (marketing / g["cac"] + g["organic_signups"]) * shock

        cust = np.where(alive, cust * (1 - g["monthly_churn"]) + new_customers, 0.0)
        revenue = cust * price
        net = np.where(alive, revenue * g["gross_margin"] - fixed - marketing, 0.0)
        balance = cash[:, t] + net

        if fr is not None and t + 1 == fr.month:
            attempted = alive.copy()
            closed = attempted & (revenue >= fr.mrr_milestone) & (g[RAISE_KEY] < fr.close_probability)
            balance = balance + np.where(closed, fr.amount, 0.0)

        newly_dead = alive & (balance < 0)
        death[newly_dead] = t + 1
        alive &= ~newly_dead

        first_profit = alive & (net > 0) & np.isnan(breakeven)
        breakeven[first_profit] = t + 1

        cash[:, t + 1] = balance
        mrr[:, t] = revenue
        customers[:, t] = cust
        net_income[:, t] = net

    return SimulationResult(plan, inputs, cash, mrr, customers, net_income,
                            death, breakeven, attempted, closed)


def run(plan: Plan, n: int | None = None, seed: int | None = None) -> SimulationResult:
    """Sample assumptions and simulate in one call."""
    return simulate(plan, sample_inputs(plan, n, seed))
