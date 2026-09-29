import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from premortem import plan_from_dict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def base_dict():
    """A small but realistic plan with uncertainty in the key drivers."""
    return {
        "company": "TestCo",
        "horizon_months": 24,
        "simulations": 2000,
        "seed": 1,
        "assumptions": {
            "starting_cash": 300000,
            "starting_customers": 20,
            "arpu": {"dist": "pert", "low": 40, "mode": 55, "high": 65},
            "gross_margin": 0.7,
            "cac": {"dist": "pert", "low": 100, "mode": 200, "high": 450},
            "marketing_spend": 10000,
            "organic_signups": {"dist": "pert", "low": 5, "mode": 15, "high": 30},
            "monthly_churn": {"dist": "pert", "low": 0.02, "mode": 0.05, "high": 0.10},
            "fixed_costs": 20000,
        },
        "levers": [
            {"name": "Cut fixed costs", "changes": {"fixed_costs": {"scale": 0.7}}},
            {"name": "Burn money", "changes": {"fixed_costs": {"add": 15000}}},
        ],
    }


@pytest.fixture
def plan(base_dict):
    return plan_from_dict(base_dict)


@pytest.fixture
def deterministic_dict():
    """No randomness at all, so results can be checked by hand."""
    return {
        "company": "Clockwork",
        "horizon_months": 6,
        "simulations": 100,
        "assumptions": {
            "starting_cash": 1000,
            "arpu": 5,
            "gross_margin": 1.0,
            "cac": 10,
            "marketing_spend": 100,
            "monthly_churn": 0.0,
            "fixed_costs": 100,
            "demand_volatility": 0.0,
        },
    }
