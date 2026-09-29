import pytest

from premortem import load_plan, plan_from_dict
from premortem.distributions import Fixed

from .conftest import ROOT


def test_defaults_fill_optional_assumptions(plan):
    assert isinstance(plan.assumptions["arpu_growth"], Fixed)
    assert plan.assumptions["demand_volatility"].mean == pytest.approx(0.15)


def test_typo_gets_a_suggestion(base_dict):
    base_dict["assumptions"]["montly_churn"] = base_dict["assumptions"].pop("monthly_churn")
    with pytest.raises(ValueError, match="Did you mean 'monthly_churn'"):
        plan_from_dict(base_dict)


def test_missing_required_assumption(base_dict):
    del base_dict["assumptions"]["cac"]
    with pytest.raises(ValueError, match="Missing required assumption 'cac'"):
        plan_from_dict(base_dict)


def test_bad_lever_operation(base_dict):
    base_dict["levers"] = [{"name": "x", "changes": {"cac": {"multiply": 2}}}]
    with pytest.raises(ValueError, match="scale"):
        plan_from_dict(base_dict)


def test_fundraise_must_be_inside_horizon(base_dict):
    base_dict["fundraise"] = {"month": 99, "amount": 1}
    with pytest.raises(ValueError, match="inside the horizon"):
        plan_from_dict(base_dict)


@pytest.mark.parametrize("name", ["saas_startup.yaml", "coffee_subscription.yaml"])
def test_examples_load(name):
    plan = load_plan(ROOT / "examples" / name)
    assert plan.levers and plan.fundraise is not None
