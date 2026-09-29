import numpy as np
import pytest

from premortem import plan_from_dict, run, sample_inputs, simulate
from premortem.engine import apply_changes


def test_hand_calculated_months(deterministic_dict):
    # Each month: 100/10 = 10 new customers, no churn, $5 each.
    # Month 1: 10 customers -> revenue 50 -> net 50 - 100 - 100 = -150 -> cash 850
    # Month 2: 20 customers -> revenue 100 -> net -100 -> cash 750
    res = run(plan_from_dict(deterministic_dict))
    assert res.customers[0, :3] == pytest.approx([10, 20, 30])
    assert res.cash[0, :4] == pytest.approx([1000, 850, 750, 700])
    assert res.survival_rate == 1.0


def test_breakeven_month_detected(deterministic_dict):
    # Net = 50t - 200 turns positive when t > 4, i.e. month 5.
    res = run(plan_from_dict(deterministic_dict))
    assert np.all(res.breakeven_month == 5)


def test_death_is_permanent(deterministic_dict):
    deterministic_dict["assumptions"]["starting_cash"] = 200
    res = run(plan_from_dict(deterministic_dict))
    assert np.all(res.death_month == 2)
    after = res.cash[:, 2:]
    assert np.allclose(after, after[:, :1])       # cash frozen after death
    assert np.all(res.mrr[:, 2:] == 0)            # no revenue after death


def test_fundraise_rescues_only_when_milestone_hit(deterministic_dict):
    d = deterministic_dict
    d["assumptions"]["starting_cash"] = 200
    d["fundraise"] = {"month": 1, "amount": 10_000, "mrr_milestone": 40}
    assert run(plan_from_dict(d)).survival_rate == 1.0
    d["fundraise"]["mrr_milestone"] = 60           # month-1 revenue is only 50
    assert run(plan_from_dict(d)).survival_rate == 0.0


def test_seed_makes_runs_reproducible(plan):
    a, b = run(plan), run(plan)
    assert np.array_equal(a.cash, b.cash)
    c = run(plan, seed=999)
    assert not np.array_equal(a.cash, c.cash)


def test_common_random_numbers(plan):
    inputs = sample_inputs(plan)
    base = simulate(plan, inputs)
    cheaper = simulate(plan, apply_changes(inputs, {"fixed_costs": {"scale": 0.5}}))
    # Same futures, lower costs: no run can end with less cash.
    assert np.all(cheaper.final_cash >= base.final_cash - 1e-6)
    assert cheaper.survival_rate >= base.survival_rate


def test_apply_changes_respects_bounds(plan):
    inputs = sample_inputs(plan, n=500)
    out = apply_changes(inputs, {"monthly_churn": {"add": 5.0}})
    assert out["monthly_churn"].max() <= 1.0
    assert inputs["monthly_churn"].max() < 1.0    # original untouched
