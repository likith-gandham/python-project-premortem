import json

import numpy as np
import pytest

from premortem import run, summarize
from premortem.analysis import cause_of_death, driver_analysis, evaluate_levers, survival_curve, what_must_be_true
from premortem.cli import main
from premortem.report import write_report


def _res(plan):
    return run(plan)


def test_survival_curve_is_monotone(plan):
    curve = survival_curve(_res(plan))
    assert curve[0] == 1.0
    assert np.all(np.diff(curve) <= 0)


def test_driver_directions_make_business_sense(plan):
    drivers = {d.key: d for d in driver_analysis(_res(plan))}
    assert drivers["cac"].worst_is_high          # expensive customers hurt
    assert drivers["monthly_churn"].worst_is_high
    assert not drivers["arpu"].worst_is_high     # low prices hurt
    assert "fixed_costs" not in drivers          # fixed input, no uncertainty


def test_cause_of_death_shares_sum_to_one(plan):
    res = _res(plan)
    drivers = driver_analysis(res)
    causes = cause_of_death(res, drivers)
    assert 0 < res.survival_rate < 1, "fixture should have both survivors and failures"
    assert sum(d.share_of_deaths for d in drivers) == pytest.approx(1.0)
    assert np.all(causes[res.survived] == "")
    assert np.all(causes[~res.survived] != "")


def test_reverse_stress_test_hits_target(plan):
    from premortem.engine import apply_changes, simulate
    res = _res(plan)
    req = what_must_be_true(plan, res.inputs, "cac", target=0.8)
    assert req.achievable and req.direction == "at most"
    sub = {k: v[:2000] for k, v in res.inputs.items()}
    check = simulate(plan, apply_changes(sub, {"cac": {"set": req.threshold}}))
    assert check.survival_rate >= 0.8


def test_unreachable_target_is_reported(plan):
    res = _res(plan)
    req = what_must_be_true(plan, res.inputs, "organic_signups", target=0.999)
    assert not req.achievable and req.threshold is None


def test_levers_ranked_and_signed(plan):
    levers = evaluate_levers(_res(plan))
    assert levers[0].name == "Cut fixed costs" and levers[0].delta > 0
    assert levers[-1].name == "Burn money" and levers[-1].delta < 0


def test_full_report_writes_valid_outputs(plan, tmp_path):
    res = _res(plan)
    summary = summarize(res)
    paths = write_report(res, summary, tmp_path, save_pngs=False)
    html = paths["html"].read_text()
    assert "TestCo survives" in html and "What must be true" in html
    data = json.loads(paths["json"].read_text())
    assert data["survival_rate"] == pytest.approx(res.survival_rate)
    assert len(paths["csv"].read_text().splitlines()) == res.n + 1


def test_cli_end_to_end(tmp_path):
    plan_path = tmp_path / "plan.yaml"
    assert main(["init", str(plan_path)]) == 0
    assert main(["init", str(plan_path)]) == 1          # refuses to overwrite
    assert main(["run", str(plan_path), "--sims", "500", "-o", str(tmp_path / "out"), "--no-png"]) == 0
    assert (tmp_path / "out" / "report.html").exists()


def test_cli_reports_bad_plan(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("assumptions: {starting_cash: 10}\n")
    assert main(["run", str(bad)]) == 2
    assert "Missing required assumption" in capsys.readouterr().err
