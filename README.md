# Premortem

**Find out how your business is most likely to die, before it does.**

[![CI](https://github.com/YOUR-USERNAME/premortem/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR-USERNAME/premortem/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Every business plan contains a spreadsheet, and every spreadsheet tells the same lie: one number per assumption, one future, one answer. Real companies don't live in that future. Customer acquisition gets more expensive, churn creeps up, the funding round slips.

Premortem takes a business plan written as **ranges** instead of single numbers, simulates **10,000 possible futures** month by month, and answers the questions a board or investor actually asks:

| Question | What Premortem does |
|---|---|
| Will we survive? | Survival probability and a survival curve over the horizon |
| When we fail, *why*? | Attributes every failed future to the assumption that killed it |
| Which assumptions matter? | Ranks drivers by how much they swing survival |
| What has to be true? | Reverse stress test: the value each driver must hit for 80% survival |
| What should we do? | Replays the same 10,000 futures under each management decision |

The name comes from the **premortem**, a decision-making technique from psychologist Gary Klein: before committing to a plan, imagine it has already failed and work out why. This project turns that exercise into a quantitative tool.

![Dashboard](docs/images/dashboard.png)

---

## A worked example

`examples/saas_startup.yaml` describes **Ledgerly**, a fictional seed-stage SaaS company with about 12 months of runway and a $4M round planned for month 16. On paper, its median-case spreadsheet looks healthy: LTV/CAC of 3.3x, payback in 8 months.

```
$ premortem run examples/saas_startup.yaml

  PREMORTEM  Ledgerly
  ──────────────────────────────────────────────────────────────
  Survival over 36 months   ████████████████████░░░░░░░░░░  65%  (Fragile)
  Median time of death       month 14

  Top causes of death
    Customer acquisition cost          ██████░░░░░░░░░░ 39%
    Failed fundraise                   ████░░░░░░░░░░░░ 23%
    Organic sign-ups / month           ███░░░░░░░░░░░░░ 18%

  What must be true for 80% survival
    Customer acquisition cost          at most $844
    Fixed costs / month                at most $79K
    Organic sign-ups / month           at least 38

  Decisions tested
    Delay two engineering hires                +25 pts  -> 90%
    Raise prices 15% (churn +10%)              +16 pts  -> 81%
    Push annual plans (churn -25%)              +8 pts  -> 73%
    Cut marketing 30%                           -3 pts  -> 62%
    Double down on marketing (+50%)             -9 pts  -> 56%
```

What the simulation reveals that the spreadsheet hides:

- **One in three futures ends in bankruptcy**, clustered just before the planned raise.
- The plan needs CAC at or below **$844**, but only **29%** of the founders' own assumed range clears that bar. The median assumption is quietly optimistic.
- The instinct to "pour fuel on growth" with 50% more marketing **lowers** survival by 9 points: at this CAC, each marketing dollar burns cash faster than it returns it.
- **Delaying two hires is the single best decision**, lifting survival from 65% to 90%, well ahead of any growth lever.

A second example, `examples/coffee_subscription.yaml` (a bootstrapped D2C coffee subscription), shows a different failure pattern: 95% of its failures happen in year one, and cutting paid social actually *improves* survival because its customers cost more than they are worth.

Each run also writes a self-contained HTML report written like an investment memo. Open [`docs/sample_report_ledgerly.html`](docs/sample_report_ledgerly.html) after cloning to see one.

![Report](docs/images/report_hero.png)

---

## Quick start

```bash
git clone https://github.com/YOUR-USERNAME/premortem.git
cd premortem
pip install -e .

premortem run examples/saas_startup.yaml --open     # simulate and open the report
premortem init my_plan.yaml                         # starter plan for your own idea
premortem run my_plan.yaml --sims 20000 --seed 1
```

Each run writes to `reports/<plan name>/`:

| File | Contents |
|---|---|
| `report.html` | The memo: verdict, findings, charts, reverse stress test, decisions |
| `summary.json` | Every metric, machine-readable |
| `simulations.csv` | One row per simulated future: inputs drawn, outcome, cause of death |
| `charts/*.png` | Each chart as an image, for slides |

It also works as a library:

```python
from premortem import load_plan, run, summarize

result = run(load_plan("examples/saas_startup.yaml"))
summary = summarize(result)
print(summary["survival_rate"], summary["narrative"][2])
```

---

## Writing a plan

Plans are plain YAML so non-programmers can write them. Any assumption can be a single number or a range:

```yaml
company: "Ledgerly"
horizon_months: 36
survival_target: 0.80

assumptions:
  starting_cash: 1200000
  arpu:          {dist: pert, low: 120, mode: 149, high: 175}    # $ per customer per month
  cac:           {dist: pert, low: 550, mode: 900, high: 1800}   # $ to win a customer
  monthly_churn: {dist: pert, low: 0.015, mode: 0.035, high: 0.07}
  fixed_costs:   {dist: normal, mean: 85000, sd: 7000, low: 60000}
  # ...

fundraise:                        # only closes if the company is alive AND hits the milestone
  month: 16
  amount: 4000000
  mrr_milestone: 120000
  close_probability: 0.65

levers:                           # decisions to test on the same futures
  - name: "Raise prices 15% (churn +10%)"
    changes: {arpu: {scale: 1.15}, monthly_churn: {scale: 1.10}}
```

Supported distributions: `pert` (recommended for three-point estimates), `triangular`, `uniform`, `normal` (with optional clipping), `lognormal`, or a fixed number. Levers can `scale`, `add` to, or `set` any assumption, and can change several at once to capture trade-offs such as higher prices causing more churn.

<details>
<summary>All assumptions</summary>

| Key | Meaning | Required |
|---|---|---|
| `starting_cash` | Cash in the bank today | yes |
| `starting_customers` | Paying customers today | no (0) |
| `arpu` | Revenue per customer per month | yes |
| `arpu_growth` | Monthly price growth | no (0) |
| `gross_margin` | Share of revenue left after direct costs | yes |
| `cac` | Marketing cost to acquire one customer | yes |
| `marketing_spend` | Monthly marketing budget | yes |
| `marketing_growth` | Monthly growth in the marketing budget | no (0) |
| `organic_signups` | Customers per month that cost nothing to acquire | no (0) |
| `monthly_churn` | Share of customers lost each month | yes |
| `fixed_costs` | Team, rent, tools per month | yes |
| `fixed_cost_growth` | Monthly growth in fixed costs | no (0) |
| `demand_volatility` | How much new-customer volume swings month to month | no (0.15) |

Typos get a helpful error: `Unknown assumption 'montly_churn'. Did you mean 'monthly_churn'?`
</details>

---

## How it works

**1. Sampling.** For each of 10,000 futures, one value is drawn for every uncertain assumption. Beta-PERT is used for three-point estimates because it is the standard in project finance and puts realistic weight on the extremes.

**2. Simulation.** Each future is played forward month by month with a customer model:

```
new customers = (marketing spend / CAC + organic sign-ups) x demand shock
customers     = customers x (1 - churn) + new customers
net income    = customers x ARPU x gross margin - fixed costs - marketing
cash          = cash + net income (+ funding, if the round closes)
```

A company dies the first month its cash goes negative and stays dead. The funding round only closes if the company is still alive, has reached the revenue milestone, *and* the investors say yes. All 10,000 futures run in parallel as NumPy arrays, so a full analysis takes about two seconds.

**3. Cause-of-death attribution.** For each failed future, every assumption is scored by how far it landed on its bad side (as a percentile), weighted by how strongly that assumption drives outcomes overall (Spearman rank correlation). The highest score is recorded as the cause of death.

**4. Reverse stress test ("what must be true").** For the top drivers, a bisection search finds the value that brings survival to the target while every other assumption stays uncertain. It then reports how much of the founder's own assumed range already meets that bar, which is often the most sobering number in the report.

**5. Decision testing with common random numbers.** Every lever is replayed against the *exact same* 10,000 futures. Differences in survival therefore come from the decision, not from sampling noise, the same variance-reduction technique used in quantitative finance.

More detail, including limitations, is in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Project structure

```
premortem/
├── premortem/
│   ├── distributions.py   # PERT, triangular, normal, lognormal... with validation
│   ├── config.py          # YAML plan loading, defaults, helpful errors
│   ├── engine.py          # vectorised monthly Monte Carlo simulation
│   ├── analysis.py        # survival, drivers, cause of death, reverse stress test, levers
│   ├── charts.py          # matplotlib charts
│   ├── report.py          # HTML memo, JSON and CSV export
│   └── cli.py             # `premortem run` / `premortem init`
├── examples/              # two fully worked business plans
├── tests/                 # 39 tests, including hand-calculated cases
├── docs/                  # methodology, sample reports, images
└── .github/workflows/     # CI on Python 3.10 to 3.12
```

## Business concepts covered

Unit economics (LTV, CAC, LTV/CAC, payback period), cash runway and burn, "default alive" versus dependence on fundraising, milestone-based financing, pricing trade-offs, sensitivity analysis, scenario planning, and decision-making under uncertainty.

## Roadmap

- Cohort-based churn that declines with customer tenure
- Correlated assumptions (e.g. higher prices with higher CAC)
- Import assumptions directly from an Excel model
- A Streamlit front end for non-technical users

## License

MIT
