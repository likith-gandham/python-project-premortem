# Methodology

This document explains the modelling choices in Premortem, why they were made, and where the model's limits are.

## Why Monte Carlo instead of scenarios

The usual approach to uncertainty in a business plan is three scenarios: base, upside, downside. It has two problems. First, the downside case usually moves every assumption to its bad value at once, which is far less likely than it looks and so gets dismissed. Second, it says nothing about *probability*: a founder cannot tell whether the downside is a 5% or a 40% risk.

Monte Carlo simulation draws every assumption independently from its own range, many thousands of times. The result is a distribution of outcomes, from which survival probabilities, percentiles and driver rankings can be read directly.

## Distributions

People naturally give three-point estimates ("churn is probably 3.5%, could be as low as 1.5% or as high as 7%"). Premortem recommends the **Beta-PERT** distribution for these. Compared with a triangular distribution, PERT concentrates more probability near the most likely value and less in the tails, which matches how expert estimates behave. Its mean is `(low + 4 x mode + high) / 6`.

Also available: uniform (no idea where in the range), normal with optional clipping (symmetric noise such as salary costs), and lognormal (positive values with a long right tail).

Every sampled value is clipped to a sensible range (churn between 0 and 100%, CAC above zero, and so on) so that no combination of inputs can produce nonsense.

## Operating model

Each month, for each simulated future:

1. **Acquisition.** New customers = marketing spend / CAC + organic sign-ups, multiplied by a demand shock. The shock is log-normal with mean 1 and a user-set volatility, so some months are better than others without biasing the average.
2. **Retention.** Existing customers are reduced by the churn rate.
3. **Revenue and profit.** Revenue = customers x ARPU. Net income = revenue x gross margin - fixed costs - marketing spend. Prices, marketing and fixed costs can each grow monthly.
4. **Financing.** If a funding round is configured, it is attempted in its month. It closes only if the company is still alive, monthly revenue has reached the milestone, and a random draw falls under the close probability.
5. **Death.** If cash is negative at month end, the company dies. Dead companies stay dead: no revenue, no costs, no rescue.

The simulation is fully vectorised: all futures advance together as NumPy arrays, one month at a time.

## Outcome score

Several analyses need to rank futures from worst to best. Premortem uses a single score: a company that died scores the month it died; a survivor scores above every dead company, ordered by the cash it ends with. A company that dies in month 20 therefore ranks above one that dies in month 8, and every survivor ranks above both.

## Driver ranking (tornado)

For each uncertain assumption, futures are split by where that assumption landed. Survival is measured among futures in the worst 10% of its range and in the best 10%. The gap between the two is the assumption's **swing**. This is more intuitive for business audiences than a correlation coefficient: "if CAC lands in its worst tenth, survival falls to 29%".

## Cause-of-death attribution

For each failed future, Premortem asks which assumption hurt it most. For every assumption:

- compute its percentile within the simulation, oriented so that 1.0 is the worst possible draw;
- subtract 0.5 so that a median draw contributes nothing;
- multiply by the absolute Spearman rank correlation between that assumption and the outcome score, so that assumptions that barely matter cannot be blamed for a failure.

The assumption with the highest score is recorded as the cause. The funding round is treated as an extra binary factor for futures that were alive to attempt it.

This is a heuristic attribution, not a causal proof. Failures usually have several contributing factors; the attribution names the one most out of line with what the plan assumed.

## Reverse stress test

For each of the top three drivers, Premortem fixes that assumption at a single value (while all others remain uncertain) and measures survival. It first checks the two ends of the plausible range (0.5th to 99.5th percentile of the assumed distribution). If neither end reaches the survival target, the target is reported as not reachable through that assumption alone. Otherwise a bisection search (22 iterations) finds the boundary value.

Finally it reports the share of the originally assumed range that already meets the requirement. A low share means the plan's success depends on the assumption landing in its optimistic tail.

To keep the search fast, it runs on the first 2,000 futures.

## Common random numbers

All random draws (assumptions, demand shocks, funding decisions) are generated once, up front. When a decision is tested, the same draws are reused with only the decision's changes applied. Two effects follow:

- the difference in survival between the plan and a decision is due to the decision, not to sampling noise, so even small effects are detectable with modest simulation counts;
- in the reverse stress test, survival changes smoothly as the fixed value moves, which is what makes bisection reliable.

This is a standard variance-reduction technique in simulation and quantitative finance.

## Limitations

- **Independence.** Assumptions are drawn independently. In reality, some move together (a higher price may raise CAC). Correlated sampling is on the roadmap. Levers can already change several assumptions at once to capture a known trade-off.
- **Constant churn.** Churn is a single monthly rate. Real churn is usually highest among new customers and falls with tenure.
- **No working capital.** Cash moves in the month revenue is earned. Businesses with inventory or long payment terms will need an adjustment.
- **Mechanical management.** The simulated company keeps following its plan even as cash runs low. Real founders cut costs when they see trouble, so absolute survival rates are conservative. Comparisons between plans and decisions remain valid because every alternative shares the same assumption.
- **Garbage in, garbage out.** The model is only as good as the ranges supplied. Its main value is making those ranges explicit and showing which ones deserve the most scrutiny.
