import numpy as np
import pytest

from premortem.distributions import Fixed, LogNormal, Normal, Pert, Triangular, Uniform, from_spec

RNG = lambda: np.random.default_rng(0)  # noqa: E731


@pytest.mark.parametrize("dist", [
    Uniform(1, 3), Triangular(1, 2, 5), Pert(1, 2, 5), Normal(10, 2), LogNormal(5, 0.3),
])
def test_sample_mean_matches_theory(dist):
    x = dist.sample(RNG(), 200_000)
    assert x.mean() == pytest.approx(dist.mean, rel=0.02)


@pytest.mark.parametrize("dist", [Uniform(1, 3), Triangular(1, 2, 5), Pert(1, 2, 5)])
def test_bounded_distributions_stay_in_bounds(dist):
    x = dist.sample(RNG(), 50_000)
    assert x.min() >= dist.low and x.max() <= dist.high


def test_normal_clipping():
    x = Normal(0, 1, low=-0.5, high=0.5).sample(RNG(), 10_000)
    assert x.min() == -0.5 and x.max() == 0.5


def test_fixed_is_not_stochastic():
    d = Fixed(4.0)
    assert not d.stochastic
    assert np.all(d.sample(RNG(), 5) == 4.0)


def test_from_spec_numbers_and_dicts():
    assert isinstance(from_spec(3), Fixed)
    assert isinstance(from_spec({"dist": "pert", "low": 1, "mode": 2, "high": 3}), Pert)
    assert isinstance(from_spec({"dist": "normal", "mean": 5, "sd": 1}), Normal)


@pytest.mark.parametrize("spec, message", [
    ({"dist": "banana", "low": 1}, "unknown distribution"),
    ({"dist": "pert", "low": 5, "mode": 2, "high": 3}, "low <= mode <= high"),
    ({"dist": "uniform", "low": 1}, "wrong parameters"),
    ("seven", "expected a number"),
    (True, "expected a number"),
])
def test_from_spec_errors_are_helpful(spec, message):
    with pytest.raises(ValueError, match=message):
        from_spec(spec, "churn")
