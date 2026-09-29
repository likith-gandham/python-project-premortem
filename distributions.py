"""Probability distributions for uncertain business assumptions.

Business plans are usually written as single numbers ("churn will be 3%").
Premortem lets every assumption be a *range* instead, which is how
experienced operators and investors actually reason about the future.

Supported in YAML::

    arpu: 149                                        # a fixed number
    arpu: {dist: uniform,    low: 120, high: 170}
    arpu: {dist: triangular, low: 120, mode: 149, high: 170}
    arpu: {dist: pert,       low: 120, mode: 149, high: 170}   # recommended
    arpu: {dist: normal,     mean: 149, sd: 12, low: 0}        # optional clipping
    arpu: {dist: lognormal,  median: 149, sigma: 0.2}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _fmt(x: float) -> str:
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) < 1:
        return f"{x:.3g}"
    return f"{x:,.2f}".rstrip("0").rstrip(".")


class Distribution:
    """Base class for all assumption distributions."""

    stochastic: bool = True

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> str:  # pragma: no cover
        raise NotImplementedError

    @property
    def mean(self) -> float:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class Fixed(Distribution):
    value: float
    stochastic = False

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return np.full(n, float(self.value))

    def describe(self) -> str:
        return f"fixed at {_fmt(self.value)}"

    @property
    def mean(self) -> float:
        return float(self.value)


@dataclass(frozen=True)
class Uniform(Distribution):
    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError("uniform needs low < high")

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(self.low, self.high, n)

    def describe(self) -> str:
        return f"anywhere from {_fmt(self.low)} to {_fmt(self.high)}"

    @property
    def mean(self) -> float:
        return (self.low + self.high) / 2


@dataclass(frozen=True)
class Triangular(Distribution):
    low: float
    mode: float
    high: float

    def __post_init__(self) -> None:
        if not (self.low <= self.mode <= self.high and self.low < self.high):
            raise ValueError("triangular needs low <= mode <= high and low < high")

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.triangular(self.low, self.mode, self.high, n)

    def describe(self) -> str:
        return f"{_fmt(self.low)} to {_fmt(self.high)}, most likely {_fmt(self.mode)}"

    @property
    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3


@dataclass(frozen=True)
class Pert(Distribution):
    """Beta-PERT: the project-finance standard for expert three-point estimates.

    Smoother than a triangle and puts less weight on the extremes, which
    matches how people's "worst case / likely / best case" guesses behave.
    """

    low: float
    mode: float
    high: float
    lamb: float = 4.0

    def __post_init__(self) -> None:
        if not (self.low <= self.mode <= self.high and self.low < self.high):
            raise ValueError("pert needs low <= mode <= high and low < high")
        if self.lamb <= 0:
            raise ValueError("pert needs lamb > 0")

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        span = self.high - self.low
        a = 1 + self.lamb * (self.mode - self.low) / span
        b = 1 + self.lamb * (self.high - self.mode) / span
        return self.low + rng.beta(a, b, n) * span

    def describe(self) -> str:
        return f"{_fmt(self.low)} to {_fmt(self.high)}, most likely {_fmt(self.mode)}"

    @property
    def mean(self) -> float:
        return (self.low + self.lamb * self.mode + self.high) / (self.lamb + 2)


@dataclass(frozen=True)
class Normal(Distribution):
    mean_: float
    sd: float
    low: float | None = None
    high: float | None = None

    def __post_init__(self) -> None:
        if self.sd <= 0:
            raise ValueError("normal needs sd > 0")

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        x = rng.normal(self.mean_, self.sd, n)
        lo = -np.inf if self.low is None else self.low
        hi = np.inf if self.high is None else self.high
        return np.clip(x, lo, hi)

    def describe(self) -> str:
        return f"around {_fmt(self.mean_)} (± {_fmt(self.sd)})"

    @property
    def mean(self) -> float:
        return float(self.mean_)


@dataclass(frozen=True)
class LogNormal(Distribution):
    median: float
    sigma: float

    def __post_init__(self) -> None:
        if self.median <= 0 or self.sigma <= 0:
            raise ValueError("lognormal needs median > 0 and sigma > 0")

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.lognormal(np.log(self.median), self.sigma, n)

    def describe(self) -> str:
        return f"typically {_fmt(self.median)}, with a long right tail"

    @property
    def mean(self) -> float:
        return float(self.median * np.exp(self.sigma**2 / 2))


_REGISTRY: dict[str, type[Distribution]] = {
    "uniform": Uniform,
    "triangular": Triangular,
    "pert": Pert,
    "normal": Normal,
    "lognormal": LogNormal,
    "fixed": Fixed,
}


def from_spec(spec: Any, name: str = "assumption") -> Distribution:
    """Turn a YAML value (number or dict) into a Distribution."""
    if isinstance(spec, Distribution):
        return spec
    if isinstance(spec, bool):
        raise ValueError(f"'{name}': expected a number or a distribution, got {spec!r}")
    if isinstance(spec, (int, float)):
        return Fixed(float(spec))
    if isinstance(spec, dict):
        params = dict(spec)
        kind = str(params.pop("dist", "")).lower().strip()
        if kind not in _REGISTRY:
            options = ", ".join(sorted(_REGISTRY))
            raise ValueError(f"'{name}': unknown distribution '{kind}'. Choose one of: {options}")
        if kind == "normal" and "mean" in params:
            params["mean_"] = params.pop("mean")
        try:
            clean = {k: (None if v is None else float(v)) for k, v in params.items()}
            return _REGISTRY[kind](**clean)
        except TypeError as exc:
            raise ValueError(f"'{name}': wrong parameters for '{kind}' ({exc})") from None
        except ValueError as exc:
            raise ValueError(f"'{name}': {exc}") from None
    raise ValueError(f"'{name}': expected a number or a distribution, got {spec!r}")
