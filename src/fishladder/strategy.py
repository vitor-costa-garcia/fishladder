"""Query strategies."""

from __future__ import annotations

import copy
import math
import random
import statistics
import typing
from collections.abc import Mapping

from fishladder.base import Output, QueryStrategy

__all__ = ["ExpectedError", "QueryAll", "QueryByCommittee", "RandomPerOutput"]


def _poisson(rate: float, rng: random.Random) -> int:
    """Draw from a Poisson distribution with Knuth's algorithm, which is fine for small rates."""
    threshold = math.exp(-rate)
    k = 0
    p = rng.random()
    while p > threshold:
        k += 1
        p *= rng.random()
    return k


class QueryAll(QueryStrategy):
    """Query every output, as long as the budget allows it.

    Examples
    --------

    >>> from fishladder import strategy

    >>> strategy.QueryAll().utilities_one({}, {}, costs={"a": 1.0, "b": 2.0})
    {'a': 1.0, 'b': 1.0}

    """

    def utilities_one(
        self, x: dict, y_pred: dict, costs: Mapping[Output, float]
    ) -> dict[Output, float]:
        return {o: 1.0 for o in costs}


class RandomPerOutput(QueryStrategy):
    """Query each output independently with a fixed probability.

    Parameters
    ----------
    p
        The probability of querying each output.
    seed
        Random number generator seed for reproducibility.

    Examples
    --------

    >>> from fishladder import strategy

    >>> s = strategy.RandomPerOutput(p=0.5, seed=42)
    >>> s.utilities_one({}, {}, costs={"a": 1.0, "b": 1.0, "c": 1.0})
    {'a': 0.0, 'b': 1.0, 'c': 1.0}

    """

    def __init__(self, p: float, seed: int | None = None):
        if not 0 <= p <= 1:
            raise ValueError(f"p must be between 0 and 1, got {p}")
        self.p = p
        self.seed = seed
        self._rng = random.Random(seed)

    def utilities_one(
        self, x: dict, y_pred: dict, costs: Mapping[Output, float]
    ) -> dict[Output, float]:
        return {o: float(self._rng.random() < self.p) for o in costs}


class QueryByCommittee(QueryStrategy):
    """Query the outputs on which a committee of regressors disagrees the most.

    For each output, the strategy trains a committee of copies of `regressor` on the revealed
    values of that output, with online bagging: each member learns from each revealed value a
    Poisson(1) number of times, so that members see different resamples of the data. The utility
    of an output is the standard deviation of the members' predictions, in the units of the
    output.

    The committee is separate from the model being evaluated, and only sees revealed labels.
    Outputs with fewer than `min_labels` revealed labels get an infinite utility, so that they
    are queried first. A committee trained on a handful of labels can agree while being wrong,
    for instance when all its members learned from the same single label, and that output would
    then rarely be queried again. For the same reason, utilities are floored at a tiny positive
    value: members that agree exactly would otherwise rule the output out for good, since the
    committee only learns from queried labels.

    Utilities of different outputs are in different units. When outputs share a budget, they are
    compared directly, so per-output budgets (`budget.PerOutput`) are usually a better fit.

    Parameters
    ----------
    regressor
        A single-output regressor, such as any River regressor. It is copied for each committee
        member and each output, and is not modified.
    n_members
        The number of members in each committee.
    min_labels
        The number of labels an output needs before its committee is trusted.
    seed
        Random number generator seed for reproducibility.

    Examples
    --------

    >>> from river import linear_model
    >>> from fishladder import strategy

    >>> s = strategy.QueryByCommittee(linear_model.LinearRegression(), n_members=3, seed=42)
    >>> s.utilities_one({"x": 1.0}, {}, costs={"a": 1.0})
    {'a': inf}

    >>> for i in range(20):
    ...     s.learn_one({"x": i / 10}, {"a": i / 5}, y_pred={})
    >>> utilities = s.utilities_one({"x": 1.0}, {}, costs={"a": 1.0})
    >>> 0 < utilities["a"] < math.inf
    True

    """

    _floor = 1e-12

    def __init__(
        self,
        regressor: typing.Any,
        n_members: int = 5,
        min_labels: int = 10,
        seed: int | None = None,
    ):
        if n_members < 2:
            raise ValueError(f"n_members must be at least 2, got {n_members}")
        if min_labels < 1:
            raise ValueError(f"min_labels must be at least 1, got {min_labels}")
        self.regressor = regressor
        self.n_members = n_members
        self.min_labels = min_labels
        self.seed = seed
        self._committees: dict[Output, list[typing.Any]] = {}
        self._n_labels: dict[Output, int] = {}
        self._rngs: dict[Output, random.Random] = {}

    def utilities_one(
        self, x: dict, y_pred: dict, costs: Mapping[Output, float]
    ) -> dict[Output, float]:
        utilities = {}
        for o in costs:
            if self._n_labels.get(o, 0) < self.min_labels:
                utilities[o] = math.inf
            else:
                committee = self._committees[o]
                spread = statistics.pstdev(m.predict_one(x) for m in committee)
                utilities[o] = max(spread, self._floor)
        return utilities

    def learn_one(self, x: dict, y: dict, y_pred: dict) -> None:
        for o, y_o in y.items():
            if o not in self._committees:
                self._committees[o] = [copy.deepcopy(self.regressor) for _ in range(self.n_members)]
                # One generator per output, seeded from its name, so that results do not depend
                # on the order in which outputs are revealed
                seed = None if self.seed is None else f"{self.seed}:{o!r}"
                self._rngs[o] = random.Random(seed)
                self._n_labels[o] = 0
            self._n_labels[o] += 1
            for member in self._committees[o]:
                for _ in range(_poisson(1.0, self._rngs[o])):
                    member.learn_one(x, y_o)


class ExpectedError(QueryStrategy):
    """Query the outputs on which the model is expected to make the largest error.

    For each output, the strategy trains a copy of `regressor` to predict the absolute error of
    the model from the features. Each time an output is revealed, the error the model made on it
    is used as a training example. The utility of an output is its predicted error, in the units
    of the output.

    The error model only learns from queried instances, which are the ones it expected to be
    hard. Its estimates can therefore be biased towards high errors.

    Outputs whose error has never been observed get an infinite utility, so that they are queried
    first. Predicted errors are floored at a tiny positive value, so that an output is never
    ruled out for good because its error model predicts a negative error.

    Utilities of different outputs are in different units. When outputs share a budget, they are
    compared directly, so per-output budgets (`budget.PerOutput`) are usually a better fit.

    Parameters
    ----------
    regressor
        A single-output regressor to estimate errors with, such as any River regressor. It is
        copied for each output and is not modified.

    Examples
    --------

    >>> from river import linear_model
    >>> from fishladder import strategy

    >>> s = strategy.ExpectedError(linear_model.LinearRegression())
    >>> s.utilities_one({"x": 1.0}, {"a": 0.0}, costs={"a": 1.0})
    {'a': inf}

    >>> for i in range(50):
    ...     x = {"x": i % 2}
    ...     s.learn_one(x, {"a": 2.0 * x["x"]}, y_pred={"a": 0.0})
    >>> low = s.utilities_one({"x": 0}, {"a": 0.0}, costs={"a": 1.0})["a"]
    >>> high = s.utilities_one({"x": 1}, {"a": 0.0}, costs={"a": 1.0})["a"]
    >>> low < high
    True

    """

    _floor = 1e-12

    def __init__(self, regressor: typing.Any):
        self.regressor = regressor
        self._error_models: dict[Output, typing.Any] = {}

    def utilities_one(
        self, x: dict, y_pred: dict, costs: Mapping[Output, float]
    ) -> dict[Output, float]:
        utilities = {}
        for o in costs:
            error_model = self._error_models.get(o)
            if error_model is None:
                utilities[o] = math.inf
            else:
                utilities[o] = max(error_model.predict_one(x), self._floor)
        return utilities

    def learn_one(self, x: dict, y: dict, y_pred: dict) -> None:
        for o, y_o in y.items():
            if o not in y_pred:
                continue
            if o not in self._error_models:
                self._error_models[o] = copy.deepcopy(self.regressor)
            self._error_models[o].learn_one(x, abs(y_o - y_pred[o]))
