"""Query strategies."""

from __future__ import annotations

import abc
import collections
import copy
import math
import random
import statistics
import typing
from collections.abc import Mapping, Sequence

from fishladder.base import Output, QueryStrategy

__all__ = [
    "ExpectedError",
    "QueryAll",
    "QueryByCommittee",
    "RandomPerOutput",
    "Randomized",
    "Split",
]


def _output_rng(seed: int | None, output: Output) -> random.Random:
    # One generator per output, seeded from its name, so that results do not depend on the order
    # in which outputs are seen
    return random.Random(None if seed is None else f"{seed}:{output!r}")


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

    def sample_one(
        self, x: dict, y_pred: dict, costs: Mapping[Output, float]
    ) -> tuple[dict[Output, float], dict[Output, float]]:
        return self.utilities_one(x, y_pred, costs), {o: self.p for o in costs}


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
                self._rngs[o] = _output_rng(self.seed, o)
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


class Randomized(QueryStrategy):
    """Base class for strategies that query each output at random, with a probability derived
    from the utilities of another strategy.

    Deterministic strategies never query the outputs on which they are confident, and so may
    never notice a drift there. Žliobaitė et al. (2014) found that randomizing which instances
    are queried makes active learning more robust to drift. This class handles the sampling, and
    subclasses only define `probability_one`, which turns a utility into a probability given the
    recent utilities of the same output.

    For each output, the class keeps the finite utilities given by `strategy` for the last
    `window` instances, whether they were queried or not. Infinite utilities, such as those of an
    output whose committee is not trained yet, and outputs with no recent utilities get a
    probability of 1. Each output is then sampled independently. A sampled output gets the
    utility given by `strategy`, floored at a tiny positive value, so that a budget shared by
    several outputs still picks the most useful of them. Other outputs get a utility of 0, and
    are not queried.

    Sampling already decides which instances are queried, so this works best with budgets that
    query whatever they can afford, such as `budget.RateBudget` or `budget.PerOutput`, rather than
    `budget.VariableThreshold`. The probabilities should also match the budget: if outputs are
    sampled more often than the budget can afford, it queries the first sampled output it can
    afford, and the probabilities hardly matter. With `rate`, probabilities are rescaled so that
    each output is sampled `rate` times per instance on average, which can be set to what the
    budget affords. For instance, `PerOutput(RateBudget(1.5))` with three outputs costing 1 each
    affords a rate of 0.5.

    Parameters
    ----------
    strategy
        The strategy whose utilities are turned into probabilities. It learns from the same
        labels as this strategy.
    window
        The number of recent utilities kept for each output.
    rate
        If given, probabilities are multiplied by `rate` divided by the average probability over
        the window, including the current instance, so that `probability_one` only needs to give
        relative probabilities. Probabilities are capped at 1, so the average rate can be lower.
    p_min
        The lowest probability of sampling an output, so that every instance can be queried.
    seed
        Random number generator seed for reproducibility.

    References
    ----------
    Žliobaitė, I., Bifet, A., Pfahringer, B., & Holmes, G. (2014). Active learning with drifting
    streaming data. IEEE Transactions on Neural Networks and Learning Systems, 25(1), 27-39.

    Examples
    --------

    A strategy that samples an output with a probability proportional to its utility, relative
    to its recent utilities:

    >>> import statistics
    >>> from fishladder import base, strategy

    >>> class Proportional(strategy.Randomized):
    ...     def probability_one(self, output, utility, recent):
    ...         return utility / statistics.mean(recent)

    >>> class Fixed(base.QueryStrategy):
    ...     def __init__(self):
    ...         self.utilities = iter([1.0, 1.0, 1.0, 4.0])
    ...     def utilities_one(self, x, y_pred, costs):
    ...         return {o: next(self.utilities) for o in costs}

    >>> s = Proportional(Fixed(), rate=0.5, seed=42)
    >>> for _ in range(4):
    ...     utilities, probabilities = s.sample_one({}, {}, costs={"a": 1.0})
    ...     print(utilities, probabilities)
    {'a': 1.0} {'a': 1.0}
    {'a': 1.0} {'a': 0.5}
    {'a': 0.0} {'a': 0.5}
    {'a': 4.0} {'a': 1.0}

    """

    _floor = 1e-12

    def __init__(
        self,
        strategy: QueryStrategy,
        window: int = 200,
        rate: float | None = None,
        p_min: float = 0.0,
        seed: int | None = None,
    ):
        if window < 1:
            raise ValueError(f"window must be at least 1, got {window}")
        if rate is not None and not 0 < rate <= 1:
            raise ValueError(f"rate must be between 0 and 1, got {rate}")
        if not 0 <= p_min <= 1:
            raise ValueError(f"p_min must be between 0 and 1, got {p_min}")
        self.strategy = strategy
        self.window = window
        self.rate = rate
        self.p_min = p_min
        self.seed = seed
        self._utilities: dict[Output, collections.deque[float]] = {}
        self._probabilities: dict[Output, collections.deque[float]] = {}
        self._rngs: dict[Output, random.Random] = {}

    @abc.abstractmethod
    def probability_one(self, output: Output, utility: float, recent: Sequence[float]) -> float:
        """Give the probability of sampling an output.

        Parameters
        ----------
        output
            The name of the output.
        utility
            The utility of the output for the current instance, as given by `strategy`. It is
            always finite.
        recent
            The finite utilities of the output for the previous instances, oldest first, without
            the current one. It is never empty.

        Returns
        -------
        The probability of sampling the output, or, with `rate`, a relative probability. Values
        are clipped to between `p_min` and 1.

        """

    def _probability(self, o: Output, u: float) -> float:
        utilities = self._utilities[o]
        if math.isinf(u):
            return 1.0
        if not utilities:
            utilities.append(u)
            return 1.0
        p = max(self.probability_one(o, u, utilities), 0.0)
        utilities.append(u)
        if self.rate is not None:
            probabilities = self._probabilities[o]
            probabilities.append(p)
            mean = statistics.fmean(probabilities)
            # When every recent probability is 0, there is nothing to tell instances apart
            p = p * self.rate / mean if mean > 0 else self.rate
        return min(max(p, self.p_min), 1.0)

    def sample_one(
        self, x: dict, y_pred: dict, costs: Mapping[Output, float]
    ) -> tuple[dict[Output, float], dict[Output, float]]:
        base = self.strategy.utilities_one(x, y_pred, costs)
        utilities = {}
        probabilities = {}
        for o in costs:
            if o not in self._rngs:
                self._rngs[o] = _output_rng(self.seed, o)
                self._utilities[o] = collections.deque(maxlen=self.window)
                self._probabilities[o] = collections.deque(maxlen=self.window)
            p = self._probability(o, base[o])
            probabilities[o] = p
            sampled = self._rngs[o].random() < p
            utilities[o] = max(base[o], self._floor) if sampled else 0.0
        return utilities, probabilities

    def utilities_one(
        self, x: dict, y_pred: dict, costs: Mapping[Output, float]
    ) -> dict[Output, float]:
        return self.sample_one(x, y_pred, costs)[0]

    def learn_one(self, x: dict, y: dict, y_pred: dict) -> None:
        self.strategy.learn_one(x, y, y_pred)


class Split(QueryStrategy):
    """Mix another strategy with random exploration.

    For each output, with probability `p`, the output is explored: it gets an infinite utility,
    and is queried as soon as the budget affords it. Otherwise, it gets the utility given by
    `strategy`. This is the split strategy of Žliobaitė et al. (2014), which spends part of the
    budget on random queries, so that drifts are noticed even where `strategy` is confident.

    With `budget.VariableThreshold`, explored outputs always pass the threshold and do not change
    it, so the threshold only adapts to the instances chosen by `strategy`. Since explored
    outputs come first in a budget shared by several outputs, `p` should stay small compared to
    what the budget affords.

    If `strategy` is randomized, the probability that an output gets a positive utility is
    reported, as `p` plus `1 - p` times the probability given by `strategy`.

    Parameters
    ----------
    strategy
        The strategy to mix with random exploration. It learns from every revealed label,
        including those of explored outputs.
    p
        The probability of exploring each output.
    seed
        Random number generator seed for reproducibility.

    References
    ----------
    Žliobaitė, I., Bifet, A., Pfahringer, B., & Holmes, G. (2014). Active learning with drifting
    streaming data. IEEE Transactions on Neural Networks and Learning Systems, 25(1), 27-39.

    Examples
    --------

    >>> from fishladder import strategy

    >>> s = strategy.Split(strategy.QueryAll(), p=0.2, seed=1)
    >>> for _ in range(3):
    ...     print(s.utilities_one({}, {}, costs={"a": 1.0, "b": 1.0}))
    {'a': 1.0, 'b': 1.0}
    {'a': 1.0, 'b': 1.0}
    {'a': inf, 'b': 1.0}

    """

    def __init__(self, strategy: QueryStrategy, p: float = 0.1, seed: int | None = None):
        if not 0 <= p <= 1:
            raise ValueError(f"p must be between 0 and 1, got {p}")
        self.strategy = strategy
        self.p = p
        self.seed = seed
        self._rngs: dict[Output, random.Random] = {}

    def sample_one(
        self, x: dict, y_pred: dict, costs: Mapping[Output, float]
    ) -> tuple[dict[Output, float], dict[Output, float]]:
        base, base_probabilities = self.strategy.sample_one(x, y_pred, costs)
        utilities = {}
        for o in costs:
            if o not in self._rngs:
                self._rngs[o] = _output_rng(self.seed, o)
            utilities[o] = math.inf if self._rngs[o].random() < self.p else base[o]
        probabilities = {o: self.p + (1 - self.p) * q for o, q in base_probabilities.items()}
        return utilities, probabilities

    def utilities_one(
        self, x: dict, y_pred: dict, costs: Mapping[Output, float]
    ) -> dict[Output, float]:
        return self.sample_one(x, y_pred, costs)[0]

    def learn_one(self, x: dict, y: dict, y_pred: dict) -> None:
        self.strategy.learn_one(x, y, y_pred)
