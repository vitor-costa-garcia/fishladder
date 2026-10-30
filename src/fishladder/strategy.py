"""Query strategies."""

from __future__ import annotations

import random
from collections.abc import Mapping

from fishladder.base import Output, QueryStrategy

__all__ = ["QueryAll", "RandomPerOutput"]


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
