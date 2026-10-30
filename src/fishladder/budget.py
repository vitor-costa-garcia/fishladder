"""Budget managers."""

from __future__ import annotations

import math

from fishladder.base import BudgetManager

__all__ = ["RateBudget", "TotalBudget", "Unlimited"]


class TotalBudget(BudgetManager):
    """A fixed amount of cost to spend over the whole stream.

    Parameters
    ----------
    total
        The total cost that can be spent.

    Examples
    --------

    >>> from fishladder import budget

    >>> b = budget.TotalBudget(3.0)
    >>> sorted(b.select_one({"a": 0.9, "b": 0.5, "c": 0.1}, costs={"a": 2.0, "b": 2.0, "c": 1.0}))
    ['a', 'c']
    >>> b.update(3.0)
    >>> b.available
    0.0

    """

    def __init__(self, total: float):
        super().__init__()
        if total < 0:
            raise ValueError(f"total must be non-negative, got {total}")
        self.total = total

    @property
    def available(self) -> float:
        return self.total - self.spent


class RateBudget(BudgetManager):
    """A budget that grows by a fixed amount with each instance.

    After `n` instances, at most `rate * n` cost has been spent. Unspent budget carries over to
    later instances. This is the cost-based counterpart of the usual "label a fraction of the
    stream" budget.

    Parameters
    ----------
    rate
        The cost made available with each instance.

    Examples
    --------

    >>> from fishladder import budget

    >>> b = budget.RateBudget(0.5)
    >>> b.available
    0.5
    >>> b.update(0.0)
    >>> b.available
    1.0

    """

    def __init__(self, rate: float):
        super().__init__()
        if rate < 0:
            raise ValueError(f"rate must be non-negative, got {rate}")
        self.rate = rate

    @property
    def available(self) -> float:
        return self.rate * (self.n_seen + 1) - self.spent


class Unlimited(BudgetManager):
    """No budget constraint. Useful as a reference for fully supervised learning."""

    @property
    def available(self) -> float:
        return math.inf
