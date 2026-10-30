"""Budget managers."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping

from fishladder.base import BudgetManager, Output

__all__ = ["PerOutput", "RateBudget", "TotalBudget", "Unlimited"]


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
    >>> b.update({"a": 2.0, "c": 1.0})
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
    >>> b.update({})
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


class PerOutput(BudgetManager):
    """Give each output its own share of a budget.

    With a single budget, cheap outputs can spend it before enough of it builds up for an
    expensive output, which may then never be queried. Here, each output gets its own copy of
    `budget`, scaled by its share, and only spends from it. Each output is queried whenever its
    utility is positive and its own budget can afford it, independently of the other outputs.

    Parameters
    ----------
    budget
        The budget to split, such as `RateBudget(1.5)`. It is copied for each output and is not
        modified.
    shares
        The share of the budget given to each output. Shares are normalized to sum to one. By
        default, every output gets the same share. Passing the costs of the outputs as shares
        gives every output the same query rate instead.

    Examples
    --------

    >>> from fishladder import budget

    >>> costs = {"cheap": 1.0, "expensive": 5.0}
    >>> b = budget.PerOutput(budget.RateBudget(1.0))
    >>> for _ in range(10):
    ...     queried = b.select_one({"cheap": 1.0, "expensive": 1.0}, costs)
    ...     b.update({o: costs[o] for o in queried})
    >>> b.spent_per_output
    {'cheap': 5.0, 'expensive': 5.0}

    """

    def __init__(self, budget: BudgetManager, shares: Mapping[Output, float] | None = None):
        super().__init__()
        if shares is not None:
            for o, s in shares.items():
                if s <= 0:
                    raise ValueError(f"share of output {o!r} must be positive, got {s}")
        self.budget = budget
        self.shares = shares
        self._budgets: dict[Output, BudgetManager] = {}
        self._shares: dict[Output, float] = {}
        self.spent_per_output: dict[Output, float] = {}

    def _init_outputs(self, outputs: list[Output]) -> None:
        shares = self.shares if self.shares is not None else {o: 1.0 for o in outputs}
        missing = [o for o in outputs if o not in shares]
        if missing:
            raise KeyError(f"no share given for outputs {missing}")
        total = sum(shares[o] for o in outputs)
        for o in outputs:
            self._shares[o] = shares[o] / total
            self._budgets[o] = copy.deepcopy(self.budget)
            self.spent_per_output[o] = 0.0

    # Each output's copy of the budget is charged cost / share. Since budgets are linear in what
    # they make available, this is the same as giving the output `share` of the budget.

    @property
    def available(self) -> float:
        if not self._budgets:
            return self.budget.available
        return sum(self._budgets[o].available * s for o, s in self._shares.items())

    def select_one(
        self, utilities: Mapping[Output, float], costs: Mapping[Output, float]
    ) -> set[Output]:
        if not self._budgets:
            self._init_outputs(list(costs))
        return {
            o
            for o, u in utilities.items()
            if u > 0 and costs[o] / self._shares[o] <= self._budgets[o].available
        }

    def update(self, queried: Mapping[Output, float]) -> None:
        super().update(queried)
        for o, b in self._budgets.items():
            if o in queried:
                self.spent_per_output[o] += queried[o]
                b.update({o: queried[o] / self._shares[o]})
            else:
                b.update({})
