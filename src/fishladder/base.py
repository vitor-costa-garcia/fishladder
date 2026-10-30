"""Base classes and protocols shared across fishladder.

fishladder follows River's conventions: features are a dictionary `x`, the outputs of an
instance are a dictionary `y` mapping each output name to its value, and everything is processed
one instance at a time through `*_one` methods.

"""

from __future__ import annotations

import abc
import typing
from collections.abc import Hashable, Iterable, Mapping

__all__ = [
    "BudgetManager",
    "Dataset",
    "Model",
    "Output",
    "QueryStrategy",
]

Output = Hashable
"""The name of an output, i.e. a key of `y`."""

Dataset = Iterable[tuple[dict[typing.Any, typing.Any], dict[typing.Any, typing.Any]]]
"""An iterable of `(x, y)` pairs, where `y` maps each output to its value."""


@typing.runtime_checkable
class Model(typing.Protocol):
    """An incremental multi-output model.

    Any River multi-output model (for instance `river.multioutput.RegressorChain`) satisfies this
    protocol. `learn_one` must accept a `y` that only contains some of the outputs, which is what
    happens when only a subset of the outputs of an instance is queried.

    """

    def predict_one(self, x: dict) -> dict: ...

    def learn_one(self, x: dict, y: dict) -> typing.Any: ...


class QueryStrategy(abc.ABC):
    """Base class for query strategies.

    A query strategy scores how useful it would be to query each output of an instance. It does
    not decide what is affordable: that is the job of a `BudgetManager`. Outputs with a utility
    lower than or equal to zero are never queried.

    """

    @abc.abstractmethod
    def utilities_one(
        self, x: dict, y_pred: dict, costs: Mapping[Output, float]
    ) -> dict[Output, float]:
        """Score each output of an instance.

        Parameters
        ----------
        x
            A dictionary of features.
        y_pred
            The model's predictions for `x`. Outputs the model has not seen yet may be missing.
        costs
            The cost of querying each output. Its keys are the outputs to score.

        Returns
        -------
        The utility of querying each output.

        """

    def learn_one(self, x: dict, y: dict, y_pred: dict) -> None:
        """Update the strategy with the outputs that were revealed for `x`.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            The revealed outputs.
        y_pred
            The model's predictions for `x`, made before it learned from `y`.

        """


class BudgetManager(abc.ABC):
    """Base class for budget managers.

    A budget manager turns the utilities given by a query strategy into the set of outputs to
    query, without spending more than what is available. Budgets are measured in cost units, not
    in number of queries.

    By default, outputs are picked greedily by decreasing utility, skipping any output that does
    not fit in what is left of the budget. Ties go to the output that has had the least spent on
    it, so that outputs with the same utility, such as with `strategy.RandomPerOutput`, take
    turns. Since all outputs share the same budget, cheap outputs can still keep an expensive one
    from ever being queried; `budget.PerOutput` avoids this by giving each output its own budget.

    """

    def __init__(self) -> None:
        self.spent = 0.0
        self.spent_per_output: dict[Output, float] = {}
        self.n_seen = 0

    @property
    @abc.abstractmethod
    def available(self) -> float:
        """The amount of cost that can be spent on the current instance."""

    def select_one(
        self, utilities: Mapping[Output, float], costs: Mapping[Output, float]
    ) -> set[Output]:
        """Select which outputs to query. This does not change the state of the budget.

        Parameters
        ----------
        utilities
            The utility of querying each output, as given by a query strategy.
        costs
            The cost of querying each output.

        Returns
        -------
        The outputs to query.

        """
        left = self.available
        selected = set()
        candidates = [o for o, u in utilities.items() if u > 0]
        ranked = sorted(
            candidates, key=lambda o: (-utilities[o], self.spent_per_output.get(o, 0.0))
        )
        for o in ranked:
            if costs[o] <= left:
                selected.add(o)
                left -= costs[o]
        return selected

    def update(self, queried: Mapping[Output, float]) -> None:
        """Record what was spent on an instance. Must be called once per instance, even when
        nothing was queried.

        Parameters
        ----------
        queried
            The cost paid for each output queried for the instance.

        """
        self.spent += sum(queried.values())
        for o, c in queried.items():
            self.spent_per_output[o] = self.spent_per_output.get(o, 0.0) + c
        self.n_seen += 1
