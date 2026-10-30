"""River-style active learning wrapper."""

from __future__ import annotations

from collections.abc import Mapping

from fishladder.base import BudgetManager, Model, Output, QueryStrategy

__all__ = ["Wrapper"]


class Wrapper:
    """Wrap a multi-output model so that it decides which outputs to ask for.

    This mirrors `river.active`: `predict_one` returns the prediction along with what to ask for,
    which makes it easy to drop into an existing River loop. The difference is that it returns a
    set of outputs instead of a boolean, and that queries are paid for with a budget.

    Each call to `predict_one` is a query decision: the cost of the outputs it asks for is
    charged to the budget right away, whether or not their labels are passed to `learn_one`.

    Parameters
    ----------
    model
        A multi-output model, such as any River multi-output model.
    strategy
        The query strategy.
    budget
        The budget manager.
    costs
        The cost of querying each output.

    Examples
    --------

    >>> from river import linear_model, multioutput
    >>> import fishladder as fl

    >>> model = fl.active.Wrapper(
    ...     model=multioutput.RegressorChain(linear_model.LinearRegression(), order=["a", "b"]),
    ...     strategy=fl.strategy.QueryAll(),
    ...     budget=fl.budget.TotalBudget(4.0),
    ...     costs={"a": 1.0, "b": 3.0},
    ... )

    >>> stream = [({"x": 1.0}, {"a": 1.0, "b": 2.0}), ({"x": 2.0}, {"a": 2.0, "b": 4.0})]
    >>> for x, y in stream:
    ...     y_pred, ask = model.predict_one(x)
    ...     if ask:
    ...         model.learn_one(x, {o: y[o] for o in ask})

    >>> model.budget.spent
    4.0

    """

    def __init__(
        self,
        model: Model,
        strategy: QueryStrategy,
        budget: BudgetManager,
        costs: Mapping[Output, float],
    ):
        self.model = model
        self.strategy = strategy
        self.budget = budget
        self.costs = dict(costs)

    def query_one(self, x: dict) -> tuple[dict, dict[Output, float], set[Output]]:
        """Predict and decide what to ask for, also returning the utilities of each output.

        This charges the budget, exactly like `predict_one`.

        Returns
        -------
        The predictions, the utility of each output, and the outputs to ask for.

        """
        y_pred = self.model.predict_one(x)
        utilities = self.strategy.utilities_one(x, y_pred, self.costs)
        ask = self.budget.select_one(utilities, self.costs)
        self.budget.update({o: self.costs[o] for o in ask})
        return y_pred, utilities, ask

    def predict_one(self, x: dict) -> tuple[dict, set[Output]]:
        """Predict the outputs of `x` and decide which of them to ask for.

        Parameters
        ----------
        x
            A dictionary of features.

        Returns
        -------
        The predictions, and the outputs whose labels should be asked for.

        """
        y_pred, _, ask = self.query_one(x)
        return y_pred, ask

    def learn_one(self, x: dict, y: dict) -> None:
        """Learn from the outputs that were revealed for `x`.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            The revealed outputs. It may contain only some of the outputs.

        """
        if not y:
            return
        self.model.learn_one(x, y)
        self.strategy.learn_one(x, y)
