"""Prequential evaluation of active learning on multi-output streams."""

from __future__ import annotations

import typing
from collections.abc import Mapping

from fishladder.active import Wrapper
from fishladder.base import BudgetManager, Dataset, Model, Output, QueryStrategy
from fishladder.log import Log
from fishladder.oracle import Oracle

__all__ = ["iter_prequential", "prequential"]


def iter_prequential(
    dataset: Dataset,
    model: Model,
    strategy: QueryStrategy,
    budget: BudgetManager,
    costs: Mapping[Output, float],
    metric: typing.Any = None,
    step: int = 1,
    log: Log | None = None,
) -> typing.Iterator[dict[str, typing.Any]]:
    """Evaluate an active learning setup on a stream, yielding results as it goes.

    For each instance, the model predicts every output, the strategy and the budget decide which
    outputs to query, and the model learns from the revealed outputs only. Evaluation is
    test-then-train: predictions are compared to all the true outputs, queried or not.

    Parameters
    ----------
    dataset
        An iterable of `(x, y)` pairs, such as a River dataset.
    model
        A multi-output model, such as any River multi-output model.
    strategy
        The query strategy.
    budget
        The budget manager.
    costs
        The cost of querying each output.
    metric
        An optional River-style multi-output metric, updated with `metric.update(y, y_pred)`.
        It is only updated once the model can predict every output of an instance.
    step
        Yield results every `step` instances. The final results are always yielded.
    log
        An optional log to record the experiment into.

    Yields
    ------
    A dictionary with the number of instances seen (`Step`), the number of outputs queried
    (`Queries`), the cost spent (`Cost`) and, if given, the metric under its class name.

    """
    if step < 1:
        raise ValueError(f"step must be at least 1, got {step}")

    learner = Wrapper(model, strategy, budget, costs)
    oracle = Oracle(costs)
    n_queries = 0
    n = 0

    def report() -> dict[str, typing.Any]:
        state: dict[str, typing.Any] = {"Step": n, "Queries": n_queries, "Cost": budget.spent}
        if metric is not None:
            state[metric.__class__.__name__] = metric
        return state

    for x, y in dataset:
        y_pred, utilities, ask = learner.query_one(x)
        if metric is not None and all(o in y_pred for o in y):
            metric.update(y, y_pred)

        y_revealed, _ = oracle.reveal(y, ask)
        learner.learn_one(x, y_revealed)

        if log is not None:
            log.append_one(n, y, y_pred, utilities, ask, oracle.costs, budget.spent)
        n_queries += len(ask)
        n += 1
        if n % step == 0:
            yield report()

    if n % step != 0:
        yield report()


def prequential(
    dataset: Dataset,
    model: Model,
    strategy: QueryStrategy,
    budget: BudgetManager,
    costs: Mapping[Output, float],
    metric: typing.Any = None,
) -> Log:
    """Evaluate an active learning setup on a stream and return the experiment log.

    See `iter_prequential` for how the evaluation works.

    Parameters
    ----------
    dataset
        An iterable of `(x, y)` pairs, such as a River dataset.
    model
        A multi-output model, such as any River multi-output model.
    strategy
        The query strategy.
    budget
        The budget manager.
    costs
        The cost of querying each output.
    metric
        An optional River-style multi-output metric, updated in place.

    Returns
    -------
    The experiment log.

    Examples
    --------

    >>> from river import linear_model, metrics, multioutput, preprocessing
    >>> import fishladder as fl

    >>> def stream(n):
    ...     for i in range(n):
    ...         x = {"x": i % 10}
    ...         yield x, {"a": 2 * x["x"], "b": x["x"] - 1}

    >>> metric = metrics.multioutput.PerOutput(metrics.MAE())
    >>> log = fl.evaluate.prequential(
    ...     dataset=stream(200),
    ...     model=multioutput.RegressorChain(
    ...         preprocessing.StandardScaler() | linear_model.LinearRegression(),
    ...         order=["a", "b"],
    ...     ),
    ...     strategy=fl.strategy.RandomPerOutput(p=0.5, seed=42),
    ...     budget=fl.budget.RateBudget(0.5),
    ...     costs={"a": 1.0, "b": 2.0},
    ...     metric=metric,
    ... )

    >>> df = log.to_frame()
    >>> df.groupby("output")["queried"].sum().to_dict()
    {'a': 75, 'b': 12}

    """
    log = Log()
    for _ in iter_prequential(dataset, model, strategy, budget, costs, metric=metric, log=log):
        pass
    return log
