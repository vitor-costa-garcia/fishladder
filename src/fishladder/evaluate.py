"""Prequential evaluation of active learning on multi-output streams."""

from __future__ import annotations

import math
import typing
from collections.abc import Callable, Mapping

from fishladder.active import Wrapper
from fishladder.base import BudgetManager, Dataset, Model, Output, QueryStrategy
from fishladder.log import Log
from fishladder.oracle import Oracle

__all__ = ["iter_prequential", "prequential"]

Delay = int | float | Mapping[Output, float] | Callable[[dict, dict, Output], float] | None
"""How many instances after being queried a label arrives: see `iter_prequential`."""


def _check_delay(d: typing.Any, o: Output) -> float:
    if d != math.inf and not (isinstance(d, (int, float)) and d >= 0 and d == int(d)):
        raise ValueError(
            f"delay of output {o!r} must be a non-negative whole number or inf, got {d!r}"
        )
    return d if d == math.inf else int(d)


def _delay_function(
    delay: Delay, costs: Mapping[Output, float]
) -> Callable[[dict, dict, Output], float]:
    """Turn any accepted form of delay into a function of `(x, y, output)`, checking its values."""
    if delay is None:
        delay = 0
    if callable(delay):
        f = delay
        return lambda x, y, o: _check_delay(f(x, y, o), o)
    if isinstance(delay, Mapping):
        unknown = [o for o in delay if o not in costs]
        if unknown:
            raise KeyError(f"delay is given for {unknown}, which have no cost")
        missing = [o for o in costs if o not in delay]
        if missing:
            raise KeyError(f"delay is missing for {missing}")
        delays = {o: _check_delay(delay[o], o) for o in costs}
    else:
        delays = {o: _check_delay(delay, o) for o in costs}
    return lambda x, y, o: delays[o]


def iter_prequential(
    dataset: Dataset,
    model: Model,
    strategy: QueryStrategy,
    budget: BudgetManager,
    costs: Mapping[Output, float],
    metric: typing.Any = None,
    step: int = 1,
    log: Log | None = None,
    delay: Delay = None,
) -> typing.Iterator[dict[str, typing.Any]]:
    """Evaluate an active learning setup on a stream, yielding results as it goes.

    For each instance, the model predicts every output, the strategy and the budget decide which
    outputs to query, and the model learns from the revealed outputs only. Evaluation is
    test-then-train: predictions are compared to all the true outputs, queried or not.

    Labels can arrive late, with `delay`. A label with a delay of `d` is learned after the
    prediction of the `d`-th instance that follows, so `d` more instances are predicted without
    it, and a delay of 0 means no delay. Queries are paid for when they are made, and the
    strategy learns from the prediction the query was made with, not from a later one. Labels
    that have not arrived when the stream ends are paid for but never learned.

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
    delay
        The number of instances after which queried labels arrive. Either a single value, one per
        output, or a function of the features, the true outputs and the queried output, called
        once per query, which can draw random delays. A delay of `math.inf` means that the label
        never arrives, as when a query fails. Defaults to no delay.

    Yields
    ------
    A dictionary with the number of instances seen (`Step`), the number of outputs queried
    (`Queries`), the cost spent (`Cost`) and, if given, the metric under its class name.

    """
    if step < 1:
        raise ValueError(f"step must be at least 1, got {step}")

    learner = Wrapper(model, strategy, budget, costs)
    oracle = Oracle(costs)
    delay_of = _delay_function(delay, oracle.costs)
    # The labels waiting to be learned, by the step after which they arrive, then by the step of
    # their instance, with the instance and the prediction its query was made with
    pending: dict[int, dict[int, tuple[dict, dict, dict]]] = {}
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
        # Draw the delays in the order of the costs, so that random delays are reproducible
        delays = {o: delay_of(x, y, o) for o in oracle.costs if o in ask}
        for o, value in y_revealed.items():
            if delays[o] != math.inf:
                arrival = pending.setdefault(n + int(delays[o]), {})
                arrival.setdefault(n, (x, {}, y_pred))[1][o] = value
        for x_old, y_old, y_pred_old in pending.pop(n, {}).values():
            learner.learn_one(x_old, y_old, y_pred=y_pred_old)

        if log is not None:
            log.append_one(n, y, y_pred, utilities, ask, oracle.costs, budget.spent, delays)
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
    delay: Delay = None,
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
    delay
        The number of instances after which queried labels arrive. Either a single value, one per
        output, or a function of the features, the true outputs and the queried output.

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

    Labels can also arrive late. Here, the labels of `b` only arrive after 20 more instances,
    and each of them has a one in ten chance of never arriving:

    >>> import math
    >>> import random

    >>> rng = random.Random(42)
    >>> def delay(x, y, output):
    ...     if output == "a":
    ...         return 0
    ...     return math.inf if rng.random() < 0.1 else 20

    >>> log = fl.evaluate.prequential(
    ...     dataset=stream(200),
    ...     model=multioutput.RegressorChain(
    ...         preprocessing.StandardScaler() | linear_model.LinearRegression(),
    ...         order=["a", "b"],
    ...     ),
    ...     strategy=fl.strategy.RandomPerOutput(p=0.5, seed=42),
    ...     budget=fl.budget.RateBudget(0.5),
    ...     costs={"a": 1.0, "b": 2.0},
    ...     delay=delay,
    ... )

    >>> df = log.to_frame()
    >>> df[df["queried"]].groupby("output")["delay"].value_counts().to_dict()
    {('a', 0.0): 75, ('b', 20.0): 9, ('b', inf): 3}

    """
    log = Log()
    for _ in iter_prequential(
        dataset, model, strategy, budget, costs, metric=metric, log=log, delay=delay
    ):
        pass
    return log
