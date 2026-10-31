"""River-style active learning wrapper."""

from __future__ import annotations

import collections
import os
import sys
import types
import typing
import warnings
from collections.abc import Mapping

from fishladder.base import BudgetManager, Model, Output, QueryStrategy, _learn_unsupervised

__all__ = ["Wrapper"]

# River's setting for learning during predict. Pipelines read it through `self`, so setting it on
# a pipeline only changes that pipeline.
_FLAG = "_LEARN_UNSUPERVISED_DURING_PREDICT"

_PACKAGE = os.path.dirname(os.path.abspath(__file__)) + os.sep


class _Setting:
    """The value a wrapper gives to the River setting of its pipelines.

    It is true while the wrapper learns from every instance, which is all that River checks, and
    false once the wrapper stops. Copies of a pipeline, such as those a chain makes when it first
    learns, share it, so they all stop with the wrapper, and the wrapper can tell its own setting
    apart from any other. It also keeps what each pipeline had before, to put it back.

    """

    def __init__(self) -> None:
        self.on = True
        # Whether each pipeline had a setting of its own before, by id, and its value
        self.previous: dict[int, tuple[bool, typing.Any]] = {}

    def __bool__(self) -> bool:
        return self.on

    def give(self, pipeline: typing.Any) -> None:
        own = vars(pipeline)
        self.previous[id(pipeline)] = (_FLAG in own, own.get(_FLAG))
        setattr(pipeline, _FLAG, self)

    def take_back(self, pipeline: typing.Any) -> None:
        """Put back what `pipeline` had before, skipping settings of wrappers that stopped."""
        if vars(pipeline).get(_FLAG) is not self:
            return
        value: typing.Any = self
        while isinstance(value, _Setting) and not value.on:
            had, value = value.previous.get(id(pipeline), (False, None))
            if not had:
                delattr(pipeline, _FLAG)
                return
        setattr(pipeline, _FLAG, value)

    def __copy__(self) -> _Setting:
        return self

    def __deepcopy__(self, memo: dict) -> _Setting:
        return self

    def __reduce__(self) -> tuple[type, tuple[bool]]:
        # A saved model keeps the setting as River's own
        return bool, (self.on,)


def _outside_stacklevel() -> int:
    """The `stacklevel` that points `warnings.warn`, called by the caller of this function, at the
    first frame outside fishladder."""
    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None and os.path.abspath(frame.f_code.co_filename).startswith(
        _PACKAGE
    ):
        frame = frame.f_back
        level += 1
    return level


def _children(obj: typing.Any) -> list[typing.Any]:
    """The objects that `obj` holds."""
    children: list[typing.Any] = []
    if isinstance(obj, Mapping):
        children.extend(obj.values())
    elif isinstance(obj, (list, tuple, set, frozenset, collections.deque, collections.UserList)):
        children.extend(obj)
    if hasattr(obj, "__dict__"):
        children.extend(vars(obj).values())
    for cls in type(obj).__mro__:
        slots = vars(cls).get("__slots__", ())
        for name in [slots] if isinstance(slots, str) else slots:
            if hasattr(obj, name):
                children.append(getattr(obj, name))
    return children


def _find(obj: typing.Any, classes: tuple[type, ...]) -> dict[type, list[typing.Any]]:
    """The instances of each of `classes` that `obj` holds, including itself."""
    found: dict[type, list[typing.Any]] = {cls: [] for cls in classes}
    seen: set[int] = set()
    stack = [obj]
    while stack:
        obj = stack.pop()
        if id(obj) in seen or isinstance(obj, (str, bytes, int, float, type, types.ModuleType)):
            continue
        seen.add(id(obj))
        try:
            matches = [cls for cls in classes if isinstance(obj, cls)]
            children = _children(obj)
        except Exception:  # noqa: BLE001
            # Only River models are looked for, so objects that cannot be read are skipped
            matches, children = [], []
        for cls in matches:
            found[cls].append(obj)
        stack.extend(children)
    return found


def _river_classes() -> tuple[type, ...] | None:
    """The River classes that are looked for in models, or `None` if River is not installed."""
    try:
        from river.compose import Pipeline
        from river.model_selection.base import ModelSelector
        from river.multioutput.chain import BaseChain, ProbabilisticClassifierChain
    except ImportError:
        return None
    return Pipeline, ModelSelector, BaseChain, ProbabilisticClassifierChain


def _warn_unsupported(
    found: dict[type, list[typing.Any]], classes: tuple[type, ...], shared: bool
) -> None:
    """Warn about the River models found that predict an instance other than once, and about
    pipelines `shared` with the strategy."""
    _, ModelSelector, BaseChain, ProbabilisticClassifierChain = classes
    reasons = []
    if shared:
        reasons.append(
            "the strategy holds pipelines of the model, so its copies of them also learn their "
            "unsupervised steps when predicting. Give the strategy pipelines of its own"
        )
    if found[ProbabilisticClassifierChain]:
        reasons.append(
            "a River probabilistic classifier chain predicts each instance many times, also when "
            "it learns, so its unsupervised steps learn from it many times"
        )
    if any(not isinstance(obj, ProbabilisticClassifierChain) for obj in found[BaseChain]):
        reasons.append(
            "a River chain also predicts when it learns, so its unsupervised steps learn twice "
            "from the instances it learns from"
        )
    if found[ModelSelector]:
        reasons.append(
            "a River model selector predicts with its best model only, so the unsupervised steps "
            "of the best model learn twice from the instances the selector learns from, and those "
            "of the other models only learn from these instances"
        )
    if reasons:
        warnings.warn(
            f"with learn_unsupervised, {', and '.join(reasons)}. Leave learn_unsupervised off "
            "for this model, or see the documentation of fishladder.active.Wrapper.",
            stacklevel=_outside_stacklevel(),
        )


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
    learn_unsupervised
        Whether to learn from the features of every instance, queried or not. By default,
        unsupervised steps, such as a scaler, only learn from the instances whose labels are
        passed to `learn_one`.

        The River pipelines in the model then update their unsupervised steps when predicting
        instead of when learning, as with `river.compose.learn_during_predict`, but only for this
        model: the setting is put on each pipeline of the model, and nothing else is changed.
        Instances passed to `learn_one` without being predicted, as when warm-starting the model,
        are predicted first. The model and the strategy also have their `learn_unsupervised(x)`
        method called, if they have one, before each prediction. The River models inside
        strategies, as in `strategy.QueryByCommittee`, are not changed, unless the strategy holds
        pipelines of the model, and a warning is raised then.

        Each instance is learned from once if each pipeline predicts it once, as with
        `river.multioutput.PerOutputRegressor`. River chains also predict when they learn, so
        their unsupervised steps learn twice from the instances they learn from. River model
        selectors and probabilistic classifier chains do not predict once either, and a warning is
        raised for all of these. Models that make their pipelines when they first learn, as River
        multi-output models do, miss the instances predicted before. Models that make new
        pipelines with `clone` do not keep the setting.

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
        learn_unsupervised: bool = False,
    ):
        self.model = model
        self.strategy = strategy
        self.budget = budget
        self.costs = dict(costs)
        self._last: tuple[dict, dict] | None = None
        self._learn_unsupervised = False
        self._setting = _Setting()
        # The pipelines given the setting
        self._flagged: list[typing.Any] = []
        self.learn_unsupervised = learn_unsupervised

    @property
    def learn_unsupervised(self) -> bool:
        """Whether to learn from the features of every instance. Turning it on puts the setting
        on the pipelines that the model has at that time."""
        return self._learn_unsupervised

    @learn_unsupervised.setter
    def learn_unsupervised(self, value: bool) -> None:
        value = bool(value)
        classes = _river_classes()
        if classes is not None and value and not self._learn_unsupervised:
            found = _find(self.model, classes)
            pipelines = found[classes[0]]
            in_strategy = {id(p) for p in _find(self.strategy, classes[:1])[classes[0]]}
            shared = any(id(pipeline) in in_strategy for pipeline in pipelines)
            # Before changing anything, in case warnings are raised as errors
            _warn_unsupported(found, classes, shared)
            self._setting = _Setting()
            for pipeline in pipelines:
                self._setting.give(pipeline)
            self._flagged = pipelines
        elif classes is not None and not value and self._learn_unsupervised:
            self._unflag(classes[0])
        self._learn_unsupervised = value

    def _unflag(self, Pipeline: type) -> None:
        # This turns the setting off wherever it was copied, so the rest only tidies up
        self._setting.on = False
        # With copies of the pipelines, such as those a chain or a strategy made since
        for pipeline in [
            *self._flagged,
            *_find(self.model, (Pipeline,))[Pipeline],
            *_find(self.strategy, (Pipeline,))[Pipeline],
        ]:
            self._setting.take_back(pipeline)
        self._flagged = []

    def query_one(
        self, x: dict
    ) -> tuple[dict, dict[Output, float], dict[Output, float], set[Output]]:
        """Predict and decide what to ask for, also returning what the decision was based on.

        This charges the budget, exactly like `predict_one`.

        Returns
        -------
        The predictions, the utility of each output, the probabilities given by randomized
        strategies (see `QueryStrategy.sample_one`), and the outputs to ask for.

        """
        y_pred = self._predict(x)
        utilities, probabilities = self.strategy.sample_one(x, y_pred, self.costs)
        ask = self.budget.select_one(utilities, self.costs)
        self.budget.update({o: self.costs[o] for o in ask})
        return y_pred, utilities, probabilities, ask

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
        y_pred, _, _, ask = self.query_one(x)
        return y_pred, ask

    def learn_one(self, x: dict, y: dict, y_pred: dict | None = None) -> None:
        """Learn from the outputs that were revealed for `x`.

        Parameters
        ----------
        x
            A dictionary of features.
        y
            The revealed outputs. It may contain only some of the outputs.
        y_pred
            The prediction returned by `predict_one` when `x` was queried. Pass it when labels
            arrive late, after the model learned from other instances: the strategy then learns
            from the prediction it made the query with. By default, the prediction is reused if
            `x` is the last instance passed to `predict_one`, which must not be changed in the
            meantime, and `x` is predicted otherwise. With `learn_unsupervised`, late labels must
            then come with `y_pred`, for unsupervised steps to learn from `x` only once.

        """
        if not y:
            return
        if y_pred is None:
            if self._last is not None and self._last[0] is x:
                y_pred = self._last[1]
            else:
                y_pred = self._predict(x)
        self.model.learn_one(x, y)
        self.strategy.learn_one(x, y, y_pred)

    def _predict(self, x: dict) -> dict:
        if self.learn_unsupervised:
            _learn_unsupervised(self.model, x)
            _learn_unsupervised(self.strategy, x)
        y_pred = self.model.predict_one(x)
        self._last = (x, y_pred)
        return y_pred
