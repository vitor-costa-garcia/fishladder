"""Metrics computed from an experiment log.

Metrics are River-style objects with `update(y_true, y_pred)` and `get()` methods, such as
`river.metrics.MAE()` or `RRMSE()`. A fresh copy of the metric is used for each output.

"""

from __future__ import annotations

import copy
import itertools
import math
import typing

from fishladder.base import Output
from fishladder.log import Log

if typing.TYPE_CHECKING:
    import pandas as pd

__all__ = ["RRMSE", "auc", "average", "average_curve", "curve", "per_output"]


class RRMSE:
    """Relative root mean squared error.

    The root mean squared error of the model, divided by that of always predicting the mean of
    the true values. Values below 1 mean that the model does better than the mean. It is the
    usual metric of multi-target regression, where it is averaged over the outputs (aRRMSE, see
    `average`).

    On a stream, the mean can be taken in two ways, given by `baseline`:

    - `"cumulative"`: the mean of all the true values seen so far, including the current one.
      At the end of a stream, this is the mean of the whole stream, as in batch benchmarks. It
      equals `sqrt(1 - R2)`, with River's `metrics.R2`.
    - `"prequential"`: the mean of the true values seen before each one, as a model that
      predicts the running mean would. This is the relative root mean squared error of MOA and
      CapyMOA, which predict 0 for the first value.

    Parameters
    ----------
    baseline
        How the mean is taken, `"cumulative"` or `"prequential"`.

    Examples
    --------

    >>> from fishladder import metrics

    >>> y_true = [3.0, -0.5, 2.0, 7.0]
    >>> y_pred = [2.5, 0.0, 2.0, 8.0]

    >>> metric = metrics.RRMSE()
    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric.update(yt, yp)
    >>> metric
    RRMSE: 0.226698

    >>> metric = metrics.RRMSE(baseline="prequential")
    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric.update(yt, yp)
    >>> metric
    RRMSE: 0.169740

    """

    bigger_is_better = False

    def __init__(self, baseline: typing.Literal["cumulative", "prequential"] = "cumulative"):
        if baseline not in ("cumulative", "prequential"):
            raise ValueError(f"baseline must be 'cumulative' or 'prequential', got {baseline!r}")
        self.baseline = baseline
        self._weight = 0.0
        self._mean = 0.0
        self._squared_error = 0.0
        # The squared error of the mean: the sum of squared deviations from the mean for
        # "cumulative", and the squared errors of the running mean for "prequential"
        self._baseline_error = 0.0

    def update(self, y_true: float, y_pred: float, w: float = 1.0) -> None:
        """Update the metric with a true value and its prediction."""
        self._squared_error += w * (y_true - y_pred) ** 2
        delta = y_true - self._mean
        if self.baseline == "prequential":
            self._baseline_error += w * delta**2
        self._weight += w
        self._mean += w * delta / self._weight
        if self.baseline == "cumulative":
            self._baseline_error += w * delta * (y_true - self._mean)

    def get(self) -> float:
        """The current value of the metric, or NaN while the mean makes no error."""
        if self._baseline_error <= 0:
            return math.nan
        return math.sqrt(self._squared_error / self._baseline_error)

    def works_with(self, model: typing.Any) -> bool:
        return True

    def __repr__(self) -> str:
        return f"RRMSE: {self.get():,.6f}"


def _scored_rows(log: Log) -> typing.Iterator[dict[str, typing.Any]]:
    """Rows where both a true value and a prediction are available."""
    for row in log:
        if row["y_true"] is not None and row["y_pred"] is not None:
            yield row


def per_output(log: Log, metric: typing.Any) -> dict[Output, typing.Any]:
    """Compute a metric for each output over the whole stream.

    Parameters
    ----------
    log
        An experiment log.
    metric
        A River-style single-output metric, such as `river.metrics.MAE()`. It is not modified.

    Returns
    -------
    A copy of the metric for each output.

    """
    metrics: dict[Output, typing.Any] = {}
    for row in _scored_rows(log):
        o = row["output"]
        if o not in metrics:
            metrics[o] = copy.deepcopy(metric)
        metrics[o].update(row["y_true"], row["y_pred"])
    return metrics


def curve(log: Log, metric: typing.Any, every: int = 1) -> pd.DataFrame:
    """Compute the running value of a metric for each output, along the stream.

    This is what learning curves are drawn from, against either `step` or `cum_cost`.

    Parameters
    ----------
    log
        An experiment log.
    metric
        A River-style single-output metric, such as `river.metrics.MAE()`. It is not modified.
    every
        Record the value of the metric every `every` instances, and after the last one.

    Returns
    -------
    A DataFrame with columns `step`, `cum_cost`, `output` and `value`. The value is NaN until
    the output has been predicted at least once, which can take a while when its labels are
    delayed.

    """
    import pandas as pd

    if every < 1:
        raise ValueError(f"every must be at least 1, got {every}")

    metrics: dict[Output, typing.Any] = {}
    scored: set[Output] = set()
    records = []
    last: dict[str, typing.Any] | None = None

    def value(o: Output) -> float:
        # River metrics return 0 before any update, which would look like a perfect score
        return metrics[o].get() if o in scored else math.nan

    for row in log:
        o = row["output"]
        if o not in metrics:
            metrics[o] = copy.deepcopy(metric)
        if row["y_true"] is not None and row["y_pred"] is not None:
            metrics[o].update(row["y_true"], row["y_pred"])
            scored.add(o)
        if (row["step"] + 1) % every == 0:
            records.append((row["step"], row["cum_cost"], o, value(o)))
        last = row
    if last is not None and (last["step"] + 1) % every != 0:
        for o in metrics:
            records.append((last["step"], last["cum_cost"], o, value(o)))
    return pd.DataFrame(records, columns=["step", "cum_cost", "output", "value"])


def average(log: Log, metric: typing.Any) -> float:
    """Average a metric over the outputs, each computed over the whole stream.

    With `RRMSE`, this is the average relative root mean squared error (aRRMSE) of multi-target
    regression.

    Parameters
    ----------
    log
        An experiment log.
    metric
        A River-style single-output metric. It is not modified.

    Returns
    -------
    The average of the metric over the outputs that were predicted at least once, or NaN if
    none were. It is NaN if the metric is NaN for any of them.

    """
    values = [m.get() for m in per_output(log, metric).values()]
    return sum(values) / len(values) if values else math.nan


def average_curve(log: Log, metric: typing.Any, every: int = 1) -> pd.DataFrame:
    """Compute the running average of a metric over the outputs, along the stream.

    With `RRMSE`, this gives learning curves of the aRRMSE.

    Parameters
    ----------
    log
        An experiment log.
    metric
        A River-style single-output metric, such as `RRMSE()`. It is not modified.
    every
        Record the value of the metric every `every` instances, and after the last one.

    Returns
    -------
    A DataFrame with columns `step`, `cum_cost` and `value`. The value is NaN until every output
    has been predicted at least once.

    """
    per_step = curve(log, metric, every).groupby(["step", "cum_cost"], sort=False)["value"]
    # mean() would skip NaN, and average over the outputs that happen to be scored
    return per_step.agg(lambda v: v.mean(skipna=False)).reset_index()


def _area(xs: list[float], ys: list[float]) -> float:
    """The area under a curve, by the trapezoidal rule, divided by the range of x."""
    points = [(x, y) for x, y in zip(xs, ys) if not math.isnan(y)]
    if len(points) < 2 or points[-1][0] == points[0][0]:
        return math.nan
    area = sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in itertools.pairwise(points))
    return area / (points[-1][0] - points[0][0])


def auc(
    log: Log, metric: typing.Any, every: int = 1, average: bool = False
) -> dict[Output, float] | float:
    """Compute the area under the learning curve of a metric, against the cost spent.

    The area is computed by the trapezoidal rule, and divided by the range of the cost, from
    the first instance where the metric is defined to the last. It is the average value of the
    metric over the run, weighted by the cost spent, so it can be compared between runs with
    different budgets, and has the same units as the metric.

    Parameters
    ----------
    log
        An experiment log.
    metric
        A River-style single-output metric, such as `RRMSE()`. It is not modified.
    every
        Record the value of the metric every `every` instances, and after the last one. Larger
        values are faster, and less precise.
    average
        Whether to compute the area under the curve of the average of the metric over the
        outputs (see `average_curve`), rather than one area per output.

    Returns
    -------
    The area for each output, or the area of the average if `average` is true. Areas are NaN
    when no cost was spent while the metric was defined.

    """
    if average:
        c = average_curve(log, metric, every)
        return _area(c["cum_cost"].tolist(), c["value"].tolist())
    return {
        o: _area(c["cum_cost"].tolist(), c["value"].tolist())
        for o, c in curve(log, metric, every).groupby("output", sort=False)
    }
