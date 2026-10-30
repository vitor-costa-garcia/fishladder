"""Metrics computed from an experiment log.

Metrics are River-style objects with `update(y_true, y_pred)` and `get()` methods, such as
`river.metrics.MAE()`. A fresh copy of the metric is used for each output.

"""

from __future__ import annotations

import copy
import math
import typing

from fishladder.base import Output
from fishladder.log import Log

if typing.TYPE_CHECKING:
    import pandas as pd

__all__ = ["curve", "per_output"]


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
