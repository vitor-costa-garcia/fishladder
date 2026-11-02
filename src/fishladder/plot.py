"""Plots of experiment logs.

Plots are drawn with matplotlib, which comes with the `plots` extra:
`pip install "fishladder[plots]"`. Their values come from `fishladder.metrics`, so a plot always
shows the same numbers as the metrics.

A run is a `Log`, or a list of logs of the same setup, for example with different seeds. For a
list, plots draw the mean over the logs and shade the range between the lowest and highest
values. This range shows the spread of the logs. It is not a confidence interval: with few
logs, a difference between two ranges that do not overlap can still be due to chance. To
compare runs, pass a dictionary of runs with their names as keys.

Each function returns the axes it drew on, so the plot can be changed afterwards, for example
with `ax.set(yscale="log")`.

"""

from __future__ import annotations

import math
import typing
import warnings
from collections.abc import Hashable, Iterable, Mapping, Sequence

import numpy as np

from fishladder import metrics
from fishladder.base import Output, _select_outputs
from fishladder.log import Log

if typing.TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

__all__ = ["Run", "budget", "curve", "queries"]

Run = Log | Sequence[Log]
"""A log, or the logs of several repeats of the same setup, such as different seeds."""

_X_COLUMNS = {"step": "step", "cost": "cum_cost"}
_X_LABELS = {"step": "instances", "cost": "cost spent"}


def _pyplot() -> typing.Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            'fishladder.plot needs matplotlib, install it with: pip install "fishladder[plots]"'
        ) from e
    return plt


def _repeats(run: Run) -> list[Log]:
    if not isinstance(run, Log | Sequence):
        raise TypeError(f"a run is a Log or a list of logs, got {type(run).__name__}")
    logs = [run] if isinstance(run, Log) else list(run)
    if not logs:
        raise ValueError("a run needs at least one log")
    for log in logs:
        if not isinstance(log, Log):
            raise TypeError(f"a run is a Log or a list of logs, got {type(log).__name__}")
    return logs


def _runs(runs: Run | Mapping[Hashable, Run]) -> dict[Hashable | None, list[Log]]:
    """The logs of each run, by name. A run without a name gets the name `None`."""
    if isinstance(runs, Mapping):
        if not runs:
            raise ValueError("no runs to plot")
        return {name: _repeats(run) for name, run in runs.items()}
    return {None: _repeats(runs)}


def _name(metric: typing.Any) -> str:
    # The repr of River metrics, and of RRMSE, starts with the name of the metric
    return repr(metric).split(":")[0].strip()


def _check_x(x: str) -> str:
    if x not in _X_COLUMNS:
        raise ValueError(f"x must be 'step' or 'cost', got {x!r}")
    return _X_COLUMNS[x]


def _outputs(outputs: Iterable[Output] | None, found: Iterable[Output]) -> list[Output]:
    """The outputs to draw: all those found, in the order they first appear, or those given."""
    seen = list(dict.fromkeys(found))
    selected = _select_outputs(seen, outputs)
    return seen if selected is None else selected


def _axes(ax: Axes | Sequence[Axes] | None, n: int, sharey: bool = False) -> list[Axes]:
    """The given axes, or new ones side by side in a new figure."""
    if n < 1:
        raise ValueError("no outputs to plot")
    if ax is None:
        _, axes = _pyplot().subplots(
            1,
            n,
            figsize=(max(6.0, 4.5 * n), 4.0),
            squeeze=False,
            sharey=sharey,
            layout="constrained",
        )
        return list(axes[0])
    from matplotlib.axes import Axes

    # Flattened, so that a grid of axes from `plt.subplots(rows, columns)` can be given
    axes = [ax] if isinstance(ax, Axes) else list(np.ravel(np.asarray(ax, dtype=object)))
    if len(axes) != n:
        raise ValueError(f"need {n} axes, got {len(axes)}")
    return axes


def _summarize(
    series: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The mean, lowest and highest of several curves, as `(x, mean, low, high)`.

    The x values of each curve must not decrease. A single curve is returned as it is.
    Otherwise, NaN values are dropped, the last value is kept where x repeats, and the curves
    are linearly interpolated to all of their x values, in the range that every curve covers.
    Curves without any value are left out.

    """
    if len(series) == 1:
        x, y = series[0]
        return x, y, y, y
    cleaned = []
    for x, y in series:
        defined = ~np.isnan(y)
        x, y = x[defined], y[defined]
        if len(x) > 0:
            last = np.append(x[1:] != x[:-1], True)
            cleaned.append((x[last], y[last]))
    if not cleaned:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty
    if len(cleaned) == 1:
        x, y = cleaned[0]
        return x, y, y, y
    start = max(x[0] for x, _ in cleaned)
    end = min(x[-1] for x, _ in cleaned)
    grid = np.unique(np.concatenate([x for x, _ in cleaned]))
    grid = grid[(grid >= start) & (grid <= end)]
    ys = np.array([np.interp(grid, x, y) for x, y in cleaned])
    return grid, ys.mean(axis=0), ys.min(axis=0), ys.max(axis=0)


def _line(
    ax: Axes,
    x: np.ndarray,
    mean: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    label: Hashable | None,
    band: bool,
    **kwargs: typing.Any,
) -> None:
    """A line at the mean, with the range between `low` and `high` shaded if `band` is true."""
    (line,) = ax.plot(x, mean, label=None if label is None else str(label), **kwargs)
    if band:
        ax.fill_between(x, low, high, color=line.get_color(), alpha=0.2, linewidth=0)


def _draw(
    ax: Axes,
    series: Sequence[tuple[np.ndarray, np.ndarray]],
    label: Hashable | None,
    band: bool,
    **kwargs: typing.Any,
) -> None:
    _line(ax, *_summarize(series), label, band and len(series) > 1, **kwargs)


def _average_title(outputs: Sequence[Output] | None) -> str:
    if outputs is None:
        return "average over the outputs"
    return "average over " + ", ".join(str(o) for o in outputs)


def _legend(ax: Axes, names: Iterable[Hashable | None]) -> None:
    if any(name is not None for name in names):
        ax.legend()


def _xy(frame: pd.DataFrame, column: str, value: str = "value") -> tuple[np.ndarray, np.ndarray]:
    return frame[column].to_numpy(dtype=float), frame[value].to_numpy(dtype=float)


def _check_baseline(named: Mapping[Hashable | None, typing.Any], baseline: Hashable | None) -> None:
    if baseline is None:
        return
    if None in named:
        raise ValueError("baseline needs a dictionary of runs by name")
    if baseline not in named:
        raise KeyError(f"baseline {baseline!r} is not one of the runs")


def _minus(
    series: Sequence[tuple[np.ndarray, np.ndarray]], ref: tuple[np.ndarray, np.ndarray]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The curves minus the reference curve, interpolated linearly to their x values.

    Values are NaN where x is outside the range of the reference.

    """
    ref_x, ref_y = ref
    result = []
    for x, y in series:
        if len(ref_x) == 0:
            result.append((x, np.full(len(x), np.nan)))
            continue
        inside = (x >= ref_x[0]) & (x <= ref_x[-1])
        result.append((x, np.where(inside, y - np.interp(x, ref_x, ref_y), np.nan)))
    return result


def _ylabel(metric: typing.Any, baseline: Hashable | None, suffix: str = "") -> str:
    label = _name(metric) + suffix
    return label if baseline is None else f"{label}, difference to {baseline}"


def _by_cost(cost: np.ndarray, value: np.ndarray, every: float) -> tuple[np.ndarray, np.ndarray]:
    """The values of a curve at each multiple of `every` of the cost spent, and at the end.

    At each multiple, the value is the one after the last instance where the cost spent was at
    most that much. It is NaN before the first instance. Costs are sums of floats, so they are
    compared to the multiples with a small tolerance.

    """
    if len(cost) == 0:
        return cost, value
    tolerance = every * 1e-9
    grid = every * np.arange(1, int((cost[-1] + tolerance) // every) + 1)
    if len(grid) == 0 or grid[-1] < cost[-1] - tolerance:
        grid = np.append(grid, cost[-1])
    i = np.searchsorted(cost, grid + tolerance, side="right") - 1
    return grid, np.where(i >= 0, value[np.maximum(i, 0)], np.nan)


def curve(
    runs: Run | Mapping[Hashable, Run],
    metric: typing.Any,
    x: typing.Literal["step", "cost"] = "step",
    every: float | None = None,
    average: bool = False,
    outputs: Iterable[Output] | None = None,
    band: bool = True,
    baseline: Hashable | None = None,
    ax: Axes | Sequence[Axes] | None = None,
) -> list[Axes] | Axes:
    """Plot prequential learning curves, one panel per output.

    The curves are the running value of the metric along the stream, from `metrics.curve`. Wrap
    the metric in River's `utils.Rolling` to show the value over the last instances, rather
    than over the whole past.

    Parameters
    ----------
    runs
        A run, or a dictionary of runs by name. A run is a log, or a list of logs of the same
        setup. With a list, the mean over the logs is drawn, and the curves are linearly
        interpolated to common x values first, in the range that all of them cover.
    metric
        A River-style single-output metric, such as `river.metrics.MAE()`. It is not modified.
    x
        What to draw the curves against, the number of instances (`"step"`) or the cost spent
        (`"cost"`).
    every
        With `x="step"`, draw a point every `every` instances, by default after every instance.
        With `x="cost"`, draw a point each time the cost spent reaches a multiple of `every`,
        with the value after the last instance that cost at most that much. All runs then have
        points at the same costs, even if they spend at different rates. By default, `every`
        gives the run that spent the least about 100 points, and the run that spent the most
        at most about 10,000. There is also a point after the last instance. The cost spent is
        counted from the start of each log, as in `metrics.curve`.
    average
        Whether to draw the average of the metric over the outputs, from
        `metrics.average_curve`, in a single panel.
    outputs
        The outputs to draw, in order, or to average over if `average` is true. Defaults to all
        of them. Logs without an output are left out of its panel. To average over outputs,
        every log needs them.
    band
        Whether to shade the range between the lowest and highest values of runs with several
        logs.
    baseline
        The name of a run in `runs` to compare the others to, such as `"random"`. Each run is
        drawn minus the mean of the baseline, at the same x, and its range is that of its own
        logs. The baseline is then drawn at 0, with the range of its own logs around it.
    ax
        The axes to draw on, one per output, or one if `average` is true. Defaults to new axes
        in a new figure.

    Returns
    -------
    The axes, in the order of the outputs, or a single one if `average` is true.

    """
    column = _check_x(x)
    if every is not None and not (math.isfinite(every) and every > 0):
        raise ValueError(f"every must be a positive number, got {every}")
    if x == "step" and every is not None and every != int(every):
        raise ValueError(f"every must be a whole number of instances, got {every}")
    named = _runs(runs)
    _check_baseline(named, baseline)
    logs_outputs = {id(lg): lg.outputs for logs in named.values() for lg in logs}
    found = [o for outs in logs_outputs.values() for o in outs]
    if x == "cost" and every is None:
        spent = [c for logs in named.values() for lg in logs if (c := lg.total_cost) > 0]
        every = max(min(spent) / 100, max(spent) / 10_000) if spent else 1.0
    # Against cost, the metric is computed after every instance, and then picked by cost
    steps = int(every) if x == "step" and every is not None else 1
    by_cost = every if x == "cost" else None

    def xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = _xy(frame, column)
        return (xs, ys) if by_cost is None else _by_cost(xs, ys, by_cost)

    def draw(
        panel: Axes, series: dict[Hashable | None, list[tuple[np.ndarray, np.ndarray]]]
    ) -> None:
        if baseline is not None:
            ref = _summarize(series[baseline])[:2]
            series = {name: _minus(s, ref) for name, s in series.items()}
        for name, s in series.items():
            _draw(panel, s, name, band)

    ylabel = _ylabel(metric, baseline)
    if average:
        selected = None if outputs is None else _outputs(outputs, found)
        (avg_ax,) = _axes(ax, 1)
        draw(
            avg_ax,
            {
                name: [xy(metrics.average_curve(log, metric, steps, selected)) for log in logs]
                for name, logs in named.items()
            },
        )
        avg_ax.set(title=_average_title(selected), xlabel=_X_LABELS[x], ylabel=ylabel)
        _legend(avg_ax, named)
        return avg_ax

    shown = _outputs(outputs, found)

    def log_curve(log: Log) -> pd.DataFrame | None:
        """The curves of the outputs shown that the log has, or None if it has none of them."""
        if outputs is None:
            return metrics.curve(log, metric, steps)
        present = [o for o in shown if o in logs_outputs[id(log)]]
        return metrics.curve(log, metric, steps, present) if present else None

    curves = {name: [log_curve(log) for log in logs] for name, logs in named.items()}
    axes = _axes(ax, len(shown))
    for panel, o in zip(axes, shown):
        draw(
            panel,
            {
                name: [xy(c[c["output"] == o]) for c in cs if c is not None]
                for name, cs in curves.items()
            },
        )
        panel.set(title=str(o), xlabel=_X_LABELS[x])
    axes[0].set_ylabel(ylabel)
    _legend(axes[0], named)
    return axes


def budget(
    runs: Sequence[Run] | Mapping[Hashable, Sequence[Run]],
    metric: typing.Any,
    budgets: Sequence[float] | Mapping[Hashable, Sequence[float]] | None = None,
    summary: typing.Literal["final", "auc"] = "final",
    every: int = 1,
    average: bool = False,
    outputs: Iterable[Output] | None = None,
    band: bool = True,
    baseline: Hashable | None = None,
    ax: Axes | Sequence[Axes] | None = None,
) -> list[Axes] | Axes:
    """Plot a metric against the budget or the cost spent, for setups run with different budgets.

    Each run gives one point. By default, it is drawn at the total cost it spent, read from the
    log, so runs with different kinds of budgets can be compared, and runs that spend less than
    their budget allows are drawn where they really are. With `budgets`, it is drawn at its
    budget instead.

    Parameters
    ----------
    runs
        A list of runs of the same setup, one per budget, or a dictionary of such lists by
        name. A run is a log, or a list of logs of the same setup with the same budget. With a
        list, the point is at the mean cost and the mean value over the logs. So
        `[log_1, log_2]` is two budgets with one log each, and `[[log_1, log_2]]` is one budget
        with two logs.
    metric
        A River-style single-output metric, such as `river.metrics.MAE()`. It is not modified.
    budgets
        The budget of each run, in the order of `runs`, to draw on the x axis instead of the
        cost spent. A list is used for every setup. A dictionary gives a list for each name in
        `runs`.
    summary
        How to sum up a run in one value: the value of the metric over the whole stream
        (`"final"`, see `metrics.per_output`), or the area under its curve against the cost
        (`"auc"`, see `metrics.auc`).
    every
        For `"auc"`, compute the metric every `every` instances, and after the last one.
    average
        Whether to draw the average of the metric over the outputs, in a single panel.
    outputs
        The outputs to draw, in order, or to average over if `average` is true. Defaults to all
        of them.
    band
        Whether to shade the range between the lowest and highest values of runs with several
        logs.
    baseline
        The name of a setup in `runs` to compare the others to, such as `"random"`. Each run is
        drawn minus the mean of the baseline, at the same x, interpolated linearly between the
        baseline's points, and its range is that of its own logs. The baseline is then drawn at
        0, with the range of its own logs around it. Points outside the range of x of the
        baseline cannot be compared and are not drawn, with a warning. Against the cost spent,
        this happens to setups that spend less or more than the baseline, so passing `budgets`
        is often better.
    ax
        The axes to draw on, one per output, or one if `average` is true. Defaults to new axes
        in a new figure.

    Returns
    -------
    The axes, in the order of the outputs, or a single one if `average` is true.

    """
    if summary not in ("final", "auc"):
        raise ValueError(f"summary must be 'final' or 'auc', got {summary!r}")
    by_name: Mapping[Hashable | None, typing.Any] = (
        runs if isinstance(runs, Mapping) else {None: runs}
    )
    if not by_name:
        raise ValueError("no runs to plot")
    for name, rs in by_name.items():
        if isinstance(rs, Log):
            where = "" if name is None else f" for {name!r}"
            raise TypeError(f"budget needs a list of runs{where}, one per budget, not a single log")
    named = {name: [_repeats(run) for run in rs] for name, rs in by_name.items()}
    for name, levels in named.items():
        if not levels:
            raise ValueError(f"run {name!r} has no budgets")
    _check_baseline(named, baseline)
    # In the order of the log, not in the order the outputs were first predicted
    found = [o for levels in named.values() for logs in levels for lg in logs for o in lg.outputs]
    shown = _outputs(outputs, found)
    # Only the outputs given are passed on, so that by default the values are those of `metrics`
    requested = None if outputs is None else shown

    # The budget of each run of each setup, if they are given
    levels_x: dict[Hashable | None, Sequence[float] | None] = {}
    for name, levels in named.items():
        if budgets is None:
            levels_x[name] = None
            continue
        if isinstance(budgets, Mapping) and name not in budgets:
            raise KeyError(f"no budgets for {name!r}")
        xs = budgets[name] if isinstance(budgets, Mapping) else budgets
        if len(xs) != len(levels):
            raise ValueError(f"{name!r} has {len(levels)} runs, but {len(xs)} budgets")
        levels_x[name] = xs

    def value(log: Log) -> typing.Any:
        """The value of the metric for each output, or its average if `average` is true."""
        if summary == "auc" and not average and requested is not None:
            # Areas of the outputs drawn that the log has. The others are drawn as missing.
            has = set(log.outputs)
            present = [o for o in requested if o in has]
            return metrics.auc(log, metric, every, outputs=present) if present else {}
        if summary == "auc":
            return metrics.auc(log, metric, every, average, requested)
        if average:
            return metrics.average(log, metric, requested)
        return {o: m.get() for o, m in metrics.per_output(log, metric).items()}

    def x(name: Hashable | None, i: int, log: Log) -> float:
        xs = levels_x[name]
        return log.total_cost if xs is None else xs[i]

    # The x value (budget or total cost) and the value of each log, by name, then by budget
    points = {
        name: [[(x(name, i, log), value(log)) for log in logs] for i, logs in enumerate(levels)]
        for name, levels in named.items()
    }

    def series(level: list[tuple[float, typing.Any]], o: Output | None) -> np.ndarray:
        """The x value and the value of each log of a budget, for output `o` or the average."""
        return np.array([(c, v if o is None else v.get(o, np.nan)) for c, v in level], dtype=float)

    def draw(panel: Axes, o: Output | None) -> None:
        ref: tuple[np.ndarray, np.ndarray] | None = None
        if baseline is not None:
            arrays = [series(level, o) for level in points[baseline]]
            ref_x = np.array([a[:, 0].mean() for a in arrays])
            ref_y = np.array([a[:, 1].mean() for a in arrays])
            # Budgets of the baseline that spent the same cost are merged, for `np.interp`
            unique_x, index = np.unique(ref_x, return_inverse=True)
            ref = (unique_x, np.bincount(index, ref_y) / np.bincount(index))
        for name, levels in points.items():
            arrays = [series(level, o) for level in levels]
            cost = np.array([a[:, 0].mean() for a in arrays])
            values = [a[:, 1] for a in arrays]
            if ref is not None:
                # The logs of a budget are drawn at their mean x, so compare them to the baseline there
                at_cost = [(np.full(len(v), c), v) for c, v in zip(cost, values)]
                values = [v for _, v in _minus(at_cost, ref)]
                outside = (cost < ref[0][0]) | (cost > ref[0][-1])
                if outside.any():
                    warnings.warn(
                        f"{outside.sum()} of the points of {name!r} are outside the range of x "
                        f"of the baseline {baseline!r}, and are not drawn",
                        stacklevel=3,
                    )
            order = np.argsort(cost, kind="stable")
            _line(
                panel,
                cost[order],
                np.array([v.mean() for v in values])[order],
                np.array([v.min() for v in values])[order],
                np.array([v.max() for v in values])[order],
                name,
                band and any(len(v) > 1 for v in values),
                marker="o",
            )

    ylabel = _ylabel(metric, baseline, "" if summary == "final" else ", area under the curve")
    xlabel = "cost spent" if budgets is None else "budget"
    if average:
        (avg_ax,) = _axes(ax, 1)
        draw(avg_ax, None)
        avg_ax.set(title=_average_title(requested), xlabel=xlabel, ylabel=ylabel)
        _legend(avg_ax, named)
        return avg_ax

    axes = _axes(ax, len(shown))
    for panel, o in zip(axes, shown):
        draw(panel, o)
        panel.set(title=str(o), xlabel=xlabel)
    axes[0].set_ylabel(ylabel)
    _legend(axes[0], named)
    return axes


def queries(
    runs: Run | Mapping[Hashable, Run],
    window: int = 100,
    cost: bool = False,
    every: int | None = None,
    outputs: Iterable[Output] | None = None,
    band: bool = True,
    ax: Axes | Sequence[Axes] | None = None,
) -> list[Axes]:
    """Plot how often each output is queried along the stream, one panel per output.

    This shows how a strategy and a budget share the labels between the outputs, and how this
    changes over time, for example after a drift.

    Parameters
    ----------
    runs
        A run, or a dictionary of runs by name. A run is a log, or a list of logs of the same
        setup, of which the mean is drawn.
    window
        The number of instances to average over. Values start once `window` instances are seen.
    cost
        Whether to draw the cost spent on the output per instance, rather than the share of
        instances for which it was queried.
    every
        Draw a point every `every` instances, and after the last one. Defaults to a tenth of
        `window`. Drawing a point for every instance of a long stream makes the lines look thick
        and jagged.
    outputs
        The outputs to draw, in order. Defaults to all of them.
    band
        Whether to shade the range between the lowest and highest values of runs with several
        logs.
    ax
        The axes to draw on, one per output. Defaults to new axes in a new figure, with the
        same y scale. The y axis starts at 0.

    Returns
    -------
    The axes, in the order of the outputs.

    """
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
    if every is None:
        every = max(1, window // 10)
    if every < 1:
        raise ValueError(f"every must be at least 1, got {every}")
    named = _runs(runs)
    column = "cost" if cost else "queried"
    frames = {name: [log.to_frame() for log in logs] for name, logs in named.items()}
    shown = _outputs(outputs, (o for logs in named.values() for lg in logs for o in lg.outputs))
    # The same scale for all outputs, to compare how the labels are shared between them
    axes = _axes(ax, len(shown), sharey=True)
    for panel, o in zip(axes, shown):
        for name, fs in frames.items():
            series = []
            for f in fs:
                rows = f[f["output"] == o]
                steps = rows["step"].to_numpy(dtype=float)
                rolling = rows[column].astype(float).rolling(window).mean().to_numpy(dtype=float)
                kept = (steps + 1) % every == 0
                kept[-1:] = True
                series.append((steps[kept], rolling[kept]))
            _draw(panel, series, name, band)
        panel.set(title=str(o), xlabel="instances")
    # Shares and costs are never negative, and a zoomed-in axis makes small changes look large.
    # Set once all panels are drawn, since the panels can share their y axis.
    for panel in axes:
        panel.set_ylim(bottom=0)
    what = "cost per instance" if cost else "share of instances queried"
    axes[0].set_ylabel(f"{what}, last {window} instances")
    _legend(axes[0], named)
    return axes
