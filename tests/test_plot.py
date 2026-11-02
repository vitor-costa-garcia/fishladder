import random
import sys
import warnings

import numpy as np
import pytest

import fishladder as fl

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

COSTS = {"a": 1.0, "b": 2.0}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


class MeanModel:
    def __init__(self):
        self.sums = {}
        self.counts = {}

    def predict_one(self, x):
        return {o: self.sums[o] / self.counts[o] for o in self.sums}

    def learn_one(self, x, y):
        for o, v in y.items():
            self.sums[o] = self.sums.get(o, 0.0) + v
            self.counts[o] = self.counts.get(o, 0) + 1


class AbsError:
    def __init__(self):
        self.total = 0.0
        self.n = 0

    def update(self, y_true, y_pred):
        self.total += abs(y_true - y_pred)
        self.n += 1

    def get(self):
        return self.total / self.n if self.n else 0.0

    def __repr__(self):
        return f"AbsError: {self.get()}"


def make_log(n=60, rate=1.0, seed=0):
    rng = random.Random(seed)
    stream = (({}, {"a": rng.random(), "b": 3 * rng.random()}) for _ in range(n))
    return fl.evaluate.prequential(
        stream,
        MeanModel(),
        fl.strategy.RandomPerOutput(p=0.5, seed=seed),
        fl.budget.RateBudget(rate),
        COSTS,
    )


def data(ax, i=0):
    return ax.get_lines()[i].get_xydata()


# curve


def test_curve_draws_metrics_curve():
    log = make_log()
    axes = fl.plot.curve(log, AbsError(), every=5)
    expected = fl.metrics.curve(log, AbsError(), every=5)
    assert [ax.get_title() for ax in axes] == ["a", "b"]
    assert axes[0].get_ylabel() == "AbsError"
    for ax, o in zip(axes, ["a", "b"]):
        c = expected[expected["output"] == o]
        np.testing.assert_allclose(data(ax), c[["step", "value"]].to_numpy(dtype=float))
        assert ax.get_legend() is None


@pytest.mark.parametrize("every", [5, 2.5])
def test_curve_against_cost_draws_points_at_multiples_of_every(every):
    log = make_log()
    axes = fl.plot.curve(log, AbsError(), x="cost", every=every)
    full = fl.metrics.curve(log, AbsError())
    for ax, o in zip(axes, ["a", "b"]):
        c = full[full["output"] == o]
        end = c["cum_cost"].iloc[-1]
        costs = [every * k for k in range(1, int(end // every) + 1)]
        costs += [end] if costs[-1] < end else []
        # The value after the last instance that cost at most each point
        values = [c.loc[c["cum_cost"] <= cost, "value"].iloc[-1] for cost in costs]
        np.testing.assert_allclose(data(ax), np.column_stack([costs, values]))


def test_curve_against_cost_starts_runs_at_same_cost():
    logs = {"slow": make_log(rate=0.5), "fast": make_log(rate=2.0)}
    for result in [
        fl.plot.curve(logs, AbsError(), x="cost", every=4)[0],
        fl.plot.curve(logs, AbsError(), x="cost", every=4, average=True),
    ]:
        slow, fast = (line.get_xdata() for line in result.get_lines())
        assert slow[0] == fast[0] == 4
        assert slow[-1] < fast[-1]


def test_by_cost_is_not_fooled_by_rounding():
    cost = np.cumsum(np.full(200, 0.3))
    x, y = fl.plot._by_cost(cost, np.arange(200.0), 0.3)
    # Each multiple of 0.3 is reached by its own instance, even where the sum rounds below it
    np.testing.assert_array_equal(y, np.arange(200.0))
    assert len(x) == 200


def test_curve_against_cost_has_about_100_points_by_default():
    small = {"a": 0.01, "b": 0.02}
    rng = random.Random(0)
    stream = (({}, {"a": rng.random(), "b": rng.random()}) for _ in range(500))
    log = fl.evaluate.prequential(
        stream, MeanModel(), fl.strategy.QueryAll(), fl.budget.Unlimited(), small
    )
    axes = fl.plot.curve(log, AbsError(), x="cost")
    assert 100 <= len(data(axes[0])) <= 101


def log_with_outputs(costs, n=20, seed=0):
    rng = random.Random(seed)
    stream = (({}, {o: rng.random() for o in costs}) for _ in range(n))
    return fl.evaluate.prequential(
        stream, MeanModel(), fl.strategy.QueryAll(), fl.budget.Unlimited(), costs
    )


def test_curve_draws_runs_with_different_outputs():
    runs = {"both": log_with_outputs({"a": 1.0, "b": 1.0}), "a only": log_with_outputs({"a": 1.0})}
    axes = fl.plot.curve(runs, AbsError())
    assert [ax.get_title() for ax in axes] == ["a", "b"]
    assert len(data(axes[1], 0)) > 0
    assert len(data(axes[1], 1)) == 0
    # The same with the outputs given
    axes = fl.plot.curve(runs, AbsError(), outputs=["b"])
    assert len(data(axes[0], 1)) == 0
    with pytest.raises(KeyError, match="'b'"):
        fl.plot.curve(runs, AbsError(), average=True, outputs=["a", "b"])


def test_budget_draws_runs_with_different_outputs():
    runs = {
        "both": [log_with_outputs({"a": 1.0, "b": 1.0})],
        "a only": [log_with_outputs({"a": 1.0})],
    }
    axes = fl.plot.budget(runs, AbsError(), summary="auc", outputs=["a", "b"])
    assert np.isnan(data(axes[1], 1)[:, 1]).all()
    assert not np.isnan(data(axes[1], 0)[:, 1]).any()


def test_curve_against_cost_has_at_most_about_10000_points():
    cheap = log_with_outputs({"a": 0.01}, n=5)
    dear = log_with_outputs({"a": 100.0}, n=500)
    ax = fl.plot.curve({"cheap": cheap, "dear": dear}, AbsError(), x="cost")[0]
    assert len(data(ax, 1)) <= 10_001


def test_curve_against_cost_counts_from_start_of_log():
    budget = fl.budget.RateBudget(1.0)
    rng = random.Random(0)

    def run():
        stream = (({}, {"a": rng.random(), "b": rng.random()}) for _ in range(60))
        strategy = fl.strategy.RandomPerOutput(p=0.5, seed=0)
        return fl.evaluate.prequential(stream, MeanModel(), strategy, budget, COSTS)

    first, second = run(), run()  # The same budget, so cum_cost of the second goes on
    assert next(iter(second))["cum_cost"] >= first.total_cost > 0
    ax = fl.plot.curve(second, AbsError(), x="cost", every=1, outputs=["a"])[0]
    assert data(ax)[0, 0] == 1
    assert data(ax)[-1, 0] == second.total_cost


def test_by_cost_is_nan_before_first_instance():
    cost = np.array([3.0, 3.0, 5.0])
    x, y = fl.plot._by_cost(cost, np.array([1.0, 2.0, 3.0]), 2)
    np.testing.assert_allclose(x, [2.0, 4.0, 5.0])
    np.testing.assert_allclose(y, [np.nan, 2.0, 3.0])


def test_curve_average():
    log = make_log()
    ax = fl.plot.curve(log, AbsError(), every=5, average=True)
    expected = fl.metrics.average_curve(log, AbsError(), every=5)
    np.testing.assert_allclose(data(ax), expected[["step", "value"]].to_numpy(dtype=float))


def test_curve_names_runs_in_legend():
    logs = {"slow": make_log(rate=0.5), "fast": make_log(rate=2.0)}
    axes = fl.plot.curve(logs, AbsError(), every=10)
    assert [t.get_text() for t in axes[0].get_legend().get_texts()] == ["slow", "fast"]
    assert len(axes[1].get_lines()) == 2


def test_curve_draws_mean_and_band_of_repeats():
    logs = [make_log(seed=s) for s in range(3)]
    ax = fl.plot.curve(logs, AbsError(), every=10, outputs=["a"])[0]
    curves = [fl.metrics.curve(log, AbsError(), every=10) for log in logs]
    values = np.array([c.loc[c["output"] == "a", "value"] for c in curves])
    defined = ~np.isnan(values).any(axis=0)
    np.testing.assert_allclose(data(ax)[:, 1], values.mean(axis=0)[defined])
    assert len(ax.collections) == 1

    ax = fl.plot.curve(logs, AbsError(), every=10, outputs=["a"], band=False)[0]
    assert len(ax.collections) == 0


def test_curve_interpolates_repeats_against_cost():
    logs = [make_log(seed=s) for s in range(2)]
    ax = fl.plot.curve(logs, AbsError(), x="cost", every=5, outputs=["b"])[0]
    xy = data(ax)
    ends = [fl.metrics.curve(log, AbsError())["cum_cost"].max() for log in logs]
    assert xy[:, 0].max() == min(ends)
    assert np.all(np.diff(xy[:, 0]) > 0)


def test_summarize_leaves_out_curves_without_values():
    x = np.array([0.0, 1.0, 2.0])
    series = [
        (x, np.array([1.0, 2.0, 3.0])),
        (x, np.full(3, np.nan)),
        (x, np.array([3.0, 4.0, 5.0])),
    ]
    _, mean, low, high = fl.plot._summarize(series)
    np.testing.assert_allclose(mean, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(low, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(high, [3.0, 4.0, 5.0])
    # With a single curve left, it is returned as it is
    np.testing.assert_allclose(fl.plot._summarize(series[:2])[1], [1.0, 2.0, 3.0])
    assert len(fl.plot._summarize(series[1:2] * 2)[0]) == 0


def test_summarize_interpolates_on_common_range():
    series = [
        (np.array([0.0, 1.0, 2.0, 3.0]), np.array([np.nan, 1.0, 3.0, 5.0])),
        (np.array([0.0, 2.0, 2.0, 4.0]), np.array([0.0, 9.0, 2.0, 4.0])),
    ]
    x, mean, low, high = fl.plot._summarize(series)
    # The first curve starts at 1 and ends at 3; the second keeps 2.0 at x = 2
    np.testing.assert_allclose(x, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(low, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(high, [1.0, 3.0, 5.0])
    np.testing.assert_allclose(mean, [1.0, 2.5, 4.0])


def test_curve_uses_given_axes():
    _, given = plt.subplots(1, 2)
    axes = fl.plot.curve(make_log(), AbsError(), ax=given)
    assert list(axes) == list(given)
    with pytest.raises(ValueError, match="need 2 axes"):
        fl.plot.curve(make_log(), AbsError(), ax=given[0])

    # A grid of axes is read row by row
    _, grid = plt.subplots(2, 1)
    assert list(fl.plot.curve(make_log(), AbsError(), ax=grid.reshape(2, 1))) == list(grid)


def test_curve_averages_given_outputs():
    log = make_log()
    ax = fl.plot.curve(log, AbsError(), every=5, average=True, outputs=["b"])
    expected = fl.metrics.curve(log, AbsError(), every=5)
    expected = expected[expected["output"] == "b"]
    np.testing.assert_allclose(data(ax), expected[["step", "value"]].to_numpy(dtype=float))
    assert ax.get_title() == "average over b"
    with pytest.raises(KeyError, match="'z'"):
        fl.plot.curve(log, AbsError(), average=True, outputs=["z"])


def test_curve_rejects_bad_arguments():
    log = make_log()
    with pytest.raises(ValueError, match="no outputs"):
        fl.plot.curve(fl.Log(), AbsError())
    with pytest.raises(ValueError, match="no outputs"):
        fl.plot.queries(log, outputs=[])
    with pytest.raises(ValueError, match="x must be"):
        fl.plot.curve(log, AbsError(), x="time")
    with pytest.raises(ValueError, match="whole number"):
        fl.plot.curve(log, AbsError(), every=2.5)
    for every in [0, -1, float("inf"), float("nan")]:
        for x in ["step", "cost"]:
            with pytest.raises(ValueError, match="positive number"):
                fl.plot.curve(log, AbsError(), x=x, every=every)
    with pytest.raises(TypeError, match="not a string"):
        fl.plot.curve(log, AbsError(), outputs="a")
    with pytest.raises(KeyError, match="'z'"):
        fl.plot.curve(log, AbsError(), outputs=["z"])
    with pytest.raises(ValueError, match="at least one log"):
        fl.plot.curve({"empty": []}, AbsError())
    with pytest.raises(TypeError, match="DataFrame"):
        fl.plot.curve(log.to_frame(), AbsError())


def test_curve_draws_difference_to_baseline():
    runs = {
        "random": [make_log(rate=0.5, seed=s) for s in range(3)],
        "fast": [make_log(rate=2.0, seed=s) for s in range(2)],
    }
    ax = fl.plot.curve(runs, AbsError(), every=10, average=True, baseline="random")
    assert ax.get_ylabel() == "AbsError, difference to random"

    def mean(logs):
        curves = [fl.metrics.average_curve(log, AbsError(), every=10) for log in logs]
        return fl.plot._summarize([fl.plot._xy(c, "step") for c in curves])[:2]

    base_x, base_y = mean(runs["random"])
    fast_x, fast_y = mean(runs["fast"])
    random_line, fast_line = ax.get_lines()
    # The baseline is drawn at 0, with the range of its own logs around it
    np.testing.assert_allclose(random_line.get_ydata(), 0, atol=1e-12)
    # The difference is only defined where the baseline has values
    inside = (fast_x >= base_x[0]) & (fast_x <= base_x[-1])
    assert not inside.all()
    expected = fast_y[inside] - np.interp(fast_x[inside], base_x, base_y)
    np.testing.assert_allclose(fast_line.get_xydata(), np.column_stack([fast_x[inside], expected]))
    assert len(ax.collections) == 2


def test_minus_is_nan_outside_reference():
    ref = (np.array([1.0, 3.0]), np.array([1.0, 3.0]))
    ((_, y),) = fl.plot._minus([(np.array([0.0, 2.0, 4.0]), np.array([5.0, 5.0, 5.0]))], ref)
    np.testing.assert_allclose(y, [np.nan, 3.0, np.nan])


@pytest.mark.parametrize("plot", [fl.plot.curve, fl.plot.budget])
def test_baseline_must_be_a_named_run(plot):
    logs = [make_log(rate=r) for r in [0.5, 1.0]]
    with pytest.raises(ValueError, match="dictionary"):
        plot(logs, AbsError(), baseline="random")
    with pytest.raises(KeyError, match="'random'"):
        plot({"qbc": logs}, AbsError(), baseline="random")


# budget


def test_budget_draws_final_metric_against_total_cost():
    rates = [2.0, 0.5, 1.0]
    logs = [make_log(rate=r) for r in rates]
    axes = fl.plot.budget(logs, AbsError())
    expected = sorted(
        (log.to_frame()["cost"].sum(), fl.metrics.per_output(log, AbsError())["b"].get())
        for log in logs
    )
    assert axes[1].get_title() == "b"
    assert axes[0].get_xlabel() == "cost spent"
    np.testing.assert_allclose(data(axes[1]), expected)


def test_budget_auc_and_average():
    logs = [make_log(rate=r) for r in [0.5, 2.0]]
    ax = fl.plot.budget({"random": logs}, AbsError(), summary="auc", every=5, average=True)
    expected = [fl.metrics.auc(log, AbsError(), every=5, average=True) for log in logs]
    np.testing.assert_allclose(data(ax)[:, 1], expected)
    assert ax.get_ylabel() == "AbsError, area under the curve"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["random"]


def test_budget_averages_repeats():
    levels = [[make_log(rate=r, seed=s) for s in range(3)] for r in [0.5, 2.0]]
    ax = fl.plot.budget(levels, AbsError(), average=True)
    for (cost, value), logs in zip(data(ax), levels):
        assert cost == pytest.approx(np.mean([log.to_frame()["cost"].sum() for log in logs]))
        assert value == pytest.approx(np.mean([fl.metrics.average(lg, AbsError()) for lg in logs]))
    assert len(ax.collections) == 1


def test_budget_draws_against_given_budgets():
    rates = [2.0, 0.5, 1.0]
    logs = [make_log(rate=r) for r in rates]
    ax = fl.plot.budget(logs, AbsError(), budgets=rates, average=True)
    expected = sorted((r, fl.metrics.average(log, AbsError())) for r, log in zip(rates, logs))
    np.testing.assert_allclose(data(ax), expected)
    assert ax.get_xlabel() == "budget"

    # A dictionary gives the budgets of each setup, which can have different numbers of runs
    runs = {"few": logs[:2], "all": logs}
    ax = fl.plot.budget(runs, AbsError(), budgets={"few": rates[:2], "all": rates}, average=True)
    np.testing.assert_allclose(data(ax, 0)[:, 0], [0.5, 2.0])
    np.testing.assert_allclose(data(ax, 1)[:, 0], [0.5, 1.0, 2.0])


def test_budget_rejects_bad_budgets():
    logs = [make_log(rate=r) for r in [0.5, 1.0]]
    with pytest.raises(ValueError, match="2 runs, but 1 budgets"):
        fl.plot.budget(logs, AbsError(), budgets=[0.5])
    with pytest.raises(KeyError, match="'random'"):
        fl.plot.budget({"random": logs}, AbsError(), budgets={"qbc": [0.5, 1.0]})


def test_budget_draws_difference_to_baseline():
    rates = [0.5, 1.0, 2.0]
    runs = {
        "random": [[make_log(rate=r, seed=s) for s in range(2)] for r in rates],
        "fast": [make_log(rate=r, seed=5) for r in rates[1:]],
    }
    ax = fl.plot.budget(
        runs,
        AbsError(),
        budgets={"random": rates, "fast": rates[1:]},
        average=True,
        baseline="random",
    )
    random_line, fast_line = ax.get_lines()
    np.testing.assert_allclose(random_line.get_ydata(), 0, atol=1e-12)
    base = [np.mean([fl.metrics.average(lg, AbsError()) for lg in logs]) for logs in runs["random"]]
    fast = [fl.metrics.average(lg, AbsError()) for lg in runs["fast"]]
    np.testing.assert_allclose(
        fast_line.get_xydata(), np.column_stack([rates[1:], np.subtract(fast, base[1:])])
    )
    assert ax.get_ylabel() == "AbsError, difference to random"


def test_budget_average_matches_metrics_when_an_output_is_never_predicted():
    log = fl.Log()
    log.append_one(0, {"a": 1.0, "b": 1.0}, {}, {}, [], COSTS, 0.0)
    log.append_one(1, {"a": 1.0, "b": 1.0}, {"a": 0.0}, {}, [], COSTS, 0.0)
    ax = fl.plot.budget([log], AbsError(), average=True)
    assert data(ax)[0, 1] == fl.metrics.average(log, AbsError()) == 1.0


def test_budget_merges_baseline_budgets_with_the_same_cost():
    # Budgets so large that they never limit the queries, so both runs spend the same
    same = [make_log(rate=5.0), make_log(rate=6.0)]
    assert same[0].total_cost == same[1].total_cost
    runs = {"random": same, "other": [make_log(rate=5.0, seed=1)]}
    # The baseline covers a single cost, so other costs cannot be compared to it
    with pytest.warns(UserWarning, match="'other' are outside"):
        ax = fl.plot.budget(runs, AbsError(), average=True, baseline="random")
    random_line, _ = ax.get_lines()
    # The two budgets of the baseline are compared to their mean, which is drawn at 0
    assert np.mean(random_line.get_ydata()) == pytest.approx(0)
    assert not np.isnan(random_line.get_ydata()).any()


def test_budget_keeps_order_of_outputs_in_log():
    log = fl.Log()
    # b is predicted before a, but comes after it in the log
    log.append_one(0, {"a": 1.0, "b": 1.0}, {"b": 0.0}, {}, [], COSTS, 0.0)
    log.append_one(1, {"a": 1.0, "b": 1.0}, {"a": 0.0, "b": 0.0}, {}, [], COSTS, 0.0)
    axes = fl.plot.budget([log], AbsError())
    assert [ax.get_title() for ax in axes] == ["a", "b"]


@pytest.mark.parametrize("summary", ["final", "auc"])
def test_budget_averages_given_outputs(summary):
    logs = [make_log(rate=r) for r in [0.5, 2.0]]
    ax = fl.plot.budget(logs, AbsError(), summary=summary, average=True, outputs=["b"])
    if summary == "final":
        expected = [fl.metrics.per_output(lg, AbsError())["b"].get() for lg in logs]
    else:
        expected = [fl.metrics.auc(lg, AbsError())["b"] for lg in logs]
    np.testing.assert_allclose(data(ax)[:, 1], expected)
    assert ax.get_title() == "average over b"


def test_budget_warns_about_points_outside_baseline():
    runs = {
        "random": [make_log(rate=r) for r in [1.0, 2.0]],
        "cheap": [make_log(rate=r) for r in [0.25, 1.5]],
    }
    with pytest.warns(UserWarning, match="1 of the points of 'cheap' are outside"):
        ax = fl.plot.budget(runs, AbsError(), average=True, baseline="random")
    # The point below the cost of the baseline is not drawn
    assert np.isnan(data(ax, 1)[:, 1]).sum() == 1

    # With the same budgets, all points can be compared
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fl.plot.budget(runs, AbsError(), budgets=[1.0, 2.0], average=True, baseline="random")


def test_budget_rejects_a_single_log():
    with pytest.raises(TypeError, match="not a single log"):
        fl.plot.budget(make_log(), AbsError())
    with pytest.raises(TypeError, match="for 'random', one per budget, not a single log"):
        fl.plot.budget({"random": make_log()}, AbsError())


def test_budget_rejects_bad_arguments():
    with pytest.raises(ValueError, match="summary must be"):
        fl.plot.budget([make_log()], AbsError(), summary="last")
    with pytest.raises(ValueError, match="no budgets"):
        fl.plot.budget({"random": []}, AbsError())


# queries


@pytest.mark.parametrize("cost", [False, True])
def test_queries_draws_rolling_mean(cost):
    log = make_log()
    axes = fl.plot.queries(log, window=10, cost=cost, every=1)
    df = log.to_frame()
    b = df[df["output"] == "b"]
    expected = b["cost" if cost else "queried"].astype(float).rolling(10).mean()
    np.testing.assert_allclose(data(axes[1])[:, 1], expected.to_numpy(), equal_nan=True)


def test_queries_draws_every_few_instances_from_zero():
    logs = {"slow": make_log(n=95, rate=0.5), "fast": make_log(n=95, rate=2.0)}
    axes = fl.plot.queries(logs, window=20)
    # A point every 20 // 10 = 2 instances, and one after the last
    np.testing.assert_array_equal(data(axes[0])[:, 0], [*range(1, 95, 2), 94])
    for ax in axes:
        bottom, top = ax.get_ylim()
        assert bottom == 0
        for line in ax.get_lines():
            assert np.nanmax(line.get_ydata()) <= top
    with pytest.raises(ValueError, match="every"):
        fl.plot.queries(logs, every=0)


def test_queries_rejects_bad_window():
    with pytest.raises(ValueError, match="window"):
        fl.plot.queries(make_log(), window=0)


def test_missing_matplotlib_is_explained(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r"fishladder\[plots\]"):
        fl.plot.curve(make_log(), AbsError())


def test_import_does_not_need_matplotlib():
    import subprocess

    code = (
        "import sys; sys.modules['matplotlib'] = None; import fishladder; "
        "assert 'matplotlib.pyplot' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
