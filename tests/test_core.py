import itertools
import math
import os
import random
import statistics
import subprocess
import sys

import numpy as np
import pytest

import fishladder as fl


class MeanModel:
    """Predicts the running mean of each output it has seen, and records what it learned from."""

    def __init__(self):
        self.sums = {}
        self.counts = {}
        self.learned = []

    def predict_one(self, x):
        return {o: self.sums[o] / self.counts[o] for o in self.sums}

    def learn_one(self, x, y):
        self.learned.append(dict(y))
        for o, v in y.items():
            self.sums[o] = self.sums.get(o, 0.0) + v
            self.counts[o] = self.counts.get(o, 0) + 1


class AbsError:
    """A minimal River-style metric."""

    def __init__(self):
        self.total = 0.0
        self.n = 0

    def update(self, y_true, y_pred):
        self.total += abs(y_true - y_pred)
        self.n += 1

    def get(self):
        return self.total / self.n if self.n else 0.0


def make_stream(n, seed=0):
    rng = random.Random(seed)
    for i in range(n):
        yield {"i": i}, {"a": rng.random(), "b": rng.random(), "c": rng.random()}


COSTS = {"a": 1.0, "b": 2.0, "c": 5.0}


def test_model_satisfies_protocol():
    assert isinstance(MeanModel(), fl.base.Model)


# Budget


def test_select_is_greedy_by_utility_and_skips_unaffordable():
    b = fl.budget.TotalBudget(6.0)
    utilities = {"a": 0.1, "b": 0.5, "c": 0.9}
    # c (5.0) is picked first, b (2.0) no longer fits, a (1.0) still does
    assert b.select_one(utilities, COSTS) == {"a", "c"}
    assert b.spent == 0.0


def test_select_ignores_non_positive_utilities():
    b = fl.budget.Unlimited()
    assert b.select_one({"a": 0.0, "b": -1.0, "c": 0.2}, COSTS) == {"c"}


def test_rate_budget_carries_over():
    b = fl.budget.RateBudget(1.0)
    assert b.select_one({"c": 1.0}, COSTS) == set()
    for _ in range(4):
        b.update({})
    assert b.available == 5.0
    assert b.select_one({"c": 1.0}, COSTS) == {"c"}


@pytest.mark.parametrize("cls", [fl.budget.TotalBudget, fl.budget.RateBudget])
def test_budget_rejects_negative(cls):
    with pytest.raises(ValueError):
        cls(-1.0)


def test_unlimited():
    assert fl.budget.Unlimited().available == math.inf


def run_per_output(budget, n=300):
    model = fl.active.Wrapper(MeanModel(), fl.strategy.QueryAll(), budget, COSTS)
    for x, _ in make_stream(n):
        model.predict_one(x)
    return budget


def test_per_output_does_not_starve_expensive_outputs():
    shared = run_per_output(fl.budget.RateBudget(0.5))
    split = run_per_output(fl.budget.PerOutput(fl.budget.RateBudget(0.5)))
    assert shared.spent == pytest.approx(split.spent, abs=max(COSTS.values()))
    assert split.spent_per_output["c"] > 0


@pytest.mark.parametrize("shares", [None, {"a": 1.0, "b": 2.0, "c": 7.0}])
def test_per_output_respects_shares(shares):
    n, rate = 300, 1.2
    b = run_per_output(fl.budget.PerOutput(fl.budget.RateBudget(rate), shares=shares), n)
    weights = shares or {o: 1.0 for o in COSTS}
    for o, w in weights.items():
        allowed = rate * n * w / sum(weights.values())
        assert allowed - COSTS[o] <= b.spent_per_output[o] <= allowed


def test_per_output_shares_as_costs_equalize_query_rates():
    b = run_per_output(fl.budget.PerOutput(fl.budget.RateBudget(0.8), shares=COSTS), n=400)
    n_queries = {o: b.spent_per_output[o] / COSTS[o] for o in COSTS}
    assert max(n_queries.values()) - min(n_queries.values()) <= 1


def test_per_output_with_total_budget():
    b = run_per_output(fl.budget.PerOutput(fl.budget.TotalBudget(30.0)))
    for o, cost in COSTS.items():
        assert 10.0 - cost < b.spent_per_output[o] <= 10.0


def test_per_output_does_not_modify_given_budget():
    inner = fl.budget.RateBudget(1.0)
    run_per_output(fl.budget.PerOutput(inner))
    assert inner.spent == 0.0
    assert inner.n_seen == 0


def test_per_output_rejects_bad_shares():
    with pytest.raises(ValueError):
        fl.budget.PerOutput(fl.budget.RateBudget(1.0), shares={"a": 0.0})
    b = fl.budget.PerOutput(fl.budget.RateBudget(1.0), shares={"a": 1.0})
    with pytest.raises(KeyError):
        b.select_one({"a": 1.0}, COSTS)


# Oracle


def test_oracle_charges_requested_outputs_even_if_missing():
    oracle = fl.Oracle(COSTS)
    y, cost = oracle.reveal({"a": 1.0}, ["a", "c"])
    assert y == {"a": 1.0}
    assert cost == 6.0


def test_oracle_rejects_negative_cost():
    with pytest.raises(ValueError):
        fl.Oracle({"a": -1.0})


# Strategies


def test_random_per_output_is_reproducible():
    u1 = [fl.strategy.RandomPerOutput(0.5, seed=1).utilities_one({}, {}, COSTS) for _ in range(3)]
    u2 = [fl.strategy.RandomPerOutput(0.5, seed=1).utilities_one({}, {}, COSTS) for _ in range(3)]
    assert u1 == u2


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_random_per_output_rejects_bad_p(p):
    with pytest.raises(ValueError):
        fl.strategy.RandomPerOutput(p)


# Wrapper


def test_wrapper_charges_budget_at_predict_time():
    model = fl.active.Wrapper(
        MeanModel(), fl.strategy.QueryAll(), fl.budget.TotalBudget(3.0), COSTS
    )
    y_pred, ask = model.predict_one({})
    assert y_pred == {}
    assert ask == {"a", "b"}
    assert model.budget.spent == 3.0
    _, ask = model.predict_one({})
    assert ask == set()


def test_wrapper_ignores_empty_labels():
    inner = MeanModel()
    model = fl.active.Wrapper(inner, fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS)
    model.learn_one({}, {})
    assert inner.learned == []


# Evaluation


@pytest.mark.parametrize(
    "budget",
    [
        fl.budget.TotalBudget(40.0),
        fl.budget.RateBudget(1.5),
        fl.budget.Unlimited(),
        fl.budget.PerOutput(fl.budget.RateBudget(1.5)),
    ],
)
def test_prequential_never_overspends(budget):
    log = fl.evaluate.prequential(
        make_stream(100), MeanModel(), fl.strategy.RandomPerOutput(0.7, seed=0), budget, COSTS
    )
    df = log.to_frame()
    per_step = df.groupby("step")["cost"].sum()
    assert per_step.cumsum().tolist() == pytest.approx(df.groupby("step")["cum_cost"].first())
    if isinstance(budget, fl.budget.TotalBudget):
        assert budget.spent <= budget.total
    if isinstance(budget, fl.budget.RateBudget):
        assert (per_step.cumsum() <= budget.rate * (per_step.index + 1) + 1e-9).all()


def test_prequential_model_learns_only_revealed_outputs():
    model = MeanModel()
    log = fl.evaluate.prequential(
        make_stream(50),
        model,
        fl.strategy.RandomPerOutput(0.3, seed=0),
        fl.budget.Unlimited(),
        COSTS,
    )
    df = log.to_frame()
    queried = df[df["queried"]]
    learned = [(o, v) for y in model.learned for o, v in y.items()]
    assert sorted(learned) == sorted(zip(queried["output"], queried["y_true"]))


def test_prequential_predicts_before_learning():
    stream = [({}, {"a": 1.0, "b": 1.0, "c": 1.0}), ({}, {"a": 3.0, "b": 3.0, "c": 3.0})]
    log = fl.evaluate.prequential(
        stream, MeanModel(), fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS
    )
    df = log.to_frame()
    assert df[df["step"] == 0]["y_pred"].isna().all()
    assert (df[df["step"] == 1]["y_pred"] == 1.0).all()


def test_log_shape():
    log = fl.evaluate.prequential(
        make_stream(10), MeanModel(), fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS
    )
    df = log.to_frame()
    assert len(df) == 30
    assert list(df.columns) == [
        "step",
        "output",
        "y_true",
        "y_pred",
        "utility",
        "probability",
        "queried",
        "cost",
        "cum_cost",
        "delay",
    ]


def test_iter_prequential_reports():
    reports = list(
        fl.evaluate.iter_prequential(
            make_stream(25),
            MeanModel(),
            fl.strategy.QueryAll(),
            fl.budget.Unlimited(),
            COSTS,
            step=10,
        )
    )
    assert [r["Step"] for r in reports] == [10, 20, 25]
    assert reports[-1]["Queries"] == 75
    assert reports[-1]["Cost"] == 25 * 8.0


def test_prequential_defaults_learn_every_label():
    model = MeanModel()
    log = fl.evaluate.prequential(iter(make_stream(20)), model)
    df = log.to_frame()
    assert len(df) == 20 * len(COSTS)
    assert df["queried"].all()
    assert (df["cost"] == 1.0).all()
    assert model.learned == [y for _, y in make_stream(20)]


def test_prequential_default_costs_with_given_strategy_and_budget():
    log = fl.evaluate.prequential(
        make_stream(10), MeanModel(), fl.strategy.QueryAll(), fl.budget.TotalBudget(5.0)
    )
    assert log.to_frame()["cost"].sum() == 5.0


def test_prequential_defaults_on_empty_stream():
    assert fl.evaluate.prequential([], MeanModel()).to_frame().empty


# Metrics


def test_per_output_matches_manual_computation():
    log = fl.evaluate.prequential(
        make_stream(30), MeanModel(), fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS
    )
    df = log.to_frame().dropna(subset=["y_pred"])
    result = fl.metrics.per_output(log, AbsError())
    for o in COSTS:
        rows = df[df["output"] == o]
        assert result[o].get() == pytest.approx((rows["y_true"] - rows["y_pred"]).abs().mean())


def test_curve_last_value_matches_per_output():
    log = fl.evaluate.prequential(
        make_stream(30), MeanModel(), fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS
    )
    curve = fl.metrics.curve(log, AbsError(), every=5)
    assert sorted(curve["step"].unique()) == [4, 9, 14, 19, 24, 29]
    final = curve[curve["step"] == 29].set_index("output")["value"]
    expected = fl.metrics.per_output(log, AbsError())
    for o in COSTS:
        assert final[o] == pytest.approx(expected[o].get())


def rrmse(y_true, y_pred, baseline):
    metric = fl.metrics.RRMSE(baseline=baseline)
    for yt, yp in zip(y_true, y_pred):
        metric.update(yt, yp)
    return metric.get()


def test_rrmse_cumulative_uses_mean_of_all_values():
    rng = np.random.default_rng(0)
    y_true, y_pred = rng.normal(size=50), rng.normal(size=50)
    expected = math.sqrt(((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())
    assert rrmse(y_true, y_pred, "cumulative") == pytest.approx(expected)


def test_rrmse_prequential_uses_mean_of_previous_values():
    rng = np.random.default_rng(0)
    y_true, y_pred = rng.normal(size=50), rng.normal(size=50)
    # As in MOA, the mean is 0 before any value is seen
    means = [0.0] + [y_true[:i].mean() for i in range(1, 50)]
    expected = math.sqrt(((y_true - y_pred) ** 2).sum() / ((y_true - means) ** 2).sum())
    assert rrmse(y_true, y_pred, "prequential") == pytest.approx(expected)


@pytest.mark.parametrize("baseline", ["cumulative", "prequential"])
def test_rrmse_weights_count_as_repeats(baseline):
    weighted = fl.metrics.RRMSE(baseline=baseline)
    for yt, yp, w in [(1.0, 0.5, 2.0), (3.0, 2.0, 1.0), (-1.0, 0.0, 3.0)]:
        weighted.update(yt, yp, w)
    # Prequential differs: repeats see each other in the running mean
    if baseline == "cumulative":
        expected = rrmse(
            [1.0, 1.0, 3.0, -1.0, -1.0, -1.0], [0.5, 0.5, 2.0, 0.0, 0.0, 0.0], baseline
        )
        assert weighted.get() == pytest.approx(expected)
    else:
        mean_1 = 1.0  # after 1.0 with weight 2
        mean_2 = 5 / 3
        squared_error = 2 * 0.25 + 1.0 + 3 * 1.0
        baseline_error = 2 * 1.0 + (3.0 - mean_1) ** 2 + 3 * (-1.0 - mean_2) ** 2
        assert weighted.get() == pytest.approx(math.sqrt(squared_error / baseline_error))


@pytest.mark.parametrize("baseline", ["cumulative", "prequential"])
def test_rrmse_is_nan_while_mean_makes_no_error(baseline):
    metric = fl.metrics.RRMSE(baseline=baseline)
    assert math.isnan(metric.get())
    if baseline == "cumulative":
        metric.update(2.0, 1.0)
        metric.update(2.0, 1.0)
        assert math.isnan(metric.get())
    assert repr(fl.metrics.RRMSE()) == "RRMSE: nan"


def test_rrmse_rejects_unknown_baseline():
    with pytest.raises(ValueError, match="baseline"):
        fl.metrics.RRMSE(baseline="whole")


def test_rrmse_works_with_per_output_and_curve():
    log = fl.evaluate.prequential(
        make_stream(30), MeanModel(), fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS
    )
    df = log.to_frame().dropna(subset=["y_pred"])
    result = fl.metrics.per_output(log, fl.metrics.RRMSE())
    curve = fl.metrics.curve(log, fl.metrics.RRMSE())
    final = curve[curve["step"] == 29].set_index("output")["value"]
    for o in COSTS:
        rows = df[df["output"] == o]
        expected = rrmse(rows["y_true"], rows["y_pred"], "cumulative")
        assert result[o].get() == pytest.approx(expected)
        assert final[o] == pytest.approx(expected)


def test_average_is_mean_over_outputs():
    log = fl.evaluate.prequential(
        make_stream(30), MeanModel(), fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS
    )
    per_output = fl.metrics.per_output(log, fl.metrics.RRMSE())
    expected = statistics.fmean(m.get() for m in per_output.values())
    assert fl.metrics.average(log, fl.metrics.RRMSE()) == pytest.approx(expected)

    curve = fl.metrics.average_curve(log, fl.metrics.RRMSE(), every=5)
    assert list(curve.columns) == ["step", "cum_cost", "value"]
    assert curve["step"].tolist() == [4, 9, 14, 19, 24, 29]
    assert curve["value"].iloc[-1] == pytest.approx(expected)


def test_average_curve_is_nan_until_every_output_is_scored():
    log = fl.evaluate.prequential(
        make_stream(30), MeanModel(), fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS
    )
    per_output = fl.metrics.curve(log, AbsError())
    curve = fl.metrics.average_curve(log, AbsError())
    some_nan = per_output.groupby("step")["value"].agg(lambda v: v.isna().any())
    assert curve.set_index("step")["value"].isna().equals(some_nan)


def test_area_is_normalized_by_cost_range():
    from fishladder.metrics import _area

    # Constant curves have their value as area
    assert _area([0.0, 5.0, 10.0], [2.0, 2.0, 2.0]) == pytest.approx(2.0)
    # Linear from 1 to 3 averages 2, over unevenly spaced costs too
    assert _area([10.0, 11.0, 20.0], [1.0, 1.2, 3.0]) == pytest.approx(2.0)
    # Points where the metric is undefined are skipped
    assert _area([0.0, 1.0, 2.0, 4.0], [math.nan, 4.0, 2.0, 2.0]) == pytest.approx(7 / 3)
    # No cost spent while the metric is defined
    assert math.isnan(_area([0.0, 0.0], [1.0, 2.0]))
    assert math.isnan(_area([0.0, 1.0], [math.nan, 2.0]))


def test_auc_integrates_curve_against_cost():
    from fishladder.metrics import _area

    log = fl.evaluate.prequential(
        make_stream(30), MeanModel(), fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS
    )
    result = fl.metrics.auc(log, fl.metrics.RRMSE(), every=3)
    curve = fl.metrics.curve(log, fl.metrics.RRMSE(), every=3)
    assert set(result) == set(COSTS)
    for o in COSTS:
        c = curve[curve["output"] == o]
        assert result[o] == pytest.approx(_area(c["cum_cost"].tolist(), c["value"].tolist()))

    average = fl.metrics.average_curve(log, fl.metrics.RRMSE(), every=3)
    expected = _area(average["cum_cost"].tolist(), average["value"].tolist())
    assert fl.metrics.auc(log, fl.metrics.RRMSE(), every=3, average=True) == pytest.approx(expected)


# Wrapper passes predictions to the strategy


class SpyStrategy(fl.strategy.QueryAll):
    def __init__(self):
        self.seen = []

    def learn_one(self, x, y, y_pred):
        self.seen.append(dict(y_pred))


def test_wrapper_gives_strategy_prediction_made_before_learning():
    spy = SpyStrategy()
    model = fl.active.Wrapper(MeanModel(), spy, fl.budget.Unlimited(), COSTS)
    stream = [({}, {"a": 1.0, "b": 1.0, "c": 1.0}), ({}, {"a": 3.0, "b": 3.0, "c": 3.0})]
    for x, y in stream:
        model.predict_one(x)
        model.learn_one(x, y)
    assert spy.seen == [{}, {"a": 1.0, "b": 1.0, "c": 1.0}]


def test_wrapper_recomputes_prediction_for_another_x():
    spy = SpyStrategy()
    model = fl.active.Wrapper(MeanModel(), spy, fl.budget.Unlimited(), COSTS)
    model.learn_one({}, {"a": 2.0})
    model.predict_one({})
    model.learn_one({}, {"a": 4.0})
    assert spy.seen == [{}, {"a": 2.0}]


# Uncertainty strategies


class Constant:
    """A single-output regressor predicting a fixed value, counting how often it learns."""

    def __init__(self, value=0.0):
        self.value = value
        self.n = 0

    def predict_one(self, x):
        return self.value

    def learn_one(self, x, y):
        self.n += 1


@pytest.mark.parametrize(
    "strategy",
    [
        fl.strategy.QueryByCommittee(Constant(), min_labels=1, seed=0),
        fl.strategy.ExpectedError(Constant()),
    ],
)
def test_unseen_outputs_get_infinite_utility(strategy):
    strategy.learn_one({}, {"a": 1.0}, {"a": 0.0})
    utilities = strategy.utilities_one({}, {"a": 0.0}, COSTS)
    assert utilities["a"] < math.inf
    assert utilities["b"] == utilities["c"] == math.inf


def test_qbc_utility_is_committee_spread():
    s = fl.strategy.QueryByCommittee(Constant(), n_members=4, min_labels=1, seed=0)
    s.learn_one({}, {"a": 1.0}, {})
    for i, member in enumerate(s._committees["a"]):
        member.value = float(i)
    assert s.utilities_one({}, {}, {"a": 1.0})["a"] == pytest.approx(1.118, abs=1e-3)


def test_qbc_online_bagging():
    regressor = Constant()
    s = fl.strategy.QueryByCommittee(regressor, n_members=10, seed=0)
    for _ in range(200):
        s.learn_one({}, {"a": 1.0}, {})
    counts = [m.n for m in s._committees["a"]]
    assert regressor.n == 0
    assert len(set(counts)) > 1
    assert sum(counts) / len(counts) == pytest.approx(200, rel=0.1)


def test_qbc_is_reproducible():
    def run():
        s = fl.strategy.QueryByCommittee(Constant(), seed=3)
        for _ in range(20):
            s.learn_one({}, {"a": 1.0}, {})
        return [m.n for m in s._committees["a"]]

    assert run() == run()


def test_qbc_needs_min_labels_before_trusting_committee():
    s = fl.strategy.QueryByCommittee(Constant(), min_labels=3, seed=0)
    for expected in (math.inf, math.inf, math.inf, s._floor):
        assert s.utilities_one({}, {}, {"a": 1.0})["a"] == expected
        s.learn_one({}, {"a": 1.0}, {})


@pytest.mark.parametrize("kwargs", [{"n_members": 1}, {"min_labels": 0}])
def test_qbc_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        fl.strategy.QueryByCommittee(Constant(), **kwargs)


class Recorder(Constant):
    def __init__(self):
        super().__init__()
        self.targets = []

    def learn_one(self, x, y):
        self.targets.append(y)


def test_expected_error_learns_absolute_errors_of_revealed_outputs():
    s = fl.strategy.ExpectedError(Recorder())
    s.learn_one({}, {"a": 1.0, "b": 5.0}, {"a": 3.0})
    assert s._error_models["a"].targets == [2.0]
    assert "b" not in s._error_models


def test_expected_error_utility_stays_positive():
    s = fl.strategy.ExpectedError(Constant(-1.0))
    s.learn_one({}, {"a": 1.0}, {"a": 1.0})
    assert s.utilities_one({}, {}, {"a": 1.0})["a"] > 0


# Variable threshold


def test_variable_threshold_queries_high_utilities():
    rng = random.Random(0)
    b = fl.budget.VariableThreshold(fl.budget.Unlimited(), step=0.05)
    queried, skipped = [], []
    for _ in range(2000):
        u = rng.random()
        q = b.select_one({"a": u}, {"a": 1.0})
        b.update({o: 1.0 for o in q})
        (queried if q else skipped).append(u)
    assert queried and skipped
    assert min(queried) > 0
    assert sum(queried) / len(queried) > sum(skipped) / len(skipped) + 0.3


def test_variable_threshold_only_adapts_when_affordable():
    b = fl.budget.VariableThreshold(fl.budget.TotalBudget(1.0), step=0.5)
    b.update({o: 1.0 for o in b.select_one({"a": 1.0}, {"a": 1.0})})
    assert b.thresholds["a"] == 1.5
    for _ in range(5):
        assert b.select_one({"a": 100.0}, {"a": 1.0}) == set()
        b.update({})
    assert b.thresholds["a"] == 1.5


def test_variable_threshold_never_overspends():
    log = fl.evaluate.prequential(
        make_stream(300),
        MeanModel(),
        fl.strategy.ExpectedError(Constant(1.0)),
        fl.budget.VariableThreshold(fl.budget.PerOutput(fl.budget.RateBudget(1.0))),
        COSTS,
    )
    df = log.to_frame()
    assert df["cost"].sum() <= 300 * 1.0


@pytest.mark.parametrize("step", [0.0, 1.0])
def test_variable_threshold_rejects_bad_step(step):
    with pytest.raises(ValueError):
        fl.budget.VariableThreshold(fl.budget.Unlimited(), step=step)


REPRO_SCRIPT = """
import random
import statistics

import fishladder as fl


class Mean:
    def __init__(self):
        self.total, self.n = 0.0, 0

    def predict_one(self, x):
        return self.total / self.n if self.n else 0.0

    def learn_one(self, x, y):
        self.total += y * x["x"]
        self.n += 1


class MultiMean:
    def __init__(self):
        self.models = {}

    def predict_one(self, x):
        return {o: m.predict_one(x) for o, m in self.models.items()}

    def learn_one(self, x, y):
        for o, y_o in y.items():
            self.models.setdefault(o, Mean()).learn_one(x, y_o)


rng = random.Random(0)
stream = [
    ({"x": rng.random()}, {"a": rng.random(), "b": rng.random(), "c": rng.random()})
    for _ in range(300)
]


class Proportional(fl.strategy.Randomized):
    def probability_one(self, output, utility, recent):
        return utility / statistics.mean(recent)


def budget():
    return fl.budget.PerOutput(fl.budget.RateBudget(1.0))


def committee():
    return fl.strategy.QueryByCommittee(Mean(), seed=0)


setups = [
    (committee(), fl.budget.VariableThreshold(budget())),
    (committee(), fl.budget.VariableThreshold(budget(), randomize=1.0, seed=0)),
    (fl.strategy.Split(committee(), p=0.2, seed=0), fl.budget.VariableThreshold(budget())),
    (Proportional(committee(), rate=0.3, p_min=0.05, seed=0), budget()),
]
for strategy, budget in setups:
    log = fl.evaluate.prequential(stream, MultiMean(), strategy, budget, {"a": 1.0, "b": 1.0, "c": 1.0})
    print([(row["utility"], row["probability"], row["queried"]) for row in log])
"""


def test_seeded_runs_do_not_depend_on_hash_seed():
    outputs = set()
    for hash_seed in ("1", "2", "3"):
        result = subprocess.run(
            [sys.executable, "-c", REPRO_SCRIPT],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1


# synth


def outputs_of(dataset, n):
    data = list(dataset.take(n))
    return {o: [y[o] for _, y in data] for o in dataset.outputs}


@pytest.mark.parametrize("correlation", [0.0, 0.5, 0.9])
def test_multitarget_correlation_and_variance(correlation):
    ys = outputs_of(fl.synth.MultiTarget(correlation=correlation, noise=0.0, seed=1), 5000)
    for v in ys.values():
        assert statistics.pstdev(v) == pytest.approx(1.0, abs=0.05)
    for a, b in itertools.combinations(ys, 2):
        assert statistics.correlation(ys[a], ys[b]) == pytest.approx(correlation, abs=0.05)


def test_multitarget_noise_lowers_correlation():
    ys = outputs_of(fl.synth.MultiTarget(correlation=0.8, noise=1.0, seed=1), 5000)
    assert statistics.correlation(ys["y0"], ys["y1"]) == pytest.approx(0.4, abs=0.05)


def test_multitarget_drift_only_changes_listed_output_after_position():
    dataset = fl.synth.MultiTarget(n_features=2, noise=0.0, drift={"y1": 1000}, seed=0)
    data = list(dataset.take(1500))
    past, future = data[:1000], data[1000:]

    def nearest_error(o, instances):
        errors = []
        for x, y in instances:
            _, y_past = min(past, key=lambda p: (p[0][0] - x[0]) ** 2 + (p[0][1] - x[1]) ** 2)
            errors.append(abs(y_past[o] - y[o]))
        return statistics.mean(errors)

    assert nearest_error("y0", future) < 0.1
    assert nearest_error("y2", future) < 0.1
    assert nearest_error("y1", future) > 0.5


def test_multitarget_is_reproducible():
    assert list(fl.synth.MultiTarget(seed=3).take(50)) == list(
        fl.synth.MultiTarget(seed=3).take(50)
    )
    dataset = fl.synth.MultiTarget(seed=3)
    assert list(dataset.take(50)) == list(dataset.take(50))
    assert list(fl.synth.MultiTarget(seed=3).take(5)) != list(fl.synth.MultiTarget(seed=4).take(5))


def test_multitarget_works_with_prequential():
    log = fl.evaluate.prequential(
        fl.synth.MultiTarget(seed=0).take(100),
        MeanModel(),
        fl.strategy.QueryAll(),
        fl.budget.Unlimited(),
        {"y0": 1.0, "y1": 1.0, "y2": 1.0},
    )
    assert len(log) == 300


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"n_features": 0}, ValueError),
        ({"n_outputs": 0}, ValueError),
        ({"correlation": 1.5}, ValueError),
        ({"noise": -1.0}, ValueError),
        ({"lengthscale": 0.0}, ValueError),
        ({"noise": {"y0": 0.1}}, KeyError),
        ({"noise": {"y0": 0.1, "y1": 0.1, "y2": 0.1, "y9": 0.1}}, KeyError),
        ({"drift": {"y9": 10}}, KeyError),
        ({"drift": {"y0": -1}}, ValueError),
        ({"n_features": 1, "n_outputs": 150}, ValueError),
    ],
)
def test_multitarget_validation(kwargs, error):
    with pytest.raises(error):
        fl.synth.MultiTarget(**kwargs)


def test_shared_budget_breaks_ties_by_least_spent():
    b = fl.budget.RateBudget(1.0)
    costs = {"a": 1.0, "b": 1.0, "c": 1.0}
    for _ in range(30):
        queried = b.select_one(dict.fromkeys(costs, 1.0), costs)
        b.update({o: costs[o] for o in queried})
    assert b.spent_per_output == {"a": 10.0, "b": 10.0, "c": 10.0}


# Delayed labels


def counting_stream(n):
    """Every output of instance t is t, so the mean model reveals what it has learned."""
    for t in range(n):
        yield {"t": t}, dict.fromkeys(COSTS, float(t))


def test_zero_delay_is_no_delay():
    runs = []
    for delay in (None, 0, {"a": 0, "b": 0, "c": 0}):
        model = MeanModel()
        log = fl.evaluate.prequential(
            make_stream(100),
            model,
            fl.strategy.RandomPerOutput(0.5, seed=1),
            fl.budget.RateBudget(2.0),
            COSTS,
            delay=delay,
        )
        runs.append((log.to_frame(), model.learned))
    for df, learned in runs[1:]:
        assert df.equals(runs[0][0])
        assert learned == runs[0][1]


@pytest.mark.parametrize("d", [1, 3, 10])
def test_constant_delay_hides_labels_for_d_instances(d):
    model = MeanModel()
    log = fl.evaluate.prequential(
        counting_stream(30), model, fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS, delay=d
    )
    df = log.to_frame().query("output == 'a'").set_index("step")
    for n in range(30):
        # Instance n is predicted with the labels of instances 0 to n - d - 1
        known = range(n - d)
        expected = statistics.mean(known) if known else None
        if expected is None:
            assert math.isnan(df.loc[n, "y_pred"])
        else:
            assert df.loc[n, "y_pred"] == pytest.approx(expected)
    # The last d labels were paid for but never learned
    assert len(model.learned) == 30 - d
    assert (df["delay"] == d).all()


def test_per_output_delays():
    model = MeanModel()
    log = fl.evaluate.prequential(
        counting_stream(10),
        model,
        fl.strategy.QueryAll(),
        fl.budget.Unlimited(),
        COSTS,
        delay={"a": 0, "b": 2, "c": math.inf},
    )
    df = log.to_frame()
    assert model.counts == {"a": 10, "b": 8}
    assert df.query("output == 'c'")["y_pred"].isna().all()
    # Labels arriving together are learned together, oldest instance first
    assert model.learned[:3] == [{"a": 0.0}, {"a": 1.0}, {"b": 0.0}]
    assert model.learned[3] == {"a": 2.0}
    assert df.query("queried").groupby("output")["delay"].first().to_dict() == {
        "a": 0,
        "b": 2,
        "c": math.inf,
    }


def test_outputs_with_same_arrival_are_learned_together():
    model = MeanModel()
    fl.evaluate.prequential(
        counting_stream(5), model, fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS, delay=1
    )
    assert model.learned == [dict.fromkeys(COSTS, float(t)) for t in range(4)]


def test_delayed_queries_are_paid_when_made():
    budget = fl.budget.Unlimited()
    log = fl.evaluate.prequential(
        counting_stream(20), MeanModel(), fl.strategy.QueryAll(), budget, COSTS, delay=math.inf
    )
    df = log.to_frame()
    assert budget.spent == 20 * sum(COSTS.values())
    assert df["y_pred"].isna().all()
    assert df.groupby("step")["cum_cost"].first().is_monotonic_increasing


def test_unqueried_outputs_have_no_delay():
    log = fl.evaluate.prequential(
        make_stream(50),
        MeanModel(),
        fl.strategy.RandomPerOutput(0.3, seed=0),
        fl.budget.Unlimited(),
        COSTS,
        delay=5,
    )
    df = log.to_frame()
    assert df[~df["queried"]]["delay"].isna().all()
    assert (df[df["queried"]]["delay"] == 5).all()


def test_strategy_learns_from_prediction_made_at_query_time():
    spy = SpyStrategy()
    log = fl.evaluate.prequential(
        counting_stream(20), MeanModel(), spy, fl.budget.Unlimited(), COSTS, delay=4
    )
    df = log.to_frame()
    at_query = [
        {o: v for o, v in zip(g["output"], g["y_pred"]) if not math.isnan(v)}
        for _, g in df.groupby("step")
    ]
    assert spy.seen == at_query[:16]


def test_delay_function_is_called_per_query_in_cost_order():
    calls = []

    def delay(x, y, output):
        calls.append((x["t"], y[output], output))
        return x["t"] % 3

    fl.evaluate.prequential(
        counting_stream(6),
        MeanModel(),
        fl.strategy.RandomPerOutput(0.5, seed=3),
        fl.budget.Unlimited(),
        COSTS,
        delay=delay,
    )
    assert calls
    for t, value, _ in calls:
        assert value == t
    for t in range(6):
        outputs = [o for s, _, o in calls if s == t]
        assert outputs == [o for o in COSTS if o in outputs]


@pytest.mark.parametrize(
    "delay, error",
    [
        (-1, ValueError),
        (1.5, ValueError),
        (math.nan, ValueError),
        ("3", ValueError),
        ({"a": 0, "b": 0}, KeyError),
        ({"a": 0, "b": 0, "c": 0, "d": 0}, KeyError),
        ({"a": 0, "b": -2, "c": 0}, ValueError),
        (lambda x, y, o: -1, ValueError),
    ],
)
def test_delay_validation(delay, error):
    with pytest.raises(error):
        fl.evaluate.prequential(
            counting_stream(5),
            MeanModel(),
            fl.strategy.QueryAll(),
            fl.budget.Unlimited(),
            COSTS,
            delay=delay,
        )


def test_wrapper_learns_from_given_prediction():
    spy = SpyStrategy()
    model = fl.active.Wrapper(MeanModel(), spy, fl.budget.Unlimited(), COSTS)
    x_old = {"t": 0}
    y_pred_old, _ = model.predict_one(x_old)
    model.learn_one({"t": 1}, {"a": 5.0})
    model.learn_one(x_old, {"a": 1.0}, y_pred=y_pred_old)
    assert spy.seen == [{}, {}]


# Regression tests from the review


@pytest.mark.parametrize("make", [np.int64, np.int32, np.float64])
def test_numpy_delays_are_accepted(make):
    model = MeanModel()
    log = fl.evaluate.prequential(
        counting_stream(10),
        model,
        fl.strategy.QueryAll(),
        fl.budget.Unlimited(),
        COSTS,
        delay=lambda x, y, o: make(2),
    )
    assert len(model.learned) == 8
    assert (log.to_frame()["delay"] == 2).all()


def test_numpy_delays_from_a_generator():
    rng = np.random.default_rng(0)
    log = fl.evaluate.prequential(
        counting_stream(50),
        MeanModel(),
        fl.strategy.QueryAll(),
        fl.budget.Unlimited(),
        COSTS,
        delay=lambda x, y, o: rng.integers(0, 10),
    )
    assert log.to_frame()["delay"].between(0, 9).all()


@pytest.mark.parametrize("delay", [True, np.float64(1.5), np.int64(-1)])
def test_delay_rejects_bools_and_bad_numpy_values(delay):
    with pytest.raises(ValueError):
        fl.evaluate.prequential(
            counting_stream(5),
            MeanModel(),
            fl.strategy.QueryAll(),
            fl.budget.Unlimited(),
            COSTS,
            delay=delay,
        )


def test_curve_is_nan_before_first_prediction():
    log = fl.evaluate.prequential(
        counting_stream(30),
        MeanModel(),
        fl.strategy.QueryAll(),
        fl.budget.Unlimited(),
        COSTS,
        delay={"a": 0, "b": 10, "c": 0},
    )
    curve = fl.metrics.curve(log, AbsError()).query("output == 'b'").set_index("step")["value"]
    # b is first predicted at step 11, once the label of step 0 arrived after step 10
    assert curve.loc[:10].isna().all()
    assert curve.loc[11:].notna().all()


def test_curve_ends_with_last_step():
    log = fl.evaluate.prequential(
        make_stream(25), MeanModel(), fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS
    )
    curve = fl.metrics.curve(log, AbsError(), every=10)
    assert sorted(set(curve["step"])) == [9, 19, 24]
    last = curve[curve["step"] == 24].set_index("output")["value"]
    expected = {o: m.get() for o, m in fl.metrics.per_output(log, AbsError()).items()}
    assert last.to_dict() == pytest.approx(expected)


def test_curve_with_every_dividing_length_has_no_duplicate_last_step():
    log = fl.evaluate.prequential(
        make_stream(20), MeanModel(), fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS
    )
    curve = fl.metrics.curve(log, AbsError(), every=10)
    assert len(curve) == 2 * len(COSTS)


def test_qbc_never_rules_out_an_output_when_members_agree():
    s = fl.strategy.QueryByCommittee(Constant(1.0), n_members=3, min_labels=1, seed=0)
    s.learn_one({}, {"a": 1.0}, {})
    utility = s.utilities_one({}, {}, {"a": 1.0})["a"]
    assert 0 < utility < 1e-6
    assert fl.budget.Unlimited().select_one({"a": utility}, {"a": 1.0}) == {"a"}


class SpyMetric:
    def __init__(self):
        self.seen = []

    def update(self, y_true, y_pred):
        self.seen.append(set(y_true))


def test_metric_ignores_outputs_without_cost():
    def stream(n):
        for t in range(n):
            yield {}, {"a": 1.0, "b": 2.0, "c": 3.0, "extra": 4.0}

    metric = SpyMetric()
    fl.evaluate.prequential(
        stream(5), MeanModel(), fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS, metric=metric
    )
    # Step 0 has no prediction yet, every later step is scored on the costed outputs only
    assert metric.seen == [set(COSTS)] * 4


def test_metric_ignores_extra_outputs_with_river():
    from river import linear_model, metrics, multioutput

    def stream(n):
        for t in range(n):
            x = {"x": t % 10}
            yield x, {"a": x["x"], "b": 2 * x["x"], "extra": 1.0}

    metric = metrics.multioutput.PerOutput(metrics.MAE())
    fl.evaluate.prequential(
        stream(50),
        multioutput.RegressorChain(linear_model.LinearRegression(), order=["a", "b"]),
        fl.strategy.QueryAll(),
        fl.budget.Unlimited(),
        {"a": 1.0, "b": 1.0},
        metric=metric,
    )
    assert set(metric.metrics) == {"a", "b"}


@pytest.mark.parametrize("position", [np.int64(10), [np.int32(10), 20]])
def test_multitarget_accepts_numpy_drift_positions(position):
    dataset = fl.synth.MultiTarget(drift={"y0": position}, seed=0)
    assert dataset.drift["y0"][0] == 10
    assert all(type(p) is int for p in dataset.drift["y0"])


def test_multitarget_rejects_fractional_drift_positions():
    with pytest.raises(TypeError):
        fl.synth.MultiTarget(drift={"y0": [10.5]}, seed=0)


# Randomized querying


class Utilities(fl.base.QueryStrategy):
    """Give fixed utilities, one per instance, and remember what it learned from."""

    def __init__(self, utilities):
        self.utilities = iter(utilities)
        self.learned = []

    def utilities_one(self, x, y_pred, costs):
        u = next(self.utilities)
        return {o: u for o in costs}

    def learn_one(self, x, y, y_pred):
        self.learned.append(y)


class Proportional(fl.strategy.Randomized):
    def probability_one(self, output, utility, recent):
        return utility / statistics.mean(recent)


class Half(fl.strategy.Randomized):
    def probability_one(self, output, utility, recent):
        return 0.5


def test_randomized_samples_with_its_probabilities():
    s = Half(Utilities([2.0] * 20_000), seed=0)
    utilities = [s.sample_one({}, {}, {"a": 1.0}) for _ in range(20_000)]
    assert utilities[0] == ({"a": 2.0}, {"a": 1.0})  # nothing to compare the first one to
    assert all(p == {"a": 0.5} for _, p in utilities[1:])
    assert all(u["a"] in (0.0, 2.0) for u, _ in utilities)
    assert statistics.mean(u["a"] > 0 for u, _ in utilities) == pytest.approx(0.5, abs=0.01)


def test_randomized_rate_rescales_probabilities():
    rng = random.Random(0)
    utilities = [rng.expovariate(1.0) for _ in range(5000)]
    s = Proportional(Utilities(utilities), rate=0.2, seed=0)
    probabilities = [s.sample_one({}, {}, {"a": 1.0})[1]["a"] for _ in utilities]
    assert statistics.mean(probabilities) == pytest.approx(0.2, abs=0.02)
    # The ranking of the utilities is kept
    assert probabilities[utilities.index(max(utilities))] == 1.0


def test_randomized_p_min_and_clipping():
    class Wild(fl.strategy.Randomized):
        def probability_one(self, output, utility, recent):
            return utility - 2

    s = Wild(Utilities([1.0, 1.0, 5.0, 2.5]), p_min=0.1, seed=0)
    probabilities = [s.sample_one({}, {}, {"a": 1.0})[1]["a"] for _ in range(4)]
    assert probabilities == [1.0, 0.1, 1.0, 0.5]


@pytest.mark.parametrize("p_min", [0.0, 0.05, 0.1])
def test_randomized_p_min_does_not_change_the_rate(p_min):
    rng = random.Random(0)
    utilities = [rng.expovariate(1.0) for _ in range(5000)]
    s = Proportional(Utilities(utilities), rate=0.2, p_min=p_min, seed=0)
    probabilities = [s.sample_one({}, {}, {"a": 1.0})[1]["a"] for _ in utilities]
    assert statistics.mean(probabilities) == pytest.approx(0.2, abs=0.02)
    assert min(probabilities) >= p_min


def test_randomized_rejects_invalid_probabilities():
    class Broken(fl.strategy.Randomized):
        def probability_one(self, output, utility, recent):
            return math.nan

    s = Broken(Utilities([1.0, 1.0]), p_min=0.1)
    s.sample_one({}, {}, {"a": 1.0})  # nothing to compare to, so not asked
    with pytest.raises(ValueError, match="finite"):
        s.sample_one({}, {}, {"a": 1.0})


def test_randomized_infinite_utilities_are_always_sampled_and_not_kept():
    s = Proportional(Utilities([math.inf, math.inf, 1.0, 3.0]), seed=0)
    assert s.sample_one({}, {}, {"a": 1.0}) == ({"a": math.inf}, {"a": 1.0})
    assert s.sample_one({}, {}, {"a": 1.0}) == ({"a": math.inf}, {"a": 1.0})
    # The first finite utility has nothing to compare to, so inf was not kept in the window
    assert s.sample_one({}, {}, {"a": 1.0})[1] == {"a": 1.0}
    assert s.sample_one({}, {}, {"a": 1.0})[1] == {"a": 1.0}  # 3 / 1, clipped


def test_randomized_window_is_bounded():
    s = Proportional(Utilities([1.0] * 10 + [2.0]), window=3, seed=0)
    for _ in range(10):
        s.sample_one({}, {}, {"a": 1.0})
    assert list(s._utilities["a"]) == [1.0, 1.0, 1.0]


@pytest.mark.parametrize("utility", [0.0, -1.0, math.nan])
def test_randomized_never_samples_what_its_strategy_vetoes(utility):
    s = Half(Utilities([1.0] + [utility] * 100), p_min=0.2, seed=0)
    s.sample_one({}, {}, {"a": 1.0})
    for _ in range(100):
        assert s.sample_one({}, {}, {"a": 1.0}) == ({"a": 0.0}, {"a": 0.0})
    # Vetoed utilities are not kept either
    assert list(s._utilities["a"]) == [1.0]


def test_randomized_keeps_the_utilities_and_probabilities_of_its_strategy():
    s = Half(fl.strategy.RandomPerOutput(p=0.2, seed=0), seed=0)
    results = [s.sample_one({}, {}, {"a": 1.0}) for _ in range(20_000)]
    # Utilities of random querying are 1, and the first sampled one has nothing to compare to
    assert {u["a"] for u, _ in results} == {0.0, 1.0}
    assert {round(p["a"], 9) for _, p in results[100:]} == {0.0, 0.1}
    assert statistics.mean(u["a"] > 0 for u, _ in results) == pytest.approx(0.1, abs=0.01)


class Partial(fl.base.QueryStrategy):
    """Only score output a, as a strategy that knows nothing about the other outputs might."""

    def utilities_one(self, x, y_pred, costs):
        return {"a": 1.0}


def test_randomized_and_split_do_not_query_outputs_left_out():
    s = Half(Partial(), seed=0)
    for _ in range(50):
        assert s.sample_one({}, {}, {"a": 1.0, "b": 1.0})[0]["b"] == 0.0
    s = fl.strategy.Split(Partial(), p=0.5, seed=0)
    utilities = [s.utilities_one({}, {}, {"a": 1.0, "b": 1.0})["b"] for _ in range(50)]
    assert set(utilities) == {0.0, math.inf}  # only explored


def test_randomized_outputs_are_independent_of_order():
    a = Half(Utilities([1.0] * 50), seed=3)
    b = Half(Utilities([1.0] * 50), seed=3)
    ua = [a.utilities_one({}, {}, {"a": 1.0, "b": 1.0}) for _ in range(50)]
    ub = [b.utilities_one({}, {}, {"b": 1.0, "a": 1.0}) for _ in range(50)]
    assert ua == ub
    assert [u["a"] for u in ua] != [u["b"] for u in ua]


def test_randomized_passes_labels_to_its_strategy():
    base = Utilities([1.0])
    Half(base).learn_one({}, {"a": 1.0}, {"a": 0.0})
    assert base.learned == [{"a": 1.0}]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 0},
        {"rate": 0.0},
        {"rate": 1.5},
        {"p_min": -0.1},
        {"p_min": 2},
        {"rate": 0.1, "p_min": 0.1},
    ],
)
def test_randomized_validation(kwargs):
    with pytest.raises(ValueError):
        Half(fl.strategy.QueryAll(), **kwargs)


def test_split_explores_with_its_probability():
    s = fl.strategy.Split(Utilities([0.5] * 20_000), p=0.1, seed=0)
    utilities = [s.utilities_one({}, {}, {"a": 1.0})["a"] for _ in range(20_000)]
    assert set(utilities) == {0.5, math.inf}
    assert statistics.mean(u == math.inf for u in utilities) == pytest.approx(0.1, abs=0.01)


def test_split_probabilities():
    assert fl.strategy.Split(fl.strategy.QueryAll(), p=0.1).sample_one({}, {}, {"a": 1.0})[1] == {}
    s = fl.strategy.Split(fl.strategy.RandomPerOutput(p=0.5), p=0.2)
    assert s.sample_one({}, {}, {"a": 1.0})[1] == {"a": pytest.approx(0.6)}


def test_split_passes_labels_to_its_strategy():
    base = Utilities([1.0])
    fl.strategy.Split(base).learn_one({}, {"a": 1.0}, {"a": 0.0})
    assert base.learned == [{"a": 1.0}]


def test_split_explored_outputs_do_not_move_the_threshold():
    s = fl.strategy.Split(Utilities([0.5] * 200), p=0.3, seed=0)
    b = fl.budget.VariableThreshold(fl.budget.Unlimited(), step=0.1)
    queried = []
    for _ in range(200):
        u = s.utilities_one({}, {}, {"a": 1.0})
        ask = b.select_one(u, {"a": 1.0})
        b.update({o: 1.0 for o in ask})
        queried.append((u["a"], bool(ask)))
    assert all(q for u, q in queried if u == math.inf)
    # Without exploration, a constant utility is queried about every other time
    exploited = [q for u, q in queried if u != math.inf]
    assert 0.3 < statistics.mean(exploited) < 0.7


def test_probabilities_are_logged():
    log = fl.evaluate.prequential(
        make_stream(20),
        MeanModel(),
        Half(fl.strategy.QueryAll(), seed=0),
        fl.budget.Unlimited(),
        COSTS,
    )
    df = log.to_frame()
    assert (df.query("step > 0")["probability"] == 0.5).all()
    assert (df["queried"] == (df["utility"] > 0)).all()

    log = fl.evaluate.prequential(
        make_stream(5), MeanModel(), fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS
    )
    assert log.to_frame()["probability"].isna().all()


def test_random_per_output_reports_its_probability():
    s = fl.strategy.RandomPerOutput(p=0.3, seed=0)
    assert s.sample_one({}, {}, {"a": 1.0, "b": 1.0})[1] == {"a": 0.3, "b": 0.3}


def test_wrapper_query_one_returns_probabilities():
    model = fl.active.Wrapper(
        MeanModel(), fl.strategy.RandomPerOutput(p=0.3, seed=0), fl.budget.Unlimited(), COSTS
    )
    _, utilities, probabilities, ask = model.query_one({})
    assert probabilities == {o: 0.3 for o in COSTS}
    assert ask == {o for o, u in utilities.items() if u > 0}


def test_variable_threshold_randomize_queries_below_threshold():
    def run(randomize):
        b = fl.budget.VariableThreshold(
            fl.budget.Unlimited(), step=0.01, randomize=randomize, seed=0
        )
        rng = random.Random(0)
        queried = []
        for t in range(3000):
            # Utilities collapse after the first 1000 instances, as when a model is confidently wrong
            u = rng.uniform(1, 2) if t < 1000 else rng.uniform(0.01, 0.02)
            ask = b.select_one({"a": u}, {"a": 1.0})
            b.update({o: 1.0 for o in ask})
            queried.append(bool(ask))
        return queried

    plain, randomized = run(0.0), run(1.0)
    # The plain threshold takes a while to come down, the randomized one keeps querying
    assert sum(plain[1000:1100]) == 0
    assert sum(randomized[1000:1100]) > 10


def test_variable_threshold_randomize_is_reproducible():
    def run(seed):
        b = fl.budget.VariableThreshold(fl.budget.Unlimited(), randomize=1.0, seed=seed)
        out = []
        for t in range(100):
            ask = b.select_one({"a": 1.0 + t % 3, "b": 2.0}, {"a": 1.0, "b": 1.0})
            b.update({o: 1.0 for o in ask})
            out.append(sorted(ask))
        return out

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_variable_threshold_randomize_validation():
    with pytest.raises(ValueError):
        fl.budget.VariableThreshold(fl.budget.Unlimited(), randomize=-1)


# Unsupervised learning


class HookedModel(MeanModel):
    def __init__(self):
        super().__init__()
        self.unsupervised = []

    def learn_unsupervised(self, x):
        self.unsupervised.append(x)


class HookedStrategy(fl.strategy.QueryAll):
    def __init__(self):
        self.unsupervised = []

    def learn_unsupervised(self, x):
        self.unsupervised.append(x)


@pytest.mark.parametrize("learn_unsupervised", [False, True])
def test_learn_unsupervised_hooks(learn_unsupervised):
    model, strategy = HookedModel(), HookedStrategy()
    fl.evaluate.prequential(
        make_stream(30),
        model,
        strategy,
        fl.budget.TotalBudget(10.0),
        COSTS,
        learn_unsupervised=learn_unsupervised,
    )
    expected = [{"i": i} for i in range(30)] if learn_unsupervised else []
    assert model.unsupervised == expected
    assert strategy.unsupervised == expected


def test_learn_unsupervised_without_hooks():
    log = fl.evaluate.prequential(
        make_stream(10),
        MeanModel(),
        fl.strategy.QueryAll(),
        fl.budget.Unlimited(),
        COSTS,
        learn_unsupervised=True,
    )
    assert len(log.to_frame()) == 30


@pytest.mark.parametrize(
    "wrap", [lambda s: fl.strategy.Split(s, p=0.5, seed=0), lambda s: Half(s, seed=0)]
)
def test_wrapping_strategies_forward_learn_unsupervised(wrap):
    inner = HookedStrategy()
    wrap(inner).learn_unsupervised({"i": 0})
    assert inner.unsupervised == [{"i": 0}]


@pytest.mark.parametrize(
    "wrap", [lambda s: fl.strategy.Split(s, p=0.5, seed=0), lambda s: Half(s, seed=0)]
)
def test_wrapping_strategies_accept_strategies_without_learn_unsupervised(wrap):
    class DuckStrategy:
        def sample_one(self, x, y_pred, costs):
            return {o: 1.0 for o in costs}, {}

    wrap(DuckStrategy()).learn_unsupervised({"i": 0})


def test_learn_unsupervised_attribute_that_is_not_a_method():
    model = MeanModel()
    model.learn_unsupervised = True
    wrapper = fl.active.Wrapper(
        model, fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS, learn_unsupervised=True
    )
    x, y = next(make_stream(1))
    wrapper.predict_one(x)
    wrapper.learn_one(x, y)


def test_learn_unsupervised_hooks_once_when_outputs_arrive_separately():
    model, strategy = HookedModel(), HookedStrategy()
    wrapper = fl.active.Wrapper(
        model, strategy, fl.budget.Unlimited(), COSTS, learn_unsupervised=True
    )
    x, y = next(make_stream(1))
    wrapper.predict_one(x)
    for o in COSTS:
        wrapper.learn_one(x, {o: y[o]})
    assert model.unsupervised == [x]
    assert strategy.unsupervised == [x]


def test_learn_unsupervised_warning_check_handles_deep_models():
    class Node:
        def __init__(self, child):
            self.child = child

    model = MeanModel()
    model.history = None
    for _ in range(5000):
        model.history = Node(model.history)
    fl.active.Wrapper(
        model, fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS, learn_unsupervised=True
    )


def test_learn_unsupervised_skips_objects_it_cannot_read():
    class Unreadable:
        __slots__ = ("value",)

        def __getattr__(self, name):
            raise RuntimeError("cannot be read")

    model = MeanModel()
    model.unreadable = Unreadable()
    fl.active.Wrapper(
        model, fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS, learn_unsupervised=True
    )
