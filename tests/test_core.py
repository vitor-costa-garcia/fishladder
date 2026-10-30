import math
import os
import random
import subprocess
import sys

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
        "queried",
        "cost",
        "cum_cost",
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
    [fl.strategy.QueryByCommittee(Constant(), seed=0), fl.strategy.ExpectedError(Constant())],
)
def test_unseen_outputs_get_infinite_utility(strategy):
    strategy.learn_one({}, {"a": 1.0}, {"a": 0.0})
    utilities = strategy.utilities_one({}, {"a": 0.0}, COSTS)
    assert utilities["a"] < math.inf
    assert utilities["b"] == utilities["c"] == math.inf


def test_qbc_utility_is_committee_spread():
    s = fl.strategy.QueryByCommittee(Constant(), n_members=4, seed=0)
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


def test_qbc_rejects_small_committee():
    with pytest.raises(ValueError):
        fl.strategy.QueryByCommittee(Constant(), n_members=1)


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
log = fl.evaluate.prequential(
    stream,
    MultiMean(),
    fl.strategy.QueryByCommittee(Mean(), seed=0),
    fl.budget.VariableThreshold(fl.budget.PerOutput(fl.budget.RateBudget(1.0))),
    {"a": 1.0, "b": 1.0, "c": 1.0},
)
print([row["utility"] for row in log])
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
