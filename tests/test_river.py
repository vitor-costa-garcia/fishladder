import pytest

import fishladder as fl

river = pytest.importorskip("river")
from river import datasets, linear_model, metrics, multioutput, preprocessing


def make_model():
    return multioutput.RegressorChain(
        preprocessing.StandardScaler() | linear_model.LinearRegression(), order=[0, 1, 2]
    )


def test_river_dataset_and_model():
    dataset = datasets.synth.Friedman(seed=42).take(300)
    stream = ((x, {0: y, 1: 2 * y, 2: -y}) for x, y in dataset)
    metric = metrics.multioutput.PerOutput(metrics.MAE())

    log = fl.evaluate.prequential(
        stream,
        make_model(),
        fl.strategy.RandomPerOutput(0.5, seed=0),
        fl.budget.RateBudget(1.0),
        costs={0: 1.0, 1: 1.0, 2: 1.0},
        metric=metric,
    )

    df = log.to_frame()
    assert len(df) == 900
    assert 0 < df["queried"].sum() < 900
    assert set(metric.get()) == {0, 1, 2}

    recomputed = fl.metrics.per_output(log, metrics.MAE())
    for o in (0, 1, 2):
        assert recomputed[o].get() == pytest.approx(metric.get()[o].get())


def test_wrapper_in_river_loop():
    model = fl.active.Wrapper(
        make_model(), fl.strategy.QueryAll(), fl.budget.TotalBudget(50.0), {0: 1.0, 1: 1.0, 2: 1.0}
    )
    for x, y in datasets.synth.Friedman(seed=0).take(100):
        y = {0: y, 1: y, 2: y}
        _, ask = model.predict_one(x)
        if ask:
            model.learn_one(x, {o: y[o] for o in ask})
    assert model.budget.spent <= 50.0


def test_rrmse_matches_river_r2():
    import math

    dataset = datasets.synth.Friedman(seed=0).take(200)
    model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    rrmse, r2 = fl.metrics.RRMSE(), metrics.R2()
    for x, y in dataset:
        y_pred = model.predict_one(x)
        rrmse.update(y, y_pred)
        r2.update(y, y_pred)
        model.learn_one(x, y)
    assert rrmse.get() == pytest.approx(math.sqrt(1 - r2.get()))
