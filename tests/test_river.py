import collections

import pytest

import fishladder as fl

river = pytest.importorskip("river")
from river import datasets, linear_model, metrics, model_selection, multioutput, preprocessing


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


# Learning from unlabelled instances


class SpyScaler(preprocessing.StandardScaler):
    """A scaler that counts the instances it learns from."""

    def __init__(self):
        super().__init__()
        self.n_learned = 0

    def learn_one(self, x):
        self.n_learned += 1
        super().learn_one(x)


class PerOutputModel:
    """One pipeline per output, which unlike `RegressorChain` does not predict when learning."""

    def __init__(self, outputs):
        self.spies = {o: SpyScaler() for o in outputs}
        self.models = {o: spy | linear_model.LinearRegression() for o, spy in self.spies.items()}

    def predict_one(self, x):
        return {o: m.predict_one(x) for o, m in self.models.items()}

    def learn_one(self, x, y):
        for o, y_o in y.items():
            self.models[o].learn_one(x, y_o)


def friedman_stream(n):
    for x, y in datasets.synth.Friedman(seed=0).take(n):
        yield x, {0: y, 1: -y}


COSTS = {0: 1.0, 1: 2.0}


def make_wrapper(model, costs=COSTS):
    return fl.active.Wrapper(
        model, fl.strategy.QueryAll(), fl.budget.Unlimited(), costs, learn_unsupervised=True
    )


needs_per_output = pytest.mark.skipif(
    not hasattr(multioutput, "PerOutputRegressor"), reason="needs River 0.26 or later"
)


def test_learn_unsupervised_sees_every_instance():
    model = PerOutputModel(COSTS)
    log = fl.evaluate.prequential(
        friedman_stream(100),
        model,
        fl.strategy.QueryAll(),
        fl.budget.TotalBudget(20.0),
        COSTS,
        delay=3,
        learn_unsupervised=True,
    )
    assert 0 < log.to_frame()["queried"].sum() < 200
    assert all(spy.n_learned == 100 for spy in model.spies.values())


def test_unsupervised_steps_only_learn_from_labels_by_default():
    model = PerOutputModel(COSTS)
    log = fl.evaluate.prequential(
        friedman_stream(100), model, fl.strategy.QueryAll(), fl.budget.TotalBudget(20.0), COSTS
    )
    queried = log.to_frame().groupby("output")["queried"].sum()
    assert all(spy.n_learned == queried[o] for o, spy in model.spies.items())
    assert all(spy.n_learned < 100 for spy in model.spies.values())


def test_late_label_learns_unsupervised_steps_once():
    model = PerOutputModel(COSTS)
    wrapper = make_wrapper(model)
    (x1, y1), (x2, y2) = friedman_stream(2)
    y_pred, _ = wrapper.predict_one(x1)
    wrapper.predict_one(x2)
    wrapper.learn_one(x1, y1, y_pred=y_pred)
    wrapper.learn_one(x2, y2)
    assert all(spy.n_learned == 2 for spy in model.spies.values())


def test_warm_start_learns_unsupervised_steps():
    model = PerOutputModel([0])
    wrapper = make_wrapper(model, costs={0: 1.0})
    for i in range(50):
        wrapper.learn_one({"x": float(i)}, {0: 2.0 * i})
    spy = model.spies[0]
    assert spy.n_learned == 50
    assert spy.means["x"] == pytest.approx(24.5)


@pytest.mark.parametrize("predicted", [True, False])
def test_outputs_revealed_separately_learn_unsupervised_steps_once(predicted):
    model = PerOutputModel(COSTS)
    wrapper = make_wrapper(model)
    x, y = next(friedman_stream(1))
    if predicted:
        wrapper.predict_one(x)
    for o in COSTS:
        wrapper.learn_one(x, {o: y[o]})
    assert all(spy.n_learned == 1 for spy in model.spies.values())


def test_chain_learns_as_with_river_learn_during_predict():
    from river import compose

    def make_chain():
        return multioutput.RegressorChain(
            SpyScaler() | linear_model.LinearRegression(), order=list(COSTS)
        )

    chain = make_chain()
    with pytest.warns(UserWarning, match="chain"):
        fl.evaluate.prequential(
            friedman_stream(50),
            chain,
            fl.strategy.QueryAll(),
            fl.budget.Unlimited(),
            COSTS,
            learn_unsupervised=True,
        )
    river_chain = make_chain()
    with compose.learn_during_predict():
        for x, y in friedman_stream(50):
            river_chain.predict_one(x)
            river_chain.learn_one(x, y)
    counts = [chain[o]["SpyScaler"].n_learned for o in COSTS]
    assert counts == [river_chain[o]["SpyScaler"].n_learned for o in COSTS]


def test_learn_unsupervised_only_changes_the_model():
    from river.compose import Pipeline

    learn_one = Pipeline.learn_one
    outside = SpyScaler() | linear_model.LinearRegression()
    seen = []

    class Model(PerOutputModel):
        def predict_one(self, x):
            seen.append((Pipeline._LEARN_UNSUPERVISED_DURING_PREDICT, Pipeline.learn_one))
            outside.predict_one(x)
            return super().predict_one(x)

        def learn_one(self, x, y):
            seen.append((Pipeline._LEARN_UNSUPERVISED_DURING_PREDICT, Pipeline.learn_one))
            super().learn_one(x, y)

    model = Model(COSTS)
    fl.evaluate.prequential(
        friedman_stream(20),
        model,
        fl.strategy.QueryAll(),
        fl.budget.Unlimited(),
        COSTS,
        learn_unsupervised=True,
    )
    assert seen and all(flag is False and f is learn_one for flag, f in seen)
    assert outside["SpyScaler"].n_learned == 0
    assert all(spy.n_learned == 20 for spy in model.spies.values())


def test_learn_unsupervised_can_be_turned_on_and_off():
    model = multioutput.RegressorChain(
        SpyScaler() | linear_model.LinearRegression(), order=list(COSTS)
    )
    wrapper = fl.active.Wrapper(model, fl.strategy.QueryAll(), fl.budget.Unlimited(), COSTS)
    with pytest.warns(UserWarning, match="chain"):
        wrapper.learn_unsupervised = True
    for x, y in friedman_stream(5):
        wrapper.predict_one(x)
        wrapper.learn_one(x, y)
    wrapper.learn_unsupervised = False
    flag = "_LEARN_UNSUPERVISED_DURING_PREDICT"
    assert not any(flag in vars(model[o]) for o in COSTS)
    assert flag not in vars(model.model)


def make_selector():
    return multioutput.RegressorChain(
        model_selection.GreedyRegressor(
            [preprocessing.StandardScaler() | linear_model.LinearRegression() for _ in range(2)]
        ),
        order=[0],
    )


class Slotted:
    __slots__ = ("model",)

    def __init__(self, model):
        self.model = model


@pytest.mark.parametrize(
    "make, match",
    [
        (make_selector, "model selector"),
        (lambda: collections.deque([make_selector()]), "model selector"),
        (lambda: Slotted(make_selector()), "model selector"),
        (
            lambda: multioutput.MonteCarloClassifierChain(
                preprocessing.StandardScaler() | linear_model.LogisticRegression()
            ),
            "probabilistic classifier chain",
        ),
    ],
    ids=["selector", "deque", "slots", "probabilistic chain"],
)
def test_warns_about_unsupported_models(make, match):
    with pytest.warns(UserWarning, match=match):
        make_wrapper(make())


@needs_per_output
def test_does_not_warn_about_supported_models(recwarn):
    make_wrapper(
        multioutput.PerOutputRegressor(
            preprocessing.StandardScaler() | linear_model.LinearRegression()
        )
    )
    fl.active.Wrapper(make_selector(), fl.strategy.QueryAll(), fl.budget.Unlimited(), {0: 1.0})
    assert not recwarn.list


@pytest.mark.parametrize(
    "make",
    [
        lambda: make_wrapper(make_selector(), costs={0: 1.0}),
        lambda: next(
            fl.evaluate.iter_prequential(
                friedman_stream(1),
                make_selector(),
                fl.strategy.QueryAll(),
                fl.budget.Unlimited(),
                {0: 1.0},
                learn_unsupervised=True,
            )
        ),
        lambda: fl.evaluate.prequential(
            friedman_stream(1),
            make_selector(),
            fl.strategy.QueryAll(),
            fl.budget.Unlimited(),
            {0: 1.0},
            learn_unsupervised=True,
        ),
    ],
    ids=["Wrapper", "iter_prequential", "prequential"],
)
def test_warning_points_at_the_caller(make):
    with pytest.warns(UserWarning, match="model selector") as record:
        make()
    assert record[0].filename == __file__


@needs_per_output
def test_per_output_regressor_learns_unsupervised_steps_once():
    model = multioutput.PerOutputRegressor(SpyScaler() | linear_model.LinearRegression())
    fl.evaluate.prequential(
        friedman_stream(50),
        model,
        fl.strategy.QueryAll(),
        fl.budget.TotalBudget(20.0),
        COSTS,
        delay=2,
        learn_unsupervised=True,
    )
    # The models of the outputs are made when they first learn, after the first 3 instances
    assert [model[o]["SpyScaler"].n_learned for o in COSTS] == [47, 47]


@needs_per_output
def test_warns_about_pipelines_shared_with_the_strategy(recwarn):
    pipeline = preprocessing.StandardScaler() | linear_model.LinearRegression()
    model = multioutput.PerOutputRegressor(pipeline)
    with pytest.warns(UserWarning, match="strategy holds pipelines of the model"):
        fl.active.Wrapper(
            model,
            fl.strategy.QueryByCommittee(pipeline),
            fl.budget.Unlimited(),
            COSTS,
            learn_unsupervised=True,
        )
    fl.active.Wrapper(
        multioutput.PerOutputRegressor(pipeline.clone()),
        fl.strategy.QueryByCommittee(pipeline),
        fl.budget.Unlimited(),
        COSTS,
        learn_unsupervised=True,
    )
    # The warning above was taken by pytest.warns, so separate pipelines raised none
    assert not recwarn.list
