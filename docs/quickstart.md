# Quickstart

```python
from river import linear_model, metrics, multioutput, preprocessing

import fishladder as fl

costs = {"y0": 1.0, "y1": 2.0, "y2": 5.0}

log = fl.evaluate.prequential(
    dataset=fl.synth.MultiTarget(n_features=5, seed=42).take(2000),
    model=multioutput.RegressorChain(
        preprocessing.StandardScaler() | linear_model.LinearRegression(),
        order=list(costs),
    ),
    strategy=fl.strategy.RandomPerOutput(p=0.5, seed=42),
    budget=fl.budget.PerOutput(fl.budget.RateBudget(1.5)),
    costs=costs,
    delay={"y0": 0, "y1": 0, "y2": 100},  # y2's labels arrive 100 instances later
)

fl.metrics.per_output(log, metrics.MAE())
```

```
{'y0': MAE: 0.704636, 'y1': MAE: 0.526393, 'y2': MAE: 0.479729}
```

The log can be turned into a DataFrame with `log.to_frame()`, and `fl.metrics.curve` gives
learning curves along the stream, against either the number of instances or the cost spent.

To add active learning to an existing River loop instead:

```python
model = fl.active.Wrapper(model, strategy, budget, costs)

for x, y in dataset:
    y_pred, ask = model.predict_one(x)  # the outputs to ask for, paid for here
    if ask:
        model.learn_one(x, {o: y[o] for o in ask})
```

## Learning from unlabelled instances

By default, unsupervised steps such as a scaler only learn from the instances whose labels were
queried. With `learn_unsupervised=True`, the River pipelines of the model learn them from every
instance, as with River's `learn_during_predict`, but for this model only:

```python
log = fl.evaluate.prequential(
    dataset=fl.synth.MultiTarget(n_features=5, seed=42).take(2000),
    model=multioutput.PerOutputRegressor(
        preprocessing.StandardScaler() | linear_model.LinearRegression()
    ),
    strategy=fl.strategy.RandomPerOutput(p=0.5, seed=42),
    budget=fl.budget.PerOutput(fl.budget.RateBudget(0.1)),
    costs=costs,
    learn_unsupervised=True,
)
```

Models that also predict when they learn, such as River chains, learn from some instances more
than once. See `fishladder.active.Wrapper` for which models learn from each instance once.

## Plots

`fl.plot` draws plots from logs with matplotlib, which comes with the `plots` extra. A run is a
log, or a list of logs of the same setup, such as different seeds. For a list, the plots draw the
mean, and shade the range between the lowest and highest values. This range is not a
confidence interval. Pass a dictionary to compare runs:

```python
from river import utils


def run(budget, seed):
    return fl.evaluate.prequential(
        dataset=fl.synth.MultiTarget(n_features=5, seed=seed).take(2000),
        model=multioutput.RegressorChain(
            preprocessing.StandardScaler() | linear_model.LinearRegression(),
            order=list(costs),
        ),
        strategy=fl.strategy.RandomPerOutput(p=0.5, seed=seed),
        budget=budget,
        costs=costs,
    )


runs = {
    "shared": [run(fl.budget.RateBudget(1.5), seed) for seed in range(3)],
    "per output": [run(fl.budget.PerOutput(fl.budget.RateBudget(1.5)), seed) for seed in range(3)],
}

# Learning curves, one panel per output, against the number of instances or the cost spent.
# Against cost, `every` is in cost, so all runs have points at the same costs.
fl.plot.curve(runs, utils.Rolling(metrics.MAE, window_size=200), x="cost", every=20)

# How often each output is queried
fl.plot.queries(runs, window=200)
```

`fl.plot.budget` compares setups across budgets. Give it a list of runs per setup, one per
budget. Each run is drawn at the cost it really spent, which is read from the log. Pass
`budgets=rates` to draw it at its budget instead:

```python
rates = [0.5, 1.0, 2.0, 4.0]
fl.plot.budget(
    {
        "shared": [[run(fl.budget.RateBudget(r), seed) for seed in range(3)] for r in rates],
        "per output": [
            [run(fl.budget.PerOutput(fl.budget.RateBudget(r)), seed) for seed in range(3)]
            for r in rates
        ],
    },
    metrics.MAE(),
)
```

Each function returns the axes it drew on, so you can change the plot afterwards, for example
with `ax.set(yscale="log")`.
