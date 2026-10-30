# fishladder

[![CI](https://github.com/vitor-costa-garcia/fishladder/actions/workflows/ci.yml/badge.svg)](https://github.com/vitor-costa-garcia/fishladder/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fishladder)](https://pypi.org/project/fishladder/)
[![License](https://img.shields.io/pypi/l/fishladder)](https://github.com/vitor-costa-garcia/fishladder/blob/main/LICENSE)

![Fishladder Banner](https://raw.githubusercontent.com/vitor-costa-garcia/fishladder/main/images/fishladder_banner.jpg)

**An evaluation framework for online multi-output active learning on data streams.**

> fishladder is in alpha: the API may still change between minor versions.

## What is this?

In many streaming applications (soft sensors, process monitoring, environmental
sensing), each instance arrives with several outputs to predict, and obtaining
the true values is expensive. An active learner has to decide, on the fly, which
labels are worth asking for, and for which outputs. Those labels often come back
late, and each output can have its own delay.

Existing stream-learning libraries support active learning only for
single-output classification, and existing active learning libraries mostly
assume a pool of data rather than a stream. fishladder fills the gap by
providing tools to **evaluate** active learning methods in this setting. Like a
fish ladder lets only some fish past a dam, the goal is to study which labels
are worth letting through.

fishladder follows [River](https://riverml.xyz)'s conventions (dictionaries for
`x` and `y`, `predict_one` and `learn_one`, River-style metrics) and works with
River's multi-output models, datasets and metrics out of the box. It
complements River rather than replacing it.

## Features

- **Output-level querying.** Ask for any subset of the outputs of an instance,
  with a cost per output.
- **Query strategies, budgets and models are separate pieces.** Any strategy
  can be combined with any budget and any model.
  - Strategies (`fl.strategy`): `RandomPerOutput`, `QueryAll`,
    `QueryByCommittee` and `ExpectedError`.
  - Budgets (`fl.budget`), measured in cost rather than in number of queries:
    `TotalBudget`, `RateBudget`, `Unlimited`, `PerOutput` to give each output
    its own share, and `VariableThreshold` to only query outputs whose utility
    is high compared to their recent ones, optionally with a randomized
    threshold.
- **Randomized querying**, which makes active learning more robust to drift.
  `fl.strategy.Split` mixes any strategy with random exploration, and
  `fl.strategy.Randomized` turns any strategy's utilities into query
  probabilities, so that new randomized strategies only define how. The
  probabilities are kept in the log.
- **Prequential evaluation** (`fl.evaluate`). Test-then-train, with
  predictions scored on every output, queried or not.
- **Delayed labels.** A delay per output, constant or drawn at random, and
  labels that never arrive. Queries are paid for when they are made, and
  labels are learned when they arrive.
- **A tidy experiment log** (`fl.Log`), with one row per instance and output,
  from which per-output metrics and learning curves indexed by cost are
  computed (`fl.metrics`).
- **A synthetic multi-target stream generator** (`fl.synth.MultiTarget`), with
  control over the correlation between outputs, their noise and smoothness, and
  when each one drifts.
- **A River-style wrapper** (`fl.active.Wrapper`) to add output-level active
  learning to an existing River application.

## Installation

```bash
pip install "fishladder[river]"
```

The `river` extra installs River, which the examples use for models and
metrics. fishladder itself only depends on NumPy and pandas. Add the `plots`
extra for the notebooks.

## Quickstart

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

The log can be turned into a DataFrame with `log.to_frame()`, and
`fl.metrics.curve` gives learning curves along the stream, against either the
number of instances or the cost spent.

To add active learning to an existing River loop instead:

```python
model = fl.active.Wrapper(model, strategy, budget, costs)

for x, y in dataset:
    y_pred, ask = model.predict_one(x)  # the outputs to ask for, paid for here
    if ask:
        model.learn_one(x, {o: y[o] for o in ask})
```

## Notebooks

The [notebooks](https://github.com/vitor-costa-garcia/fishladder/tree/main/notebooks) show fishladder in use with River: a quickstart,
the River wrapper, the query strategies, synthetic streams with drift, and
delayed labels.

## Contributing and feedback

Ideas, use cases and criticism are welcome through
[GitHub issues](https://github.com/vitor-costa-garcia/fishladder/issues).

## License

BSD 3-Clause. See [LICENSE](https://github.com/vitor-costa-garcia/fishladder/blob/main/LICENSE).
