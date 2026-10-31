# Notebooks

Examples of fishladder in use, with [River](https://riverml.xyz) for the data, models and metrics.

| Notebook | What it shows |
|---|---|
| [01_quickstart.ipynb](01_quickstart.ipynb) | Per-output costs, query strategies and budgets, prequential evaluation, the experiment log, learning curves indexed by cost, and learning from unlabelled instances with `learn_unsupervised` |
| [02_river_wrapper.ipynb](02_river_wrapper.ipynb) | Adding output-level active learning to an existing River application with `fl.active.Wrapper` |
| [03_query_strategies.ipynb](03_query_strategies.ipynb) | Query-by-committee and expected-error strategies, why they need `fl.budget.VariableThreshold`, and when they beat random querying (and when they don't) |
| [04_synthetic_streams.ipynb](04_synthetic_streams.ipynb) | Generating streams with `fl.synth.MultiTarget`, and what shared and per-output budgets do when a single output drifts |
| [05_delayed_labels.ipynb](05_delayed_labels.ipynb) | Delays per output with `delay`, how uncertainty strategies spend a shared budget while waiting for labels, and how delays slow down recovery from drift |
| [06_randomized_querying.ipynb](06_randomized_querying.ipynb) | Randomizing query strategies with a randomized threshold, `fl.strategy.Split` and `fl.strategy.Randomized`, on a local drift that moves between two points with a low budget, and the query probabilities in the log |

To run them, install fishladder from a clone of this repository with the River and plotting
extras:

```bash
pip install -e ".[river,plots]" jupyter
jupyter lab notebooks/
```
