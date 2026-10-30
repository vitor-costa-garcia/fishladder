# Notebooks

Examples of fishladder in use, with [River](https://riverml.xyz) for the data, models and metrics.

| Notebook | What it shows |
|---|---|
| [01_quickstart.ipynb](01_quickstart.ipynb) | Per-output costs, query strategies and budgets, prequential evaluation, the experiment log, and learning curves indexed by cost |
| [02_river_wrapper.ipynb](02_river_wrapper.ipynb) | Adding output-level active learning to an existing River application with `fl.active.Wrapper` |

To run them, install fishladder from a clone of this repository with the River and plotting
extras:

```bash
pip install -e ".[river,plots]" jupyter
jupyter lab notebooks/
```
