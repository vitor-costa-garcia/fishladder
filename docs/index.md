# fishladder

**An evaluation framework for online multi-output active learning on data streams.**

```{note}
fishladder is in alpha: the API may still change between minor versions.
```

In many streaming applications (soft sensors, process monitoring, environmental sensing), each
instance arrives with several outputs to predict, and obtaining the true values is expensive. An
active learner has to decide, on the fly, which labels are worth asking for, and for which
outputs. Those labels often come back late, and each output can have its own delay.

fishladder provides tools to **evaluate** active learning methods in this setting. It follows
[River](https://riverml.xyz)'s conventions and works with River's multi-output models, datasets
and metrics out of the box.

```{toctree}
:maxdepth: 2

installation
quickstart
api/index
```
