"""fishladder: evaluation of online multi-output active learning on data streams."""

from fishladder import active, base, budget, evaluate, metrics, strategy, synth
from fishladder.log import Log
from fishladder.oracle import Oracle

__version__ = "0.1.0"

__all__ = [
    "Log",
    "Oracle",
    "active",
    "base",
    "budget",
    "evaluate",
    "metrics",
    "strategy",
    "synth",
]
