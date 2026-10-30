"""Oracles, which reveal the true value of outputs at a cost."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from fishladder.base import Output

__all__ = ["Oracle"]


class Oracle:
    """An oracle simulated from the true outputs of a stream.

    Each output has its own cost. Querying an output is charged even when the stream has no true
    value for it, since the cost is paid for asking.

    Parameters
    ----------
    costs
        The cost of querying each output.

    Examples
    --------

    >>> from fishladder import Oracle

    >>> oracle = Oracle(costs={"a": 1.0, "b": 5.0})
    >>> oracle.reveal({"a": 0.3, "b": 1.2}, outputs={"b"})
    ({'b': 1.2}, 5.0)

    """

    def __init__(self, costs: Mapping[Output, float]):
        for o, c in costs.items():
            if c < 0:
                raise ValueError(f"cost of output {o!r} must be non-negative, got {c}")
        self.costs = dict(costs)

    def cost(self, outputs: Iterable[Output]) -> float:
        """The cost of querying `outputs`."""
        return sum(self.costs[o] for o in outputs)

    def reveal(self, y: Mapping[Output, object], outputs: Iterable[Output]) -> tuple[dict, float]:
        """Reveal the true value of some outputs.

        Parameters
        ----------
        y
            The true outputs of the instance.
        outputs
            The outputs to reveal.

        Returns
        -------
        The revealed outputs, and the cost of revealing them.

        """
        outputs = list(outputs)
        return {o: y[o] for o in outputs if o in y}, self.cost(outputs)
