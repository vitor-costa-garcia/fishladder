"""The experiment log."""

from __future__ import annotations

import typing
from collections.abc import Iterable, Mapping

from fishladder.base import Output

if typing.TYPE_CHECKING:
    import pandas as pd

__all__ = ["Log"]

COLUMNS = (
    "step",
    "output",
    "y_true",
    "y_pred",
    "utility",
    "queried",
    "cost",
    "cum_cost",
    "delay",
)


class Log:
    """A tidy record of an experiment, with one row per instance and output.

    The log is the single source of truth of an experiment: metrics and plots are computed from
    it, so they can be recomputed without rerunning the experiment.

    Columns
    -------
    step
        The index of the instance in the stream, starting at 0.
    output
        The name of the output.
    y_true
        The true value of the output, or `None` if the stream did not provide it.
    y_pred
        The prediction made before learning from the instance, or `None` if the model could not
        predict this output yet.
    utility
        The utility given by the query strategy.
    queried
        Whether the output was queried.
    cost
        The cost paid for this output.
    cum_cost
        The total cost spent up to and including this instance.
    delay
        For queried outputs, the number of instances after which the label was learned: 0 if it
        was learned right after this instance, `inf` if it never arrives. `None` for outputs that
        were not queried. A label with a finite delay may still not have arrived when the stream
        ended, which happens when `step + delay` is beyond the last step.

    """

    def __init__(self) -> None:
        self._rows: list[tuple] = []

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> typing.Iterator[dict[str, typing.Any]]:
        for row in self._rows:
            yield dict(zip(COLUMNS, row))

    def append_one(
        self,
        step: int,
        y_true: Mapping[Output, typing.Any],
        y_pred: Mapping[Output, typing.Any],
        utilities: Mapping[Output, float],
        queried: Iterable[Output],
        costs: Mapping[Output, float],
        cum_cost: float,
        delays: Mapping[Output, float] | None = None,
    ) -> None:
        """Record one instance. Every output in `costs` gets a row.

        `delays` gives the delay of each queried output, which defaults to 0.

        """
        queried = set(queried)
        delays = delays or {}
        for o in costs:
            q = o in queried
            self._rows.append(
                (
                    step,
                    o,
                    y_true.get(o),
                    y_pred.get(o),
                    utilities.get(o, 0.0),
                    q,
                    costs[o] if q else 0.0,
                    cum_cost,
                    delays.get(o, 0) if q else None,
                )
            )

    def to_frame(self) -> pd.DataFrame:
        """Return the log as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(self._rows, columns=list(COLUMNS))

    def to_csv(self, path: str, **kwargs: typing.Any) -> None:
        """Write the log to a CSV file. Extra arguments go to `pandas.DataFrame.to_csv`."""
        self.to_frame().to_csv(path, index=False, **kwargs)
