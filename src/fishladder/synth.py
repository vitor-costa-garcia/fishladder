"""Synthetic data streams."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Mapping

import numpy as np

__all__ = ["MultiTarget"]


class MultiTarget:
    """A multi-target regression stream with correlated outputs and per-output drift.

    The features are drawn uniformly from the unit hypercube. Each output is the sum of a function
    shared by all outputs, a function of its own, and Gaussian noise:

    $$y_o = \\sqrt{\\rho} \\, g(x) + \\sqrt{1 - \\rho} \\, g_o(x) + \\sigma_o \\epsilon$$

    where $\\rho$ is `correlation` and $\\sigma_o$ is the noise of output $o$. The functions are
    smooth random functions, made of random cosines, with zero mean and unit variance. They are
    uncorrelated with each other, so every output has a signal of unit variance, and two outputs
    with noise $\\sigma$ have a correlation of $\\rho / (1 + \\sigma^2)$.

    When an output drifts, its own function is replaced by a new one, abruptly. The shared function
    does not change, so the outputs that do not drift are unaffected.

    There are only so many smooth functions that are uncorrelated with each other. With a large
    lengthscale and many outputs and drifts, some functions are thus less smooth than requested.

    Parameters
    ----------
    n_features
        The number of features.
    n_outputs
        The number of outputs, which are named `y0`, `y1`, etc.
    correlation
        The share of each output's signal that comes from the shared function, between 0
        (uncorrelated outputs) and 1 (identical outputs, up to noise).
    noise
        The standard deviation of the noise, relative to that of the signal. Either a single value
        or one per output.
    lengthscale
        How smooth the functions are. Large values give almost linear functions, small values
        give functions that change quickly over the features and are harder to learn. Either a
        single value or one per output, which applies to the output's own function. The shared
        function mixes all of them.
    drift
        The instances at which each output drifts, counted from zero, as one position or several.
        Outputs that are not listed never drift.
    seed
        Random seed for reproducibility.

    Examples
    --------

    >>> import fishladder as fl

    >>> dataset = fl.synth.MultiTarget(
    ...     n_features=3,
    ...     n_outputs=2,
    ...     correlation=0.5,
    ...     noise={"y0": 0.1, "y1": 1.0},
    ...     drift={"y1": 1000},
    ...     seed=42,
    ... )

    >>> for x, y in dataset.take(2):
    ...     print({i: round(v, 2) for i, v in x.items()}, {o: round(v, 2) for o, v in y.items()})
    {0: 0.17, 1: 0.98, 2: 0.06} {'y0': 1.1, 'y1': -1.2}
    {0: 0.2, 1: 0.39, 2: 0.72} {'y0': -0.79, 'y1': 1.96}

    """

    n_terms = 100
    """The number of random cosines per lengthscale."""

    n_samples = 5000
    """The number of points used to make the functions uncorrelated with unit variance."""

    def __init__(
        self,
        n_features: int = 10,
        n_outputs: int = 3,
        correlation: float = 0.5,
        noise: float | Mapping[str, float] = 0.1,
        lengthscale: float | Mapping[str, float] = 1.0,
        drift: Mapping[str, int | Iterable[int]] | None = None,
        seed: int | None = None,
    ):
        if n_features < 1:
            raise ValueError(f"n_features must be at least 1, got {n_features}")
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be at least 1, got {n_outputs}")
        if not 0 <= correlation <= 1:
            raise ValueError(f"correlation must be between 0 and 1, got {correlation}")
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.outputs = [f"y{i}" for i in range(n_outputs)]
        self.correlation = correlation
        self.noise = self._per_output("noise", noise)
        self.lengthscale = self._per_output("lengthscale", lengthscale)
        for o, s in self.noise.items():
            if s < 0:
                raise ValueError(f"noise of output {o!r} must be non-negative, got {s}")
        for o, ls in self.lengthscale.items():
            if ls <= 0:
                raise ValueError(f"lengthscale of output {o!r} must be positive, got {ls}")
        self.drift: dict[str, list[int]] = {o: [] for o in self.outputs}
        for o, positions in (drift or {}).items():
            self._check_output("drift", o)
            positions = [positions] if isinstance(positions, int) else list(positions)
            if any(p < 0 for p in positions):
                raise ValueError(f"drift positions of output {o!r} must be non-negative")
            self.drift[o] = sorted(positions)
        self.seed = seed
        self._draw_functions()

    def _check_output(self, param: str, o: str) -> None:
        if o not in self.outputs:
            raise KeyError(f"{param} is given for {o!r}, which is not one of {self.outputs}")

    def _per_output(self, param: str, value: float | Mapping[str, float]) -> dict[str, float]:
        if not isinstance(value, Mapping):
            return dict.fromkeys(self.outputs, value)
        for o in value:
            self._check_output(param, o)
        missing = [o for o in self.outputs if o not in value]
        if missing:
            raise KeyError(f"{param} is missing for {missing}")
        return {o: value[o] for o in self.outputs}

    def _rng(self, *key: int) -> np.random.Generator:
        # Each part of the stream has its own generator, so that, for instance, adding a drift to
        # an output does not change the features
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng([self.seed, *key])

    def _draw_functions(self) -> None:
        """Draw the functions, once, so that iterating twice gives the same concepts.

        Every function is a weighted sum of the same random cosines, so it is described by its
        weights. One set of cosines is drawn per lengthscale.

        """
        lengthscales = sorted(set(self.lengthscale.values()))
        n = self.n_terms
        blocks = {ls: slice(j * n, (j + 1) * n) for j, ls in enumerate(lengthscales)}
        self._frequencies = np.vstack(
            [
                self._rng(0, j).normal(0, 1 / ls, size=(n, self.n_features))
                for j, ls in enumerate(lengthscales)
            ]
        )
        self._phases = np.concatenate(
            [self._rng(1, j).uniform(0, 2 * np.pi, size=n) for j in range(len(lengthscales))]
        )

        # The shared function, then the functions of each output: its initial one, and one more
        # for each drift. Each output's functions only use the cosines of its lengthscale.
        weights = [self._rng(2).normal(size=len(self._phases))]
        keys: list[tuple[str, int]] = [("", 0)]
        for k in range(max(len(p) for p in self.drift.values()) + 1):
            for i, o in enumerate(self.outputs):
                if k > len(self.drift[o]):
                    continue
                w = np.zeros(len(self._phases))
                w[blocks[self.lengthscale[o]]] = self._rng(3, i, k).normal(size=n)
                weights.append(w)
                keys.append((o, k))

        # Make the functions uncorrelated with unit variance over the features. The symmetric
        # (Löwdin) orthogonalization gives the uncorrelated functions closest to the ones drawn,
        # treating them all alike, unlike Gram-Schmidt which distorts the last ones the most.
        sample = self._rng(4).uniform(size=(self.n_samples, self.n_features))
        cosines = self._cosines(sample)
        self._offset = cosines.mean(axis=0)
        covariance = np.cov(cosines, rowvar=False)
        w = np.array(weights)
        w /= np.sqrt(np.einsum("fd,de,fe->f", w, covariance, w))[:, None]
        eigenvalues, eigenvectors = np.linalg.eigh(w @ covariance @ w.T)
        if eigenvalues.min() < 1e-6 * eigenvalues.max():
            raise ValueError(
                "there are too many outputs and drifts for the random cosines to give "
                "uncorrelated functions, increase MultiTarget.n_terms"
            )
        basis = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.T @ w

        self._shared = basis[0]
        self._own: dict[str, list[np.ndarray]] = {o: [] for o in self.outputs}
        for (o, _), w in zip(keys[1:], basis[1:], strict=True):
            self._own[o].append(w)

    def _cosines(self, x: np.ndarray) -> np.ndarray:
        return np.cos(x @ self._frequencies.T + self._phases)

    def __iter__(self) -> Iterator[tuple[dict[int, float], dict[str, float]]]:
        rng = self._rng(5)
        shared_weight = np.sqrt(self.correlation)
        own_weight = np.sqrt(1 - self.correlation)
        for t in itertools.count():
            x = rng.uniform(size=self.n_features)
            cosines = self._cosines(x) - self._offset
            shared = cosines @ self._shared
            y = {}
            for o in self.outputs:
                own = cosines @ self._own[o][sum(p <= t for p in self.drift[o])]
                noise = self.noise[o] * rng.normal()
                y[o] = float(shared_weight * shared + own_weight * own + noise)
            yield dict(enumerate(x.tolist())), y

    def take(self, k: int) -> Iterator[tuple[dict[int, float], dict[str, float]]]:
        """Iterate over the first `k` instances."""
        return itertools.islice(self, k)
