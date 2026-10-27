# fishladder

[![CI](https://github.com/vitor-costa-garcia/fishladder/actions/workflows/ci.yml/badge.svg)](https://github.com/vitor-costa-garcia/fishladder/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fishladder)](https://pypi.org/project/fishladder/)
[![License](https://img.shields.io/pypi/l/fishladder)](https://github.com/vitor-costa-garcia/fishladder/blob/main/LICENSE)

**An evaluation framework for online multi-output active learning on data streams.**

> ⚠️ **Early development.** This release (`0.0.1`) mainly reserves the package name.
> There is no usable API yet, and everything described below is planned work that
> may change.

## What is this?

In many streaming applications (soft sensors, process monitoring, environmental
sensing), each instance arrives with several outputs to predict, and obtaining
the true values is expensive. An active learner has to decide, on the fly, which
labels are worth asking for, and for which outputs.

Existing stream-learning libraries support active learning only for
single-output classification, and existing active learning libraries mostly
assume a pool of data rather than a stream. fishladder aims to fill the gap by
providing tools to **evaluate** active learning methods in this setting. Like a
fish ladder lets only some fish past a dam, the goal is to study which labels
are worth letting through.

## Planned scope

- **Multi-output streams:** multi-target regression first, multi-label
  classification afterwards. Single-output streams as the special case of one
  output.
- **Output-level querying:** oracles that can reveal a subset of the outputs of
  an instance, with per-output costs.
- **Budget management** measured in cost, not just number of queries.
- **Prequential, cost-indexed evaluation** with per-output metrics.
- **A tidy experiment log**, from which metrics and plots are computed.
- **Baseline query strategies**, plus a synthetic multi-output stream generator.
- Compatibility with existing streaming libraries (such as
  [River](https://riverml.xyz)) rather than a replacement for them.

None of this is implemented in `0.0.1`.

## Installation

```bash
pip install fishladder
```

This currently installs an empty placeholder package.

## Contributing and feedback

The project scope is still being defined. Ideas, use cases and criticism are
welcome through [GitHub issues](https://github.com/vitor-costa-garcia/fishladder/issues).

## License

BSD 3-Clause. See [LICENSE](https://github.com/vitor-costa-garcia/fishladder/blob/main/LICENSE).
