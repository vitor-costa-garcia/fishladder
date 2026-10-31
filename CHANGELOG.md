# CHANGELOG



## Unreleased

### Feature

* Add `learn_unsupervised` to `Wrapper`, `iter_prequential` and `prequential`, so that the
  unsupervised steps of the model's River pipelines, such as a scaler, learn from every instance,
  queried or not. This is River's `learn_during_predict` for the model only, and a warning is
  raised for River chains and model selectors, which learn from some instances more than once.
  Models and strategies can define a `learn_unsupervised(x)` method to do the same.

* Add RRMSE, averaged metrics and AUC against cost ([`04f6d86`](https://github.com/vitor-costa-garcia/fishladder/commit/04f6d86edc51dac561b70e4797840e6a446bd76d))

### Build

* Require River 0.21 or later for the `river` extra

### Documentation

* Show `learn_unsupervised` in the quickstart notebook and the quickstart page

* Say that the notebooks need River 0.26 or later

* Add Sphinx documentation with an API reference

* Add Zenodo DOI ([`92348ca`](https://github.com/vitor-costa-garcia/fishladder/commit/92348ca7ebcd92b4b25f4428d48ce963c747d13c))

## v0.1.0 (2026-10-30)

### Build

* Initialize pyproject.toml for fishladder project ([`521c8ad`](https://github.com/vitor-costa-garcia/fishladder/commit/521c8ad6f16d658601787b5cbd6157dab81262d9))

* Add pytest-cov to development dependencies ([`ccfa07a`](https://github.com/vitor-costa-garcia/fishladder/commit/ccfa07a55f38d44189cc8f568ec831b37a77ae39))

* Require Python 3.11 ([`f51a1b2`](https://github.com/vitor-costa-garcia/fishladder/commit/f51a1b2be8de5ac352de589a5c4d217e4aaab376))

* Release 0.1.0 ([`b35d015`](https://github.com/vitor-costa-garcia/fishladder/commit/b35d01534ceb80ddbf19dae7079bf4b9263d1834))

### Ci

* Add CI workflow for linting, testing, and building ([`a41615e`](https://github.com/vitor-costa-garcia/fishladder/commit/a41615e654df21ccddeec87c8aa4a8afe376819f))

* Add GitHub Actions workflow for publishing to PyPI ([`695afcc`](https://github.com/vitor-costa-garcia/fishladder/commit/695afcc0f093d20cdbedca65685632dad9b0397a))

### Documentation

* Enhance README with project overview and installation ([`1d83a1e`](https://github.com/vitor-costa-garcia/fishladder/commit/1d83a1ec1c6a7583db831f356432c4f00cf590dc))

* Add banner image to README ([`d58a310`](https://github.com/vitor-costa-garcia/fishladder/commit/d58a310b5d1128bbd0e16dbdd4324fc3fba95768))

* Add files via upload ([`6f3c0e1`](https://github.com/vitor-costa-garcia/fishladder/commit/6f3c0e162950534e9bb2ddf03db836b075798a77))

* Add example notebooks showing River integration ([`aa3eac7`](https://github.com/vitor-costa-garcia/fishladder/commit/aa3eac726a0e0aacad845d1af97de33fb670809b))

* Add notebook comparing uncertainty-based query strategies ([`f128c9c`](https://github.com/vitor-costa-garcia/fishladder/commit/f128c9c63e5bcf3a391ecccfc07c6a9d318b7135))

* Add notebook on synthetic streams and per-output drift ([`65c31d1`](https://github.com/vitor-costa-garcia/fishladder/commit/65c31d17444515c9bf6627606ee3bc537d5c51d1))

* Add notebook on delayed labels ([`a3c62d6`](https://github.com/vitor-costa-garcia/fishladder/commit/a3c62d603e7fbd2324f4375ec833f84bb67e7869))

* Re-run the quickstart notebook for the probability column ([`8a25dd3`](https://github.com/vitor-costa-garcia/fishladder/commit/8a25dd3ae6d8a168dfcddf31355f716e0c9c9807))

* Add notebook on randomized querying ([`f71eecc`](https://github.com/vitor-costa-garcia/fishladder/commit/f71eecce1803824de37ad14dda8e392652064526))

* Add citation metadata ([`0fe87c2`](https://github.com/vitor-costa-garcia/fishladder/commit/0fe87c2688acc0187618b6b4db3bbe0d6d4df85b))

### Feature

* Add core API: strategies, budgets, oracle, prequential evaluation ([`25f8a46`](https://github.com/vitor-costa-garcia/fishladder/commit/25f8a468c9486130fd13731fa42fad86a5141c60))

* Add PerOutput budget manager to prevent starving expensive outputs ([`2c07c5a`](https://github.com/vitor-costa-garcia/fishladder/commit/2c07c5aa72a319eda6de739125074ae1f736ef32))

* Add query-by-committee and expected-error strategies ([`f3786aa`](https://github.com/vitor-costa-garcia/fishladder/commit/f3786aa44912ce1606ed280a76cba29c55dfa12e))

* Add synthetic multi-target stream generator ([`b9ff23c`](https://github.com/vitor-costa-garcia/fishladder/commit/b9ff23c175f97783be83e9640e95948fbbaca087))

* Support delayed labels in prequential evaluation ([`085ba7d`](https://github.com/vitor-costa-garcia/fishladder/commit/085ba7d23478f8a3cc832bbfa9d7cb70ab30fd13))

* Support randomized querying ([`3ec2196`](https://github.com/vitor-costa-garcia/fishladder/commit/3ec21964901e5f5dcd06396fe3e7daccdfe48092))

### Fix

* Make seeded runs independent of hash randomization ([`77a1072`](https://github.com/vitor-costa-garcia/fishladder/commit/77a1072f11975fbd5442673f1e211b2ff8ea3863))

* Break ties in shared budgets by least spent output ([`eb1babc`](https://github.com/vitor-costa-garcia/fishladder/commit/eb1babccba02fd8c5b0dfeeed96b6212591a2164))

* Wait for a few labels before trusting a committee ([`fab4402`](https://github.com/vitor-costa-garcia/fishladder/commit/fab440253ae8be0fd6bd889673c5d7e8cdca64c6))

* Fix bugs found in review before 0.1.0 ([`ff0cae7`](https://github.com/vitor-costa-garcia/fishladder/commit/ff0cae72691097fb4f4a76d47700d8bff3f94865))

* Fix issues in randomized querying found in review ([`4d81986`](https://github.com/vitor-costa-garcia/fishladder/commit/4d819867b0ffc2f58cd2ab68388409da4814cd73))

### Test

* Add test for fishladder version ([`a720cb3`](https://github.com/vitor-costa-garcia/fishladder/commit/a720cb3ffcc8c3cc5126584da746063d8b6dbd28))

* Fix indentation in test_import.py ([`229547b`](https://github.com/vitor-costa-garcia/fishladder/commit/229547bdeff7838bd6318be40f0da20ce25998af))

* Add blank line before test function ([`ad832a2`](https://github.com/vitor-costa-garcia/fishladder/commit/ad832a2806b2f71f003d5c7a2041cebf3eb446bc))

### Unknown

* Initial commit ([`735e6fe`](https://github.com/vitor-costa-garcia/fishladder/commit/735e6fee8bc30fff3b41b77d668f7333f812f1d2))

* Initialize __init__.py with version number ([`7aa25fc`](https://github.com/vitor-costa-garcia/fishladder/commit/7aa25fc93931ea5219be0bcbc0b01a7ef11d06de))
