# AGENTS.md

* Set up a virtual environment, then run `pip install -e ".[dev,docs]"` to install development
  dependencies.
    - Add `pip install -e ".[plots]" jupyter` to run the notebooks.
* Run doctests and unit tests with `pytest -q`. This runs `tests` and the doctests in `src`.
* Run lint and type checks with `ruff check .` and `mypy src`.
* Check notebooks execute with
  `jupyter nbconvert --to notebook --execute --stdout notebooks/*.ipynb > /dev/null`.
* Build documentation with `sphinx-build -W -b html docs docs/_build/html`.
* Clean documentation with `rm -rf docs/_build`.
* MUST run formatter `ruff format .` before committing code.
* MUST ask the user before pushing to GitHub.
* SHOULD add user-facing changes to the `Unreleased` section of `CHANGELOG.md`.

## Commit messages

* SHOULD use conventional commit messages. You MAY introduce breaking changes, but MUST
  NOT include `!` or `BREAKING CHANGE` in the commit message since we are pre v1.0.0.
* MUST NOT add `Co-Authored-By: Claude <noreply@anthropic.com>` (or similar) to commit messages.
* MUST add `Assisted-by: AGENT_NAME:MODEL_VERSION [TOOL1] [TOOL2] ...` to the end of
  commit messages and PR descriptions.
    * `AGENT_NAME` is the name of the harness (/tool/framework)
    * `MODEL_VERSION` is the specific model version. SHOULD be as specific as possible
       including model family, version, and size.
    * `[TOOL1] [TOOL2]` are optional agent developer tools or MCP. SHOULD NOT include
       basic developer tools.
    * Example: `Assisted-by: claude-code:claude-opus-5.5 github-mcp-server`
    * Example: `Assisted-by: copilot:gemini-3.7-flash`

## Documentation

* SHOULD use simplified technical english in documentation and comments.
* SHOULD NOT use long words where a short one will do.
* SHOULD use everyday words except where technical terms are required.
* If it is possible to cut a word out, you SHOULD cut it out.
* SHOULD avoid complex sentence structures.
* Consider using commas or parenthesis over em-dash or removing the clause entirely.
* Docstrings use the NumPy style, with code between single backticks.
