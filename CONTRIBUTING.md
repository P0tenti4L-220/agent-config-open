# Contributing

Use Python 3.11+ and install with `uv sync --frozen --extra dev` or
`python -m pip install -e '.[dev]'` in a virtual environment.

Create a short branch, make a focused change, and open a pull request with the
problem, resulting behavior, and actual validation commands. Include a
meaningful regression case for behavioral fixes. Keep examples synthetic.

```bash
.venv/bin/python -m pytest -m 'not live' -q
python3 scripts/check-public.py
git diff --cached
git diff --cached --check
```

Live tests require explicit `--run-live`, credentials, and a designated test
repository. Do not include credentials or raw model responses in issues or
pull requests. Describe unperformed checks as `not_run`.

See [architecture](docs/agent-architecture.md), [privacy](docs/privacy.md),
and [next steps](docs/NEXT_STEPS.md) for boundaries and open work.
