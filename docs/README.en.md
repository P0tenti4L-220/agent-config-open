# agent-config-open

A Python CLI agent that turns development failures into evidence-grounded
reports, context packages, and handoffs. It uses one bounded agent loop with
seven restricted tools. DeepSeek and Anthropic are supported; read-only GitHub
issue lookup is optional.

## Latest update (2026-11-03)

Context selection now keeps exact evidence windows, unresolved failures, and
attempts that returned no new evidence. The new `read_evidence` tool continues
reading a registered snapshot from the current run with version-bound UTF-8
byte cursors; it does not fetch a new remote copy. Display proofs distinguish
shown ranges from omitted content. See [context selection](context-selection.md)
and the [verification record](verification.md). Paired live-model quality, cost,
and latency experiments remain `not_run`.

## Quick start

Python 3.11+ is required. The demo uses synthetic files and scripted responses;
it makes no network calls and needs no API key.

```bash
git clone https://github.com/P0tenti4L-220/agent-config-open.git
cd agent-config-open
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/agent-triage demo --output local/demo
```

Alternatively: `uv sync --frozen --extra dev`, then
`uv run --frozen agent-triage demo --output local/demo`.
On Windows, use `.venv\Scripts\python.exe` and `.venv\Scripts\agent-triage.exe`.

The demo investigates `KeyError: 'items'`: a reader expects `items`, while its
JSON input provides `checks`. The terminal prints paths to `report.md`,
`context.json`, and `handoff.md`. A `partial` analysis preserves explicit
missing evidence; it is not a failed command or a verified root cause.

## Live analysis

```bash
mkdir -p local
cp examples/deepseek.toml local/config.toml
export DEEPSEEK_API_KEY='<your-key>'
.venv/bin/agent-triage analyze \
  --project-root /path/to/project \
  --symptom-file /path/to/project/logs/failure.log \
  --project-id my-project --goal 'Investigate the failure and suggest checks' \
  --config local/config.toml --output local/runs
```

The log must be UTF-8, at most 2 MiB, and inside the declared project root.
See `examples/anthropic.toml` for Anthropic and `examples/github-mcp.toml` for
optional read-only GitHub lookup. Configure an available model ID for your account.

Live analysis sends selected project material to your configured model provider.
Local reports and traces can contain sensitive information. Keep them out of Git.
The agent restricts paths and tools, but cannot guarantee complete prompt-injection
or secret isolation. It does not fix code, run arbitrary shell, or publish issues.

## Development

```bash
.venv/bin/python -m pytest -m 'not live' -q
.venv/bin/agent-triage doctor --profile offline --output local/doctor
python3 scripts/check-public.py
```

Live tests require explicit `--run-live`. See [architecture](agent-architecture.md),
[privacy](privacy.md), [verification](verification.md), and [next steps](NEXT_STEPS.md).
License: [MIT](../LICENSE).
