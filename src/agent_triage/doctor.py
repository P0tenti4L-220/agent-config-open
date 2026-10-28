"""Read-only environment readiness checks.

``doctor`` answers one question: *can this checkout run the requested profile
on this interpreter?*  It does not install anything, start the MCP process,
make network calls or print credential values.  A ready doctor report is not a
successful live acceptance run; ``network`` and ``model_call`` stay ``not_run``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .config import load_config
from .contracts import Config, NonEmptyStr, StrictModel
from .errors import InputError, ValidationFailure
from .store import atomic_write_json

DoctorStatus = Literal["pass", "blocked", "error", "not_run"]


class DoctorCheck(StrictModel):
    check_id: str
    status: DoctorStatus
    detail: NonEmptyStr
    required_for: list[Literal["offline", "connected"]] = Field(default_factory=list)


class DoctorReport(StrictModel):
    schema_version: Literal[1] = 1
    profile: Literal["offline", "connected"]
    status: Literal["ready", "blocked", "error"]
    checks: list[DoctorCheck]
    notes: list[str] = Field(default_factory=list)
    network: Literal["not_run"] = "not_run"
    model_call: Literal["not_run"] = "not_run"
    output_dir: str
    probe_cleaned: bool = True

    def summary(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "status": self.status,
            "checks": [check.model_dump(mode="json") for check in self.checks],
            "notes": self.notes,
            "network": self.network,
            "model_call": self.model_call,
            "output_dir": Path(self.output_dir).name,
            "probe_cleaned": self.probe_cleaned,
            "note": (
                "environment readiness only; network and real model/MCP calls remain not_run "
                "and doctor ready is not AC15.live acceptance"
            ),
        }


def _check(check_id: str, status: DoctorStatus, detail: str, required_for: list[str]) -> DoctorCheck:
    return DoctorCheck(check_id=check_id, status=status, detail=detail, required_for=required_for)  # type: ignore[arg-type]


def run_doctor(
    *,
    profile: str,
    output: Path,
    config_path: Path | None = None,
    project_root: Path | None = None,
    env: dict[str, str] | None = None,
    interpreter_version: tuple[int, int, int] | None = None,
) -> DoctorReport:
    if profile not in {"offline", "connected"}:
        raise InputError("doctor profile must be offline or connected")
    environment = dict(os.environ) if env is None else dict(env)
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    version = interpreter_version or (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    checks: list[DoctorCheck] = []
    notes: list[str] = []

    python_ok = version >= (3, 11, 0)
    checks.append(
        _check(
            "interpreter",
            "pass" if python_ok else "blocked",
            f"Python {version[0]}.{version[1]}.{version[2]} ({Path(sys.executable).name})"
            + ("" if python_ok else "; requires 3.11+"),
            ["offline", "connected"],
        )
    )

    try:
        import agent_triage

        package_path = Path(agent_triage.__file__).resolve()
        package_version = agent_triage.__version__
    except Exception as exc:  # noqa: BLE001 - import failure is the finding
        checks.append(
            _check("package_import", "blocked", f"agent_triage import failed: {type(exc).__name__}", ["offline", "connected"])
        )
        package_path = None
        package_version = None
    else:
        expected = (root / "src" / "agent_triage").resolve()
        if expected.is_dir() and not (
            package_path == expected / "__init__.py" or expected in package_path.parents
        ):
            checks.append(
                _check(
                    "import_root",
                    "blocked",
                    "the imported agent_triage is not from this worktree; use ./.venv/bin/python -m agent_triage",
                    ["offline", "connected"],
                )
            )
        else:
            checks.append(_check("import_root", "pass", "agent_triage is imported from this worktree", ["offline", "connected"]))
        checks.append(
            _check("package_import", "pass", f"agent_triage {package_version} imported", ["offline", "connected"])
        )

    required_modules = {"pydantic": "offline", "anthropic": "connected"}
    for module, scope in required_modules.items():
        found = importlib.util.find_spec(module) is not None
        checks.append(
            _check(
                f"dependency:{module}",
                "pass" if found else "blocked",
                f"module {module} is {'available' if found else 'missing'}",
                ["offline", "connected"] if scope == "offline" else ["connected"],
            )
        )

    config: Config | None = None
    try:
        config = load_config(config_path)
        checks.append(
            _check(
                "config",
                "pass",
                f"config is valid; provider={config.provider}; model={config.model}"
                + (f"; file={config_path.name}" if config_path else "; built-in defaults"),
                ["offline", "connected"],
            )
        )
    except Exception as exc:  # noqa: BLE001 - config errors are the finding
        checks.append(_check("config", "blocked", f"config is invalid: {type(exc).__name__}", ["offline", "connected"]))

    if config is not None:
        if config.provider in {"anthropic", "deepseek"}:
            present = bool(environment.get(config.credential_env))
            checks.append(
                _check(
                    "provider_credential",
                    "pass" if present else "blocked",
                    f"model credential {config.credential_env} is {'present' if present else 'missing'} (value never read)",
                    ["connected"],
                )
            )
            checks.append(_check("model_name", "pass" if config.model and config.model != "replay" else "blocked", "model name is configured" if config.model and config.model != "replay" else "model name must be explicit for live runs", ["connected"]))
        else:
            checks.append(
                _check(
                    "provider_credential",
                    "blocked",
                    "connected readiness requires provider=anthropic or deepseek with a credential; replay is offline only",
                    ["connected"],
                )
            )
            checks.append(
                _check(
                    "model_name",
                    "blocked",
                    "connected readiness requires an explicit live model, not the replay model",
                    ["connected"],
                )
            )

        mcp = config.mcp
        if not mcp.enabled:
            checks.append(
                _check(
                    "mcp_config",
                    "blocked",
                    "GitHub MCP is disabled; connected readiness requires the read-only MCP configuration",
                    ["connected"],
                )
            )
        else:
            assert mcp.command is not None
            executable = shutil.which(mcp.command) if "/" not in mcp.command else (mcp.command if Path(mcp.command).is_file() else None)
            checks.append(
                _check(
                    "mcp_executable",
                    "pass" if executable else "blocked",
                    "MCP executable found" if executable else "MCP executable was not found on PATH or disk",
                    ["connected"],
                )
            )
            token_present = bool(environment.get(mcp.token_env))
            checks.append(
                _check(
                    "mcp_token",
                    "pass" if token_present else "blocked",
                    f"GitHub token variable {mcp.token_env} is {'present' if token_present else 'missing'} (value never read)",
                    ["connected"],
                )
            )
            checks.append(
                _check(
                    "mcp_repo",
                    "pass" if mcp.repo else "blocked",
                    "repository scope is configured" if mcp.repo else "repository scope is missing",
                    ["connected"],
                )
            )

    output = output.resolve()
    probe = output / f".doctor-probe-{os.getpid()}"
    cleaned = True
    try:
        output.mkdir(parents=True, exist_ok=True)
        if not output.is_dir():
            raise OSError("output path is not a directory")
        probe.write_text("probe\n", encoding="utf-8")
        if probe.read_text(encoding="utf-8") != "probe\n":
            raise OSError("probe read-back mismatch")
        checks.append(_check("output_probe", "pass", "output directory accepts a temporary probe", ["offline", "connected"]))
    except OSError as exc:
        checks.append(
            _check("output_probe", "blocked", f"output directory is not usable: {type(exc).__name__}", ["offline", "connected"])
        )
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            cleaned = False

    checks.append(_check("network", "not_run", "doctor never opens a network connection", ["connected"]))
    checks.append(_check("model_call", "not_run", "doctor never calls the model", ["connected"]))

    required = [check for check in checks if profile in check.required_for]
    if any(check.status == "error" for check in required):
        status: Literal["ready", "blocked", "error"] = "error"
    elif any(check.status == "blocked" for check in required):
        status = "blocked"
    else:
        status = "ready"

    if profile == "connected" and status == "ready":
        notes.append("connected readiness only means configuration is present; run the live smoke test to prove the chain")
    if profile == "offline":
        notes.append("offline readiness does not need a model credential or GitHub MCP")
    report = DoctorReport(
        profile=profile,  # type: ignore[arg-type]
        status=status,
        checks=checks,
        notes=notes,
        output_dir=str(output),
        probe_cleaned=cleaned,
    )
    try:
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output / "doctor.json", report.model_dump(mode="json"))
    except OSError:
        report.notes.append("doctor.json could not be written because the output directory is unusable")
    return report


def check_doctor_report(report: DoctorReport) -> None:
    """Raise if a report claims readiness with a blocked required check."""
    required = [check for check in report.checks if report.profile in check.required_for]
    if report.status == "ready" and any(check.status == "blocked" for check in required):
        raise ValidationFailure("doctor report contradicts its own required checks")
