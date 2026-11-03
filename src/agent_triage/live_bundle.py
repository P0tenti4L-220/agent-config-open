"""Portable live-run bundles and their import into acceptance evidence.

A live pytest pass and a run artifact must describe the same run.  The live test
writes a bundle from the real run directory, the bundle verifier recomputes the
issue chain from the copied artifacts, and the HC09 importer turns that verified
bundle into a ``run_artifact`` result bound to the run's target and the current
case definition.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import Field, ValidationError, field_validator, model_validator

from .acceptance import Artifact, ExecutionResult, Target, trusted_manifest
from .case_manifest import case_definition_hash
from .contracts import CheckStatus, NonEmptyStr, Sha256, StrictInt, StrictModel
from .errors import InputError, ValidationFailure
from .livecheck import verify_issue_chain
from .store import atomic_write_json, read_json, utcnow

BUNDLE_SCHEMA_VERSION = 1
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
MANIFEST_NAME = "manifest.json"
CHAIN_NAME = "chain.json"
RUN_ID_RE = re.compile(r"^run-[0-9]{8}-[0-9]{6}-[0-9a-f]{6}$")

_FILE_NAMES = {
    "run.json",
    "event.json",
    "config.json",
    "analysis.json",
    "checks.json",
    "display.json",
    "trace.jsonl",
    "report.md",
    "handoff.md",
    "context.json",
    "investigation.json",
    "lifecycle.json",
    "versions.json",
    "replay.json",
    "review-status.json",
    "evidence/index.json",
}
_DIR_PREFIXES = ("evidence/snippets/", "evidence/objects/", "request-views/")


class BundleFile(StrictModel):
    relative_path: str = Field(min_length=1, max_length=300)
    sha256: Sha256
    size: StrictInt = Field(ge=0, le=MAX_FILE_BYTES)

    @field_validator("relative_path")
    @classmethod
    def _safe_relative_path(cls, value: str) -> str:
        pure = PurePosixPath(value)
        if pure.is_absolute() or ".." in pure.parts or "\\" in value:
            raise ValueError("bundle path must be a safe relative POSIX path")
        return value


class LiveBundleManifest(StrictModel):
    schema_version: StrictInt = BUNDLE_SCHEMA_VERSION
    bundle_id: str
    run_id: str
    issue_ref: str
    marker: NonEmptyStr
    marker_sha256: Sha256
    created_at: str
    target: Target
    chain: dict[str, Any]
    files: list[BundleFile] = Field(min_length=1, max_length=500)

    @field_validator("bundle_id", "run_id")
    @classmethod
    def _run_id_shape(cls, value: str) -> str:
        if not RUN_ID_RE.fullmatch(value):
            raise ValueError("live bundle id must look like run-YYYYMMDD-HHMMSS-abcdef")
        return value

    @model_validator(mode="after")
    def _version(self) -> "LiveBundleManifest":
        if self.schema_version != BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported live bundle schema_version")
        return self


def write_live_bundle(
    run_dir: Path,
    bundle_dir: Path,
    *,
    issue_ref: str,
    marker: str,
    target: Target,
) -> LiveBundleManifest:
    run_dir = run_dir.resolve()
    bundle_dir = bundle_dir.resolve()
    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        raise InputError(f"live bundle directory is not empty: {bundle_dir}")
    chain = verify_issue_chain(run_dir, issue_ref=issue_ref, marker=marker)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    files: list[BundleFile] = []
    total = 0
    for path, relative in _candidate_files(run_dir):
        data = path.read_bytes()
        if len(data) > MAX_FILE_BYTES:
            raise InputError(f"live bundle file is too large: {relative}")
        total += len(data)
        if total > MAX_BUNDLE_BYTES:
            raise InputError("live bundle exceeds the total size limit")
        destination = bundle_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        files.append(
            BundleFile(
                relative_path=relative,
                sha256=hashlib.sha256(data).hexdigest(),
                size=len(data),
            )
        )
    atomic_write_json(bundle_dir / CHAIN_NAME, chain)
    files.append(
        BundleFile(
            relative_path=CHAIN_NAME,
            sha256=hashlib.sha256((bundle_dir / CHAIN_NAME).read_bytes()).hexdigest(),
            size=(bundle_dir / CHAIN_NAME).stat().st_size,
        )
    )
    manifest = LiveBundleManifest(
        bundle_id=run_dir.name,
        run_id=chain.get("run_id", run_dir.name),
        issue_ref=issue_ref,
        marker=marker,
        marker_sha256=hashlib.sha256(marker.encode("utf-8")).hexdigest(),
        created_at=utcnow().isoformat(),
        target=target,
        chain=chain,
        files=files,
    )
    atomic_write_json(bundle_dir / MANIFEST_NAME, manifest.model_dump(mode="json"))
    return manifest


def verify_live_bundle(bundle_dir: Path, *, issue_ref: str, marker: str) -> LiveBundleManifest:
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValidationFailure("live bundle has no manifest.json")
    try:
        manifest = LiveBundleManifest.model_validate(read_json(manifest_path))
    except (InputError, ValidationError) as exc:
        raise ValidationFailure("live bundle manifest is invalid", detail=str(exc)) from exc
    if manifest.issue_ref != issue_ref:
        raise ValidationFailure("live bundle was produced for a different issue_ref")
    if manifest.marker != marker:
        raise ValidationFailure("live bundle marker does not match the frozen marker")
    if manifest.marker_sha256 != hashlib.sha256(marker.encode("utf-8")).hexdigest():
        raise ValidationFailure("live bundle marker hash is inconsistent")
    listed: set[str] = set()
    for item in manifest.files:
        if not _is_allowed_relative(item.relative_path):
            raise ValidationFailure(
                f"live bundle lists a path outside the allowlist: {item.relative_path}"
            )
        path = (bundle_dir / item.relative_path).resolve()
        if bundle_dir not in path.parents:
            raise ValidationFailure("live bundle file escapes the bundle directory")
        if not path.is_file():
            raise ValidationFailure(f"live bundle file is missing: {item.relative_path}")
        data = path.read_bytes()
        if len(data) != item.size:
            raise ValidationFailure(f"live bundle file size changed: {item.relative_path}")
        if hashlib.sha256(data).hexdigest() != item.sha256:
            raise ValidationFailure(f"live bundle file hash changed: {item.relative_path}")
        listed.add(item.relative_path)
    present = {
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    }
    if present - listed:
        raise ValidationFailure(
            "live bundle contains files not covered by the manifest: "
            + ", ".join(sorted(present - listed)[:3])
        )
    run_payload = read_json(bundle_dir / "run.json")
    if str(run_payload.get("run_id")) != manifest.run_id:
        raise ValidationFailure("live bundle run.json and manifest run_id differ")
    chain = verify_issue_chain(bundle_dir, issue_ref=issue_ref, marker=marker)
    if chain.get("read_call_id") != manifest.chain.get("read_call_id"):
        raise ValidationFailure("live bundle manifest and recomputed chain disagree")
    return manifest


def import_live_bundle(
    bundle_dir: Path,
    evidence_dir: Path,
    *,
    project_root: Path,
    attempt: str = "1",
    evaluation_id: str | None = None,
) -> tuple[ExecutionResult, Path]:
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,40}", attempt):
        raise InputError("attempt id must contain only letters, digits, dot, underscore or dash")
    bundle_dir = bundle_dir.resolve()
    evidence_dir = evidence_dir.resolve()
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValidationFailure("live bundle has no manifest.json")
    try:
        raw = read_json(manifest_path)
        manifest = LiveBundleManifest.model_validate(raw)
    except (InputError, ValidationError) as exc:
        raise ValidationFailure("live bundle manifest is invalid", detail=str(exc)) from exc
    # The manifest carries the frozen marker; verify_live_bundle re-runs the
    # complete issue chain against the copied artifacts before import.
    issue_ref = manifest.issue_ref
    if not (bundle_dir / "evidence" / "index.json").is_file():
        raise ValidationFailure("live bundle has no evidence index")
    verify_live_bundle(bundle_dir, issue_ref=issue_ref, marker=manifest.marker)

    case_manifest = trusted_manifest(project_root)
    case_hash = None
    if case_manifest is not None and "HC09" in case_manifest.cases:
        case_hash = case_definition_hash(case_manifest.cases["HC09"])

    destination = evidence_dir / "live" / manifest.bundle_id
    if destination.resolve() != bundle_dir:
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(bundle_dir, destination)

    artifacts: list[Artifact] = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(evidence_dir).as_posix()
        data = path.read_bytes()
        artifacts.append(
            Artifact(
                relative_path=relative,
                sha256=hashlib.sha256(data).hexdigest(),
                media_type="application/json" if path.suffix == ".json" else "text/plain",
            )
        )
    result = ExecutionResult(
        evaluation_id=evaluation_id
        or _evidence_evaluation_id(evidence_dir)
        or (
            utcnow().strftime("eval-%Y%m%d-%H%M%S-")
            + hashlib.sha256(manifest.bundle_id.encode()).hexdigest()[:6]
        ),
        check_id="HC09",
        attempt_id=attempt,
        status=CheckStatus.PASS,
        source_type="run_artifact",
        case_definition_hash=case_hash,
        target=manifest.target,
        execution={
            "argv": ["agent-triage", "live-bundle-import"],
            "exit_code": 0,
            "executor_id": "live-bundle",
        },
        related_run_ids=[manifest.run_id],
        artifacts=artifacts[:50],
        nodes=[],
    )
    path = evidence_dir / "results" / f"HC09-{attempt}.json"
    atomic_write_json(path, result.model_dump(mode="json"))
    return result, path


def _evidence_evaluation_id(evidence_dir: Path) -> str | None:
    """Bind an imported run artifact to the directory that owns it."""
    path = evidence_dir / "eval.json"
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except InputError:
        return None
    if payload.get("schema_version") != 2:
        return None
    value = payload.get("evaluation_id")
    return value if isinstance(value, str) else None


def _is_allowed_relative(relative: str) -> bool:
    if relative == CHAIN_NAME:
        return True
    return relative in _FILE_NAMES or any(
        relative.startswith(prefix) and len(relative) > len(prefix)
        for prefix in _DIR_PREFIXES
    )


def _candidate_files(run_dir: Path):
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir).as_posix()
        if _is_allowed_relative(relative):
            yield path, relative
