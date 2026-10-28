"""Development-task handoff records.

A handoff describes where development stands at a point in time.  It is data,
not authority: verifying it never executes the recorded commands, never resets
the worktree and never grants write access beyond the user's own request.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from .acceptance import code_commit, tree_digest, tree_dirty
from .contracts import CheckStatus, NonEmptyStr, RelativePath, Sha256, StrictModel
from .errors import InputError, ValidationFailure
from .store import read_json, utcnow

HANDOFF_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")


class HandoffCheck(StrictModel):
    check_id: str
    result_ref: RelativePath | None = None
    status_at_handoff: CheckStatus

    @field_validator("check_id")
    @classmethod
    def _check_id_shape(cls, value: str) -> str:
        if not HANDOFF_ID_RE.fullmatch(value):
            raise ValueError("check_id contains unsupported characters")
        return value

    @model_validator(mode="after")
    def _status_shape(self) -> "HandoffCheck":
        if self.status_at_handoff in (CheckStatus.PASS, CheckStatus.FAIL) and self.result_ref is None:
            raise ValueError("pass/fail at handoff requires a result reference")
        if self.status_at_handoff == CheckStatus.NOT_RUN and self.result_ref is not None:
            raise ValueError("not_run must not point at a result")
        return self


class NextAction(StrictModel):
    action: NonEmptyStr
    expected_evidence: NonEmptyStr


class DevelopmentHandoff(StrictModel):
    schema_version: Literal[1] = 1
    task_id: str
    goal: NonEmptyStr
    base_commit: str | None = None
    observed_tree_digest: Sha256
    allowed_paths: list[RelativePath] = Field(default_factory=list, max_length=100)
    requirement_ids: list[str] = Field(min_length=1, max_length=100)
    completed: list[NonEmptyStr] = Field(default_factory=list, max_length=100)
    pending: list[NonEmptyStr] = Field(default_factory=list, max_length=100)
    changed_paths: list[RelativePath] = Field(default_factory=list, max_length=200)
    checks: list[HandoffCheck] = Field(default_factory=list, max_length=100)
    blocked_by: list[NonEmptyStr] = Field(default_factory=list, max_length=50)
    known_risks: list[NonEmptyStr] = Field(default_factory=list, max_length=50)
    next_action: NextAction
    worktree_state: Literal["clean", "dirty"]
    updated_at: datetime | None = None

    @field_validator("task_id")
    @classmethod
    def _task_id_shape(cls, value: str) -> str:
        if not HANDOFF_ID_RE.fullmatch(value):
            raise ValueError("task_id contains unsupported characters")
        return value

    @field_validator("base_commit")
    @classmethod
    def _commit_shape(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("base_commit must be a full 40-character git commit")
        return value

    @model_validator(mode="after")
    def _no_duplicates(self) -> "DevelopmentHandoff":
        for label, values in (
            ("completed", self.completed),
            ("pending", self.pending),
            ("requirement_ids", self.requirement_ids),
            ("checks", [item.check_id for item in self.checks]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must not repeat entries")
        return self


def _changed_paths(root: Path) -> list[str] | None:
    """Non-ignored tracked/untracked changes; ``None`` outside git."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    paths: list[str] = []
    for line in completed.stdout.splitlines():
        value = line[3:]
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        paths.append(value.strip('"'))
    return sorted(set(paths))


def load_handoff(path: Path) -> DevelopmentHandoff:
    try:
        payload = read_json(path)
    except InputError:
        raise
    try:
        return DevelopmentHandoff.model_validate(payload)
    except ValidationError as exc:
        raise ValidationFailure("development handoff does not match the contract", detail=str(exc)) from exc


def verify_handoff(path: Path, *, project_root: Path) -> dict[str, Any]:
    handoff = load_handoff(path)
    root = project_root.resolve()
    current_digest = tree_digest(root)
    current_commit = code_commit(root)
    digest_match = current_digest == handoff.observed_tree_digest
    commit_match = handoff.base_commit is None or handoff.base_commit == current_commit
    problems: list[str] = []
    checks: list[dict[str, Any]] = []
    for check in handoff.checks:
        if check.result_ref is None:
            checks.append(
                {
                    "check_id": check.check_id,
                    "status_at_handoff": check.status_at_handoff.value,
                    "state": "current",
                    "result_ref": None,
                    "reason": "declared not_run at handoff",
                }
            )
            continue
        candidate = (root / check.result_ref).resolve()
        inside = candidate == root or root in candidate.parents
        exists = candidate.is_file()
        state = "current" if digest_match and inside and exists else "stale"
        reason = None
        if not inside:
            reason = "result reference escapes the project root"
            problems.append(reason)
        elif not exists:
            reason = "result reference is missing"
            problems.append(reason)
        elif not digest_match:
            reason = "the tree changed since the handoff; re-run the check before trusting it"
        checks.append(
            {
                "check_id": check.check_id,
                "status_at_handoff": check.status_at_handoff.value,
                "state": state,
                "result_ref": check.result_ref,
                "reason": reason,
            }
        )

    dirty = tree_dirty(root)
    if handoff.base_commit and current_commit is None:
        problems.append("cannot verify base_commit because the current tree is not a git worktree")
    if handoff.worktree_state == "clean" and dirty is True:
        problems.append("handoff says clean but the worktree has changes; do not reset it silently")
    elif handoff.worktree_state == "clean" and dirty is None:
        problems.append("cannot verify a clean worktree outside git; record the digest and treat state as unknown")
    actual_changes = _changed_paths(root)
    declared_changes = list(handoff.changed_paths)
    for changed in declared_changes:
        covered = any(
            changed == allowed or changed.startswith(allowed.rstrip("/") + "/")
            for allowed in handoff.allowed_paths
        )
        if not covered:
            problems.append(f"changed path is outside allowed_paths: {changed}")
    if actual_changes:
        for changed in actual_changes:
            if changed in declared_changes:
                continue
            covered = any(
                changed == allowed or changed.startswith(allowed.rstrip("/") + "/")
                for allowed in handoff.allowed_paths
            )
            if not covered:
                problems.append(f"worktree change is outside allowed_paths: {changed}")
    if not digest_match:
        problems.append("tree digest changed since the handoff; all check references are stale")
    if handoff.base_commit and current_commit and handoff.base_commit != current_commit:
        problems.append("HEAD moved since the handoff; re-read the task before acting")

    state = "current" if not problems else "changed"
    return {
        "schema_version": 1,
        "task_id": handoff.task_id,
        "state": state,
        "digest_match": digest_match,
        "observed_tree_digest": handoff.observed_tree_digest,
        "current_tree_digest": current_digest,
        "commit_match": commit_match,
        "base_commit": handoff.base_commit,
        "current_commit": current_commit,
        "worktree_state_at_handoff": handoff.worktree_state,
        "worktree_dirty_now": dirty,
        "checks": checks,
        "problems": problems,
        "next_action": handoff.next_action.model_dump(mode="json"),
        "allowed_paths": handoff.allowed_paths,
        "changed_paths_now": actual_changes,
        "verified_at": utcnow().isoformat(),
        "note": (
            "handoff is data, not authority: no recorded command was executed, no path was reset, "
            "and allowed_paths does not grant permissions beyond the user's request"
        ),
    }