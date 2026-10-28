"""Acceptance case definitions and their canonical digests.

The same loader is used by ``eval`` (to execute) and ``accept`` (to decide what
the current checkout is allowed to claim).  If the two sides loaded different
definitions, a locally substituted case could satisfy a check under the same
name.  Keeping the loader, the per-case digest and the manifest digest here
makes that binding explicit.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from .contracts import StrictModel
from .errors import InputError


class EvalCase(StrictModel):
    schema_version: Literal[1, 2] = 2
    id: str
    title: str
    level: Literal["offline", "local_mcp_integration", "live", "human"]
    source: Literal["pytest", "run_artifact", "human_review", "aggregate"] = "pytest"
    pytest: list[str] = Field(default_factory=list, max_length=10)
    children: list[str] = Field(default_factory=list, max_length=20)
    expectation: str
    status: Literal["not_run"] = "not_run"

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{1,63}", value):
            raise ValueError("case id must be a safe check id")
        return value

    @model_validator(mode="before")
    @classmethod
    def _upgrade_v1(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if value.get("schema_version", 1) != 1:
            return value
        payload = dict(value)
        payload["schema_version"] = 2
        if payload.get("level") == "live_and_manual":
            payload["level"] = "live"
        payload.setdefault("source", "pytest")
        return payload

    @model_validator(mode="after")
    def _source_shape(self) -> "EvalCase":
        if self.source == "pytest":
            if not self.pytest:
                raise ValueError("pytest source requires at least one test node")
            if self.children:
                raise ValueError("pytest source must not declare children")
        elif self.source == "aggregate":
            if not self.children:
                raise ValueError("aggregate source requires children")
            if self.pytest:
                raise ValueError("aggregate source must not run tests itself")
        else:
            if self.pytest:
                raise ValueError(f"{self.source} must not declare pytest nodes")
            if self.children:
                raise ValueError(f"{self.source} must not declare children")
        return self


@dataclass(frozen=True)
class CaseManifest:
    directory: Path
    cases: dict[str, EvalCase]
    digest: str

    def require(self, check_id: str) -> EvalCase | None:
        return self.cases.get(check_id)


def load_cases(directory: Path) -> list[EvalCase]:
    if not directory.is_dir():
        raise InputError(f"cases directory not found: {directory}")
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for path in sorted(directory.glob("*.json")):
        try:
            case = EvalCase.model_validate_json(path.read_bytes())
        except (ValidationError, OSError) as exc:
            raise InputError(f"invalid eval case: {path.name}", detail=str(exc)) from exc
        if case.id in seen:
            raise InputError(f"duplicate eval case id: {case.id}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise InputError("no eval cases found")
    child_ids = {child for case in cases for child in case.children}
    unknown = sorted(child_ids - seen)
    if unknown:
        raise InputError("aggregate cases reference unknown children: " + ", ".join(unknown))
    return cases


def case_definition_payload(case: EvalCase) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "id": case.id,
        "title": case.title,
        "level": case.level,
        "source": case.source,
        "pytest": list(case.pytest),
        "children": list(case.children),
        "expectation": case.expectation,
    }


def case_definition_hash(case: EvalCase) -> str:
    return _sha256_text(_canonical(case_definition_payload(case)))


def manifest_digest(cases: list[EvalCase]) -> str:
    entries = [
        {"id": case.id, "definition_sha256": case_definition_hash(case)}
        for case in sorted(cases, key=lambda item: item.id)
    ]
    return _sha256_text(_canonical(entries))


def load_manifest(directory: Path) -> CaseManifest:
    cases = load_cases(directory)
    return CaseManifest(
        directory=directory,
        cases={case.id: case for case in cases},
        digest=manifest_digest(cases),
    )


def trusted_manifest(project_root: Path) -> CaseManifest | None:
    directory = project_root / "tests" / "cases"
    if not directory.is_dir():
        return None
    return load_manifest(directory)


def _canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()