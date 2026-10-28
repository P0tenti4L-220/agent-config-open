"""Frozen quality datasets, case references and byte-level verification.

The acceptance layer already binds a result to a *check* definition.  This
module adds the missing binding to the concrete diagnostic *case*: every
quality case names its canonical event, fixture files, reference material and
rubric, and every file is verified by raw-byte SHA-256 before it is copied into
an evidence directory.

The manifest is data only.  It is never imported as Python, never executed and
never resolved outside its own directory.  A dataset whose stored digest does
not match its contents is rejected instead of being repaired.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from .contracts import NonEmptyStr, RelativePath, Sha256, StrictInt, StrictModel
from .errors import InputError, ValidationFailure

QUALITY_SCHEMA_VERSION = 1
QUALITY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$")
PLAYBOOK_IDS = (
    "structured-output-mismatch",
    "tool-failure-and-recovery",
    "context-and-handoff-gap",
)
MAX_DATASET_FILE_BYTES = 8 * 1024 * 1024
MAX_DATASET_TOTAL_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 512 * 1024


class ArtifactRef(StrictModel):
    relative_path: RelativePath
    sha256: Sha256
    size_bytes: StrictInt = Field(ge=0, le=MAX_DATASET_FILE_BYTES)
    media_type: str = Field(min_length=3, max_length=100)

    @field_validator("media_type")
    @classmethod
    def _media_type_shape(cls, value: str) -> str:
        if "/" not in value:
            raise ValueError("media_type must look like type/subtype")
        return value


class DatasetCase(StrictModel):
    case_id: str = Field(min_length=4, max_length=40)
    group_id: NonEmptyStr
    split: Literal["pilot", "dev", "holdout"]
    category: NonEmptyStr
    playbook_id: Literal[
        "structured-output-mismatch",
        "tool-failure-and-recovery",
        "context-and-handoff-gap",
    ]
    input_ref: ArtifactRef
    fixture_refs: list[ArtifactRef] = Field(default_factory=list, max_length=20)
    reference_ref: ArtifactRef
    rubric_ref: ArtifactRef
    replay_ref: ArtifactRef | None = None
    origin: Literal["synthetic", "sanitized_real"]
    source_run_ref: ArtifactRef | None = None

    @field_validator("case_id")
    @classmethod
    def _case_id_shape(cls, value: str) -> str:
        if not QUALITY_ID_RE.fullmatch(value):
            raise ValueError("case_id contains unsupported characters")
        return value


class DatasetManifest(StrictModel):
    schema_version: Literal[1] = QUALITY_SCHEMA_VERSION
    dataset_id: str = Field(min_length=2, max_length=64)
    revision: str = Field(min_length=1, max_length=64)
    cases: list[DatasetCase] = Field(min_length=1, max_length=200)
    digest: Sha256

    @field_validator("dataset_id")
    @classmethod
    def _dataset_id_shape(cls, value: str) -> str:
        if not QUALITY_ID_RE.fullmatch(value):
            raise ValueError("dataset_id contains unsupported characters")
        return value

    @model_validator(mode="after")
    def _unique_cases(self) -> "DatasetManifest":
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("dataset repeats a case_id")
        return self


@dataclass(frozen=True)
class QualityDataset:
    root: Path
    manifest: DatasetManifest
    files: dict[str, Path]

    @property
    def digest(self) -> str:
        return self.manifest.digest

    @property
    def dataset_id(self) -> str:
        return self.manifest.dataset_id

    @property
    def cases(self) -> dict[str, DatasetCase]:
        return {case.case_id: case for case in self.manifest.cases}

    def require_case(self, case_id: str) -> DatasetCase:
        case = self.cases.get(case_id)
        if case is None:
            raise ValidationFailure(f"dataset {self.dataset_id} does not define case {case_id}")
        return case

    def file_for(self, ref: ArtifactRef) -> Path:
        path = self.files.get(ref.relative_path)
        if path is None:
            raise ValidationFailure(f"dataset file is not registered: {ref.relative_path}")
        return path


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def dataset_digest(manifest: DatasetManifest) -> str:
    payload = manifest.model_dump(mode="json")
    payload.pop("digest", None)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def compute_dataset_digest(payload: dict[str, Any]) -> str:
    """Compute the digest of a not-yet-validated manifest payload."""
    without = dict(payload)
    without.pop("digest", None)
    return hashlib.sha256(canonical_json(without).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_symlink_chain(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = root / relative
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValidationFailure(f"dataset path must not be a symbolic link: {relative}")
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValidationFailure(f"dataset path escapes its root: {relative}")
    if not resolved.is_file():
        raise ValidationFailure(f"dataset file is missing: {relative}")
    return resolved


def _verify_ref(root: Path, ref: ArtifactRef, *, total: list[int]) -> Path:
    path = _reject_symlink_chain(root, ref.relative_path)
    size = path.stat().st_size
    if size > MAX_DATASET_FILE_BYTES:
        raise ValidationFailure(f"dataset file exceeds {MAX_DATASET_FILE_BYTES} bytes: {ref.relative_path}")
    if size != ref.size_bytes:
        raise ValidationFailure(f"dataset file size does not match the manifest: {ref.relative_path}")
    if _sha256_file(path) != ref.sha256:
        raise ValidationFailure(f"dataset file hash does not match the manifest: {ref.relative_path}")
    total[0] += size
    if total[0] > MAX_DATASET_TOTAL_BYTES:
        raise ValidationFailure("dataset exceeds the total size budget")
    return path


def _all_refs(case: DatasetCase) -> list[ArtifactRef]:
    refs = [case.input_ref, case.reference_ref, case.rubric_ref, *case.fixture_refs]
    if case.replay_ref is not None:
        refs.append(case.replay_ref)
    if case.source_run_ref is not None:
        refs.append(case.source_run_ref)
    return refs


def load_dataset(root: Path) -> QualityDataset:
    root = Path(root)
    if root.is_symlink():
        raise InputError(f"dataset root must not be a symbolic link: {root}")
    if not root.is_dir():
        raise InputError(f"dataset directory not found: {root}")
    root = root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise InputError("dataset manifest.json is missing or a symbolic link")
    raw = manifest_path.read_bytes()
    if len(raw) > MAX_MANIFEST_BYTES:
        raise InputError("dataset manifest is larger than 512 KiB")
    try:
        manifest = DatasetManifest.model_validate_json(raw)
    except ValidationError as exc:
        raise ValidationFailure("dataset manifest does not match the expected shape", detail=str(exc)) from exc
    actual = dataset_digest(manifest)
    if actual != manifest.digest:
        raise ValidationFailure(
            "dataset manifest digest does not match its contents; the dataset was modified after freezing"
        )
    total = [0]
    files: dict[str, Path] = {}
    for case in manifest.cases:
        for ref in _all_refs(case):
            path = _verify_ref(root, ref, total=total)
            prior = files.get(ref.relative_path)
            if prior is not None and prior != path:
                raise ValidationFailure(f"dataset path is ambiguous: {ref.relative_path}")
            files[ref.relative_path] = path
    return QualityDataset(root=root, manifest=manifest, files=files)


def write_dataset_manifest(root: Path, payload: dict[str, Any]) -> DatasetManifest:
    """Freeze a manifest payload by adding its canonical digest."""
    candidate = dict(payload)
    candidate["digest"] = compute_dataset_digest(candidate)
    manifest = DatasetManifest.model_validate(candidate)
    if dataset_digest(manifest) != manifest.digest:
        raise ValidationFailure("dataset digest is not canonical")
    target = root / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest