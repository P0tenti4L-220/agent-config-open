"""Prepare and import acceptance-level human reviews with verifiable evidence.

Schema 3 does not trust a hash string on its own.  A quality review must point
at the bounded run bundle that was read, list every file in it and name the
event, run metadata, analysis, display records and reference material.  Import
re-hashes those bytes, checks the run/config/case binding, and refuses to write
a record that does not line up.  ``accept`` repeats the checks read-only on
every run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from .acceptance import (
    CASE_EVIDENCE_ROLES,
    FROZEN_QUALITY_CASES,
    HUMAN_SUMMARY_CHECKS,
    REQUIRED_PLAYBOOKS,
    Artifact,
    EvaluationEnvelope,
    ExecutionResult,
    HumanCaseScore,
    HumanCheckRecord,
    ImportOrigin,
    Target,
    _safe_artifact_path,
    _sha256_file,
    human_review_problems,
)
from .case_manifest import CaseManifest, case_definition_hash
from .contracts import Analysis, DisplayedEvidence, RelativePath, StrictInt, StrictModel
from .errors import InputError, ValidationFailure
from .quality import ArtifactRef, DatasetCase, QualityDataset, load_dataset
from .review import analysis_content_hash
from .store import RunMetadata, atomic_write_json, read_json, utcnow

MAX_REVIEW_BYTES = 1024 * 1024
MAX_BUNDLE_FILES = 200
MAX_BUNDLE_BYTES = 64 * 1024 * 1024
RECORD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,119}$")
REQUIRED_RUN_FILES = ("event.json", "run.json", "analysis.json", "display.json")


def read_envelope(evidence_dir: Path) -> EvaluationEnvelope:
    evidence_dir = Path(evidence_dir).resolve()
    path = evidence_dir / "eval.json"
    if not path.is_file():
        raise InputError("evidence directory has no eval.json envelope")
    payload = read_json(path)
    try:
        envelope = EvaluationEnvelope.model_validate(payload)
    except ValidationError as exc:
        raise ValidationFailure("eval.json has no valid evaluation envelope", detail=str(exc)) from exc
    provenance_path = evidence_dir / "provenance.json"
    if provenance_path.is_file():
        try:
            provenance_id = read_json(provenance_path).get("evaluation_id")
        except InputError as exc:
            raise ValidationFailure("provenance.json is unreadable", detail=str(exc)) from exc
        if provenance_id is not None and provenance_id != envelope.evaluation_id:
            raise ValidationFailure(
                "provenance.json evaluation_id does not match eval.json; the root identity is ambiguous"
            )
    return envelope


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".json": "application/json",
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".jsonl": "application/x-ndjson",
        ".xml": "application/xml",
    }.get(suffix, "application/octet-stream")


def _artifact(path: Path, root: Path) -> Artifact:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValidationFailure(f"review file escapes the evidence directory: {path}") from exc
    return Artifact(
        relative_path=relative,
        sha256=_sha256_file(resolved),
        media_type=_media_type(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _role(record: HumanCaseScore, role: str) -> str:
    return record.roles[role]


def _relative_under(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValidationFailure(f"path escapes the evidence directory: {path}") from exc


def _verify_artifact(evidence_dir: Path, artifact: Artifact, *, label: str) -> Path:
    path = _safe_artifact_path(evidence_dir, artifact.relative_path)
    size = path.stat().st_size
    if artifact.size_bytes is not None and artifact.size_bytes != size:
        raise ValidationFailure(f"{label} size changed: {artifact.relative_path}")
    if _sha256_file(path) != artifact.sha256:
        raise ValidationFailure(f"{label} hash does not match: {artifact.relative_path}")
    return path


def _bundle_files(bundle: Path) -> list[Path]:
    files: list[Path] = []
    for current, dirnames, filenames in os.walk(bundle, followlinks=False):
        directory = Path(current)
        for name in list(dirnames):
            if (directory / name).is_symlink():
                raise ValidationFailure(f"bundle directory must not be a symbolic link: {name}")
        for name in filenames:
            path = directory / name
            if path.is_symlink():
                raise ValidationFailure(f"bundle file must not be a symbolic link: {path.name}")
            if not path.is_file():
                raise ValidationFailure(f"bundle entry is not a regular file: {path.name}")
            files.append(path)
    return sorted(files)


def _load_request_views(bundle: Path) -> list[Any]:
    from .request_views import RequestView

    directory = bundle / "request-views"
    if not directory.is_dir():
        return []
    views = []
    for path in sorted(directory.glob("*.json")):
        try:
            views.append(RequestView.model_validate(read_json(path)))
        except (InputError, ValidationError) as exc:
            raise ValidationFailure(f"request view is invalid: {path.name}", detail=str(exc)) from exc
    return views


def _load_evidence(bundle: Path) -> list[Any]:
    from .store import EvidenceIndex

    path = bundle / "evidence" / "index.json"
    if not path.is_file():
        return []
    try:
        return EvidenceIndex.model_validate(read_json(path)).evidence
    except (InputError, ValidationError) as exc:
        raise ValidationFailure("evidence index in the review bundle is invalid", detail=str(exc)) from exc


def _verify_display(bundle: Path) -> list[str]:
    from .request_views import verify_display_records

    path = bundle / "display.json"
    if not path.is_file():
        return ["display.json is missing from the review bundle"]
    try:
        raw = read_json(path)
        displayed = [DisplayedEvidence.model_validate(item) for item in raw.get("displayed", [])]
    except (InputError, ValidationError, AttributeError) as exc:
        return [f"display.json is invalid: {exc}"]
    views = _load_request_views(bundle)
    evidence = _load_evidence(bundle)
    problems, mode = verify_display_records(displayed, views, evidence)
    if mode != "request_view":
        problems.append(
            "display records are not request-view bound; rerun the analysis so display evidence "
            "proves what entered the provider request"
        )
    return problems


def _dataset_from_record(record: HumanCheckRecord, evidence_dir: Path) -> QualityDataset:
    if record.dataset_ref is None:
        raise ValidationFailure("quality review has no dataset_ref")
    manifest_path = _verify_artifact(evidence_dir, record.dataset_ref, label="dataset manifest")
    dataset = load_dataset(manifest_path.parent)
    if dataset.digest != record.dataset_digest:
        raise ValidationFailure("dataset digest does not match the imported manifest")
    if dataset.dataset_id != record.dataset_id:
        raise ValidationFailure("dataset_id does not match the imported manifest")
    return dataset


def _rubric_index(dataset: QualityDataset) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset_id": dataset.dataset_id,
        "dataset_digest": dataset.digest,
        "cases": [
            {"case_id": case.case_id, "rubric_sha256": case.rubric_ref.sha256}
            for case in sorted(dataset.manifest.cases, key=lambda item: item.case_id)
        ],
    }


def _check_rubric(record: HumanCheckRecord, dataset: QualityDataset, evidence_dir: Path) -> list[str]:
    problems: list[str] = []
    if record.rubric_ref is None or record.rubric_digest is None:
        return ["quality review has no rubric binding"]
    path = _verify_artifact(evidence_dir, record.rubric_ref, label="rubric index")
    if record.rubric_ref.sha256 != record.rubric_digest:
        problems.append("rubric_digest does not match rubric_ref.sha256")
    try:
        payload = read_json(path)
    except InputError as exc:
        return [f"rubric index is unreadable: {exc}"]
    expected = _rubric_index(dataset)
    if payload != expected:
        problems.append("rubric index does not match the frozen dataset rubrics")
    return problems


def _verify_case_bundle(
    case: HumanCaseScore,
    dataset_case: DatasetCase,
    evidence_dir: Path,
    target: Target,
) -> list[str]:
    problems: list[str] = []
    bundle = (evidence_dir / case.bundle_dir).resolve()
    if evidence_dir.resolve() not in bundle.parents:
        return [f"{case.case_id}: bundle_dir escapes the evidence directory"]
    if not bundle.is_dir():
        return [f"{case.case_id}: review bundle is missing"]
    try:
        on_disk = {_relative_under(path, evidence_dir) for path in _bundle_files(bundle)}
    except ValidationFailure as exc:
        return [f"{case.case_id}: {exc}"]
    declared = {artifact.relative_path for artifact in case.artifacts}
    if on_disk != declared:
        missing = sorted(declared - on_disk)[:3]
        extra = sorted(on_disk - declared)[:3]
        problems.append(
            f"{case.case_id}: bundle allowlist mismatch (missing={missing}, unregistered={extra})"
        )
    if len(declared) > MAX_BUNDLE_FILES:
        problems.append(f"{case.case_id}: bundle has too many files")
    total = 0
    for artifact in case.artifacts:
        try:
            total += _verify_artifact(evidence_dir, artifact, label=f"{case.case_id} bundle").stat().st_size
        except ValidationFailure as exc:
            problems.append(str(exc))
    if total > MAX_BUNDLE_BYTES:
        problems.append(f"{case.case_id}: bundle exceeds the total size budget")

    actual_hashes: dict[str, str] = {}
    prefix = case.bundle_dir.rstrip("/") + "/"
    for role in CASE_EVIDENCE_ROLES:
        relative = case.roles[role]
        if not relative.startswith(prefix):
            problems.append(f"{case.case_id}: {role} role is outside the declared bundle")
        if relative not in declared:
            problems.append(f"{case.case_id}: {role} role is not in the bundle allowlist")
        try:
            path = _safe_artifact_path(evidence_dir, relative)
            actual_hashes[role] = _sha256_file(path)
        except (OSError, ValidationFailure):
            problems.append(f"{case.case_id}: {role} role is missing or unsafe")
    if case.reviewed_subject_hashes.get("event") != actual_hashes.get("event"):
        problems.append(f"{case.case_id}: reviewed event hash does not match the bundle")
    if actual_hashes.get("event") != dataset_case.input_ref.sha256:
        problems.append(f"{case.case_id}: event does not match the frozen dataset input")
    if actual_hashes.get("reference") != dataset_case.reference_ref.sha256:
        problems.append(f"{case.case_id}: reference does not match the frozen dataset reference")
    if case.reviewed_subject_hashes.get("reference") != actual_hashes.get("reference"):
        problems.append(f"{case.case_id}: reviewed reference hash does not match the bundle")

    run_path = evidence_dir / case.roles["run"]
    try:
        meta = RunMetadata.model_validate(read_json(run_path))
    except (InputError, ValidationError) as exc:
        return problems + [f"{case.case_id}: run.json is invalid: {exc}"]
    if meta.run_id != case.run_id:
        problems.append(f"{case.case_id}: run_id does not match run.json")
    mode = "replay" if meta.provider == "replay" else "live"
    if mode != case.provider_mode:
        problems.append(
            f"{case.case_id}: provider_mode={case.provider_mode}, but run.json says {meta.provider}"
        )
    expected_config = target.live_config_hash if mode == "live" else target.config_hash
    if expected_config is None:
        problems.append(f"{case.case_id}: no configuration fingerprint was declared for this run mode")
    elif meta.config_fingerprint != expected_config:
        problems.append(f"{case.case_id}: run.json config fingerprint does not match the reviewed target")

    analysis_path = evidence_dir / case.roles["analysis"]
    try:
        analysis = Analysis.model_validate(read_json(analysis_path))
    except (InputError, ValidationError) as exc:
        return problems + [f"{case.case_id}: analysis.json is invalid: {exc}"]
    content_hash = analysis_content_hash(analysis)
    if case.reviewed_subject_hashes.get("analysis_content") != content_hash:
        problems.append(f"{case.case_id}: reviewed analysis content hash does not match analysis.json")
    if case.reviewed_subject_hashes.get("analysis") != _sha256_file(analysis_path):
        problems.append(f"{case.case_id}: reviewed analysis bytes do not match analysis.json")
    if case.reviewed_subject_hashes.get("run") != _sha256_file(run_path):
        problems.append(f"{case.case_id}: reviewed run hash does not match run.json")
    try:
        display_hash = _sha256_file(evidence_dir / case.roles["display"])
    except OSError:
        display_hash = ""
    if case.reviewed_subject_hashes.get("display") != display_hash:
        problems.append(f"{case.case_id}: reviewed display hash does not match display.json")
    problems.extend(f"{case.case_id}: {item}" for item in _verify_display(bundle))
    return problems


def verify_human_record(
    record: HumanCheckRecord,
    evidence_dir: Path,
    *,
    envelope: EvaluationEnvelope | None,
    manifest: CaseManifest | None,
) -> list[str]:
    """Return hard binding problems.

    Judgment problems (too few passes, a pass with unresolved blockers) are
    handled by :func:`human_review_problems` and produce ``fail``, not
    ``error``.  This function only decides whether the record may be imported
    or counted at all.
    """
    evidence_dir = Path(evidence_dir).resolve()
    problems: list[str] = []
    if envelope is None:
        problems.append("no root eval.json envelope; the review cannot claim a current evaluation")
    else:
        if record.evaluation_id != envelope.evaluation_id:
            problems.append(
                f"review belongs to {record.evaluation_id}, but this directory is {envelope.evaluation_id}"
            )
        if (
            record.target.code_commit != envelope.target.code_commit
            or record.target.relevant_tree_digest != envelope.target.relevant_tree_digest
        ):
            problems.append("review target does not match the evaluation target")
    if manifest is None:
        problems.append("no trusted tests/cases manifest; the check definition cannot be verified")
    else:
        spec = manifest.cases.get(record.check_id)
        if spec is None:
            problems.append(f"check {record.check_id} is not defined in tests/cases")
        else:
            if spec.source != "human_review":
                problems.append(f"check {record.check_id} does not accept human_review evidence")
            if record.case_definition_hash != case_definition_hash(spec):
                problems.append("case_definition_hash does not match the current check definition")

    if record.imported_from is not None:
        try:
            source_path = _safe_artifact_path(evidence_dir, record.imported_from.source_record_path)
        except ValidationFailure as exc:
            problems.append(str(exc))
        else:
            if _sha256_file(source_path) != record.imported_from.source_record_sha256:
                problems.append("imported source record hash does not match the retained copy")
            else:
                try:
                    source = read_json(source_path)
                except InputError as exc:
                    problems.append(f"imported source record is unreadable: {exc}")
                else:
                    if source.get("evaluation_id") != record.imported_from.source_evaluation_id:
                        problems.append("imported source evaluation_id does not match the retained copy")
                    if source.get("record_id") != record.record_id:
                        problems.append("imported source record_id does not match")

    if record.check_id in HUMAN_SUMMARY_CHECKS:
        try:
            dataset = _dataset_from_record(record, evidence_dir)
        except ValidationFailure as exc:
            return problems + [str(exc)]
        dataset_ids = [case.case_id for case in dataset.manifest.cases]
        if tuple(sorted(dataset_ids)) != tuple(sorted(FROZEN_QUALITY_CASES)):
            problems.append("dataset must freeze exactly the five QA case ids")
        playbooks = {case.playbook_id for case in dataset.manifest.cases}
        if not set(REQUIRED_PLAYBOOKS).issubset(playbooks):
            problems.append("dataset must cover all three playbooks")
        problems.extend(_check_rubric(record, dataset, evidence_dir))
        seen: set[str] = set()
        for case in record.human_review.cases:
            if case.case_id in seen:
                problems.append(f"duplicate case binding: {case.case_id}")
                continue
            seen.add(case.case_id)
            dataset_case = dataset.cases.get(case.case_id)
            if dataset_case is None:
                problems.append(f"{case.case_id} is not defined in the frozen dataset")
                continue
            problems.extend(
                _verify_case_bundle(
                    case,
                    dataset_case,
                    evidence_dir,
                    record.target,
                )
            )
        paths = [artifact.relative_path for artifact in record.artifacts]
        if len(paths) != len(set(paths)):
            problems.append("record repeats an artifact path")
    return problems


def _commit_review(*, record: HumanCheckRecord, evidence_dir: Path, status: str) -> dict[str, Any]:
    target = evidence_dir / "human" / f"{record.check_id}.json"
    atomic_write_json(target, record.model_dump(mode="json"))
    receipt = {
        "schema_version": 1,
        "check_id": record.check_id,
        "record_id": record.record_id,
        "record_sha256": _sha256_file(target),
        "check_status": status,
        "imported_at": utcnow().isoformat(),
        "source_evaluation_id": (
            record.imported_from.source_evaluation_id if record.imported_from else record.evaluation_id
        ),
    }
    atomic_write_json(evidence_dir / "human" / "receipts" / f"{record.check_id}.json", receipt)
    return {"receipt": receipt, "record_path": str(target)}


def import_review(
    evidence_dir: Path,
    review_path: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    evidence_dir = Path(evidence_dir).resolve()
    review_path = Path(review_path)
    envelope = read_envelope(evidence_dir)
    if not review_path.is_file():
        raise InputError(f"review file not found: {review_path.name}")
    raw = review_path.read_bytes()
    if len(raw) > MAX_REVIEW_BYTES:
        raise ValidationFailure("review file is larger than 1 MiB")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InputError("review file is not valid JSON", detail=str(exc)) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 3:
        raise ValidationFailure("only schema 3 human reviews can be imported; v1/v2 stay historical")
    try:
        record = HumanCheckRecord.model_validate(payload)
    except ValidationError as exc:
        raise ValidationFailure("review does not match schema 3", detail=str(exc)) from exc
    if not RECORD_ID_RE.fullmatch(record.record_id):
        raise ValidationFailure("record_id contains unsupported characters")
    manifest = _manifest_for(project_root)
    source_id = record.evaluation_id
    source_sha = hashlib.sha256(raw).hexdigest()
    imported_path = evidence_dir / "human" / "imports" / f"{record.record_id}-{source_sha[:12]}.json"
    imported_path.parent.mkdir(parents=True, exist_ok=True)
    imported_path.write_bytes(raw)
    rewritten = record.model_copy(
        update={
            "evaluation_id": envelope.evaluation_id,
            "imported_from": ImportOrigin(
                source_evaluation_id=source_id,
                source_record_sha256=source_sha,
                source_record_path=_relative_under(imported_path, evidence_dir),
                imported_at=utcnow(),
            ),
        }
    )
    try:
        problems = verify_human_record(
            rewritten,
            evidence_dir,
            envelope=envelope,
            manifest=manifest,
        )
    except ValidationFailure as exc:
        imported_path.unlink(missing_ok=True)
        raise
    if problems:
        imported_path.unlink(missing_ok=True)
        raise ValidationFailure(
            f"review does not bind to verifiable evidence: {problems[0]}",
            detail="; ".join(problems[:8]),
        )
    judgment = human_review_problems(rewritten.check_id, rewritten.human_review)
    status = "fail" if judgment else "pass"
    committed = _commit_review(record=rewritten, evidence_dir=evidence_dir, status=status)
    return {
        "status": "imported",
        "check_id": rewritten.check_id,
        "check_status": status,
        "judgment_problems": judgment,
        **committed,
    }


def _manifest_for(project_root: Path | None = None) -> CaseManifest | None:
    from .case_manifest import trusted_manifest

    manifest = trusted_manifest(project_root or Path(__file__).resolve().parents[2])
    if manifest is None:
        raise ValidationFailure("current checkout has no trusted tests/cases manifest")
    return manifest


def _copy_dataset(dataset: QualityDataset, evidence_dir: Path) -> Path:
    destination = evidence_dir / "human" / "dataset"
    if destination.exists():
        existing = load_dataset(destination)
        if existing.digest != dataset.digest:
            raise ValidationFailure("evidence directory already holds a different quality dataset")
        return destination
    destination.mkdir(parents=True)
    manifest_payload = dataset.manifest.model_dump(mode="json")
    atomic_write_json(destination / "manifest.json", manifest_payload)
    for relative, source in dataset.files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    copied = load_dataset(destination)
    if copied.digest != dataset.digest:
        raise ValidationFailure("copied dataset does not reproduce the frozen digest")
    return destination


def _subject_run_dir(subjects: Path, case_id: str) -> Path:
    candidate = subjects / case_id
    if not candidate.is_dir():
        raise InputError(f"subjects directory has no run for {case_id}")
    if not (candidate / "run.json").is_file():
        children = [item for item in candidate.iterdir() if item.is_dir() and (item / "run.json").is_file()]
        if len(children) != 1:
            raise InputError(f"subjects/{case_id} must contain exactly one run directory")
        candidate = children[0]
    return candidate


def _copy_case_bundle(run_dir: Path, bundle: Path, reference: Path) -> list[str]:
    bundle.mkdir(parents=True)
    copied: list[str] = []
    for name in REQUIRED_RUN_FILES:
        source = run_dir / name
        if not source.is_file():
            raise ValidationFailure(f"run {run_dir.name} is missing {name}")
        shutil.copyfile(source, bundle / name)
        copied.append(name)
    index = run_dir / "evidence" / "index.json"
    if index.is_file():
        (bundle / "evidence").mkdir(parents=True)
        shutil.copyfile(index, bundle / "evidence" / "index.json")
        copied.append("evidence/index.json")
    views = run_dir / "request-views"
    if views.is_dir():
        (bundle / "request-views").mkdir(parents=True)
        for path in sorted(views.glob("*.json")):
            shutil.copyfile(path, bundle / "request-views" / path.name)
            copied.append(f"request-views/{path.name}")
    reference_name = "reference" + (reference.suffix or ".md")
    shutil.copyfile(reference, bundle / reference_name)
    copied.append(reference_name)
    return copied


def prepare_review(
    *,
    evidence_dir: Path,
    check_id: str,
    dataset_root: Path | None,
    subjects_dir: Path | None,
    output: Path,
    project_root: Path | None = None,
) -> dict[str, Any]:
    evidence_dir = Path(evidence_dir).resolve()
    envelope = read_envelope(evidence_dir)
    manifest = _manifest_for(project_root)
    spec = manifest.cases.get(check_id)
    if spec is None:
        raise InputError(f"unknown check: {check_id}")
    if spec.source != "human_review":
        raise ValidationFailure(f"{check_id} does not accept a human review")
    template: dict[str, Any] = {
        "schema_version": 3,
        "record_id": f"review-{utcnow().strftime('%Y%m%d-%H%M%S')}",
        "evaluation_id": envelope.evaluation_id,
        "check_id": check_id,
        "case_definition_hash": case_definition_hash(spec),
        "target": envelope.target.model_dump(mode="json"),
        "reviewer_id": "",
        "review_method": "human",
        "reviewed_at": "",
        "artifacts": [],
        "human_review": {
            "reviewer": "",
            "method": "human",
            "reviewed_at": "",
            "rubric_version": "",
            "summary": "",
        },
    }
    if check_id in HUMAN_SUMMARY_CHECKS:
        if dataset_root is None or subjects_dir is None:
            raise InputError("quality review prepare requires --dataset and --subjects")
        dataset = load_dataset(dataset_root)
        if tuple(sorted(case.case_id for case in dataset.manifest.cases)) != tuple(
            sorted(FROZEN_QUALITY_CASES)
        ):
            raise ValidationFailure("dataset must freeze exactly QA01..QA05")
        if not set(REQUIRED_PLAYBOOKS).issubset({case.playbook_id for case in dataset.manifest.cases}):
            raise ValidationFailure("dataset must cover all three playbooks")
        copied_dataset = _copy_dataset(dataset, evidence_dir)
        dataset_manifest = _artifact(copied_dataset / "manifest.json", evidence_dir)
        rubric_path = evidence_dir / "human" / "rubric.json"
        atomic_write_json(rubric_path, _rubric_index(dataset))
        rubric_artifact = _artifact(rubric_path, evidence_dir)
        template["dataset_ref"] = dataset_manifest.model_dump(mode="json")
        template["dataset_digest"] = dataset.digest
        template["dataset_id"] = dataset.dataset_id
        template["rubric_ref"] = rubric_artifact.model_dump(mode="json")
        template["rubric_digest"] = rubric_artifact.sha256
        template["artifacts"] = [dataset_manifest.model_dump(mode="json"), rubric_artifact.model_dump(mode="json")]
        cases = []
        subjects = Path(subjects_dir).resolve()
        for dataset_case in sorted(dataset.manifest.cases, key=lambda item: item.case_id):
            run_dir = _subject_run_dir(subjects, dataset_case.case_id)
            run_meta = RunMetadata.model_validate(read_json(run_dir / "run.json"))
            bundle_dir = evidence_dir / "human" / "bundles" / dataset_case.case_id
            reference_path = dataset.file_for(dataset_case.reference_ref)
            if bundle_dir.exists():
                # AC15.quality and HC10 review the same five runs.  Reuse a
                # complete bundle only when it still belongs to the same run
                # and reference; otherwise a stale bundle would silently stand
                # in for a different subject.
                try:
                    existing_run = RunMetadata.model_validate(read_json(bundle_dir / "run.json"))
                except (InputError, ValidationError) as exc:
                    raise ValidationFailure(f"existing bundle for {dataset_case.case_id} is invalid", detail=str(exc)) from exc
                if existing_run.run_id != run_meta.run_id:
                    raise ValidationFailure(
                        f"existing bundle for {dataset_case.case_id} belongs to {existing_run.run_id}, "
                        f"not {run_meta.run_id}"
                    )
                existing_reference = bundle_dir / ("reference" + (reference_path.suffix or ".md"))
                if not existing_reference.is_file() or _sha256_file(existing_reference) != dataset_case.reference_ref.sha256:
                    raise ValidationFailure(
                        f"existing bundle for {dataset_case.case_id} has a different reference"
                    )
                copied = [_relative_under(path, bundle_dir) for path in _bundle_files(bundle_dir)]
            else:
                copied = _copy_case_bundle(run_dir, bundle_dir, reference_path)
            artifacts = [_artifact(bundle_dir / name, evidence_dir) for name in copied]
            roles = {
                "event": _relative_under(bundle_dir / "event.json", evidence_dir),
                "run": _relative_under(bundle_dir / "run.json", evidence_dir),
                "analysis": _relative_under(bundle_dir / "analysis.json", evidence_dir),
                "display": _relative_under(bundle_dir / "display.json", evidence_dir),
                "reference": _relative_under(
                    bundle_dir / ("reference" + (reference_path.suffix or ".md")), evidence_dir
                ),
            }
            analysis = Analysis.model_validate(read_json(bundle_dir / "analysis.json"))
            hashes = {role: _sha256_file(evidence_dir / path) for role, path in roles.items()}
            hashes["analysis_content"] = analysis_content_hash(analysis)
            cases.append(
                {
                    "case_id": dataset_case.case_id,
                    "run_id": run_meta.run_id,
                    "provider_mode": "replay" if run_meta.provider == "replay" else "live",
                    "bundle_dir": _relative_under(bundle_dir, evidence_dir),
                    "artifacts": [item.model_dump(mode="json") for item in artifacts],
                    "roles": roles,
                    "reviewed_subject_hashes": hashes,
                    "verdict": None,
                    "rationale": "",
                    "unsupported_fact": False,
                    "unexecuted_check_reported_pass": False,
                    "unauthorized_write": False,
                }
            )
            template["artifacts"].extend(item.model_dump(mode="json") for item in artifacts)
        template["human_review"]["cases"] = cases
    elif check_id == "NC20.manual":
        template["human_review"]["playbooks"] = [
            {
                "playbook_id": playbook,
                "sample": "",
                "steps": "",
                "outcome": "",
                "blockers": [],
                "verdict": None,
            }
            for playbook in REQUIRED_PLAYBOOKS
        ]
    elif check_id == "HC13":
        template["human_review"]["handoff"] = {
            "operator": "",
            "no_chat_history": True,
            "target_identified": False,
            "facts_identified": False,
            "gaps_identified": False,
            "next_step_identified": False,
            "result": "",
            "blockers": [],
            "verdict": None,
        }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, template)
    return {
        "status": "prepared",
        "check_id": check_id,
        "output": str(output),
        "evaluation_id": envelope.evaluation_id,
        "cases": [case["case_id"] for case in template["human_review"].get("cases", [])],
        "note": "prepared values are unverified until import; verdicts are intentionally blank",
    }