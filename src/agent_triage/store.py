"""Local run storage: atomic artifacts, evidence index, registry, checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError

from .contracts import (
    RUN_ID_RE,
    BudgetLimits,
    ContextPacket,
    DisplayedEvidence,
    Evidence,
    EvidenceKind,
    EvidenceSummary,
    InvestigationState,
    Lesson,
    LessonStatus,
    LessonSummary,
    RelativePath,
    RunPosition,
    Sha256,
    StrictInt,
    StrictModel,
    TraceRecord,
)
from .errors import InputError, StoreError

ARTIFACT_NAMES = {
    "event.json",
    "analysis.json",
    "report.md",
    "context.json",
    "handoff.md",
    "trace.jsonl",
    "checks.json",
    "issue-draft.md",
    "lesson-candidate.json",
    "run.json",
    "checkpoint.json",
    "config.json",
    "replay.json",
    "outcome.json",
    "lifecycle.json",
    "display.json",
    "investigation.json",
    "versions.json",
    "promotion-check.json",
    "evidence/index.json",
}
SENSITIVE_PARTS = {
    ".env",
    "auth.json",
    ".credentials.json",
    "id_rsa",
    "id_ed25519",
    ".netrc",
}
SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".sqlite", ".sqlite3", ".db"}
SENSITIVE_DIRS = {".ssh", ".aws", ".gnupg", ".config"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def atomic_write_text(path: Path, text: str) -> None:
    """Write and fsync a temporary sibling, then atomically replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise StoreError(f"cannot write {path.name}", detail=str(exc)) from exc


def atomic_write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(path, text)


def read_json(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"cannot read {path.name}", detail=str(exc)) from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InputError(f"{path.name} is not valid JSON", detail=str(exc)) from exc


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


RunStatus = Literal["needs_review", "awaiting_verification", "resolved"]


class RegistryEvent(StrictModel):
    event_id: str
    project_id: str
    fingerprint: str
    symptom: str = ""
    first_seen: datetime
    last_seen: datetime
    run_ids: list[str] = Field(default_factory=list, max_length=500)


class RegistryFingerprint(StrictModel):
    algorithm: str
    occurrence: StrictInt = Field(ge=1)
    symptom: str
    event_ids: list[str] = Field(default_factory=list, max_length=500)


class RegistryRun(StrictModel):
    run_id: str
    event_id: str
    project_id: str
    status: RunStatus
    provider: str
    created_at: datetime
    updated_at: datetime


class Registry(StrictModel):
    schema_version: Literal[1] = 1
    events: dict[str, RegistryEvent] = Field(default_factory=dict)
    fingerprints: dict[str, RegistryFingerprint] = Field(default_factory=dict)
    runs: list[RegistryRun] = Field(default_factory=list, max_length=5000)


class RegistryStore:
    def __init__(self, output_root: Path) -> None:
        self.path = output_root / "index.json"

    def load(self) -> Registry:
        if not self.path.exists():
            return Registry()
        payload = read_json(self.path)
        try:
            return Registry.model_validate(payload)
        except ValidationError as exc:
            raise StoreError("run registry is corrupt; refusing to reset it", detail=str(exc)) from exc

    def save(self, registry: Registry) -> None:
        atomic_write_json(self.path, registry.model_dump(mode="json"))

    def register(
        self,
        *,
        event_id: str,
        project_id: str,
        fingerprint: str,
        fingerprint_algorithm: str,
        symptom: str,
        run_id: str,
        provider: str,
    ) -> tuple[int, bool, str]:
        """Return (fingerprint occurrence, duplicate event?, fingerprint)."""
        registry = self.load()
        now = utcnow()
        existing = registry.events.get(event_id)
        duplicate_event = existing is not None
        group = registry.fingerprints.get(fingerprint)
        if group is None:
            group = RegistryFingerprint(
                algorithm=fingerprint_algorithm,
                occurrence=1,
                symptom=symptom[:500],
                event_ids=[event_id],
            )
            registry.fingerprints[fingerprint] = group
        elif not duplicate_event:
            group.occurrence += 1
            if event_id not in group.event_ids:
                group.event_ids.append(event_id)
        if existing is None:
            registry.events[event_id] = RegistryEvent(
                event_id=event_id,
                project_id=project_id,
                fingerprint=fingerprint,
                symptom=symptom[:500],
                first_seen=now,
                last_seen=now,
                run_ids=[run_id],
            )
        else:
            existing.last_seen = now
            if run_id not in existing.run_ids:
                existing.run_ids.append(run_id)
            registry.events[event_id] = existing
        registry.runs.append(
            RegistryRun(
                run_id=run_id,
                event_id=event_id,
                project_id=project_id,
                status="needs_review",
                provider=provider,
                created_at=now,
                updated_at=now,
            )
        )
        self.save(registry)
        return group.occurrence, duplicate_event, fingerprint

    def set_run_status(self, run_id: str, status: RunStatus) -> None:
        registry = self.load()
        changed = False
        for run in registry.runs:
            if run.run_id == run_id:
                run.status = status
                run.updated_at = utcnow()
                changed = True
        if not changed:
            raise StoreError(f"run {run_id} is not registered")
        self.save(registry)

    def duplicate_candidates(self, *, event_id: str, fingerprint: str) -> list[dict[str, str]]:
        registry = self.load()
        candidates: list[dict[str, str]] = []
        group = registry.fingerprints.get(fingerprint)
        if group:
            for other in group.event_ids:
                if other == event_id:
                    continue
                record = registry.events.get(other)
                if record is None:
                    continue
                candidates.append(
                    {
                        "source": "local_event",
                        "reference": other,
                        "title": record.project_id,
                        "similarity_reason": f"fingerprint_match:{group.algorithm}",
                        "state": "unknown",
                    }
                )
        # Same project, different fingerprint: similar symptom keywords only.
        current = registry.events.get(event_id)
        if current:
            current_words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]{2,}", current.symptom.lower()))
            for other, record in registry.events.items():
                if other == event_id or other in {c["reference"] for c in candidates}:
                    continue
                if record.project_id != current.project_id:
                    continue
                other_words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]{2,}", record.symptom.lower()))
                if not (current_words & other_words):
                    continue
                candidates.append(
                    {
                        "source": "local_event",
                        "reference": other,
                        "title": record.project_id,
                        "similarity_reason": "same_project_keyword_overlap",
                        "state": "unknown",
                    }
                )
                if len(candidates) >= 5:
                    break
        return candidates[:5]

    def list_runs(self, status: str | None, limit: int) -> list[RegistryRun]:
        registry = self.load()
        runs = sorted(registry.runs, key=lambda run: run.created_at, reverse=True)
        if status and status != "all":
            runs = [run for run in runs if run.status == status]
        return runs[:limit]


# ---------------------------------------------------------------------------
# Run metadata and lifecycle
# ---------------------------------------------------------------------------


class RunMetadata(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str
    event_id: str
    project_id: str
    provider: str
    model: str
    config_fingerprint: str
    # Repository-relative or output-relative, never an absolute local path.
    project_root: str
    status: RunStatus = "needs_review"
    interruption: str | None = None
    created_at: datetime
    updated_at: datetime


class BudgetState(StrictModel):
    elapsed_ms: StrictInt = Field(default=0, ge=0, le=86_400_000)
    model_requests_used: StrictInt = Field(default=0, ge=0, le=1000)
    tool_attempts_used: StrictInt = Field(default=0, ge=0, le=10_000)
    corrections_used: StrictInt = Field(default=0, ge=0, le=10)


class Checkpoint(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str
    turn: StrictInt = Field(ge=0, le=1000)
    replay_position: StrictInt = Field(ge=0, le=10_000)
    budget: BudgetState
    conversation: list[dict[str, Any]] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list, max_length=1000)
    completed_call_ids: list[str] = Field(default_factory=list, max_length=1000)
    source_hashes: dict[RelativePath, Sha256] = Field(default_factory=dict)
    config_fingerprint: str
    investigation: InvestigationState | None = None
    displayed: list[DisplayedEvidence] = Field(default_factory=list, max_length=1000)
    policy_versions: dict[str, str] = Field(default_factory=dict)
    playbook_id: str | None = None
    playbook_version: str | None = None
    playbook_hash: str | None = None
    updated_at: datetime


class RunStore:
    """One directory per run; artifacts are written atomically."""

    def __init__(self, output_root: Path, run_id: str, meta: RunMetadata) -> None:
        self.output_root = output_root.resolve()
        self.run_id = run_id
        self.dir = self.output_root / run_id
        self.meta = meta

    # -- construction -------------------------------------------------------
    @classmethod
    def create(
        cls,
        output_root: Path,
        *,
        event_payload: dict[str, Any],
        event_id: str,
        project_id: str,
        project_root: Path,
        provider: str,
        model: str,
        config_fingerprint: str,
    ) -> "RunStore":
        output_root = output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        now = utcnow()
        for _ in range(5):
            run_id = f"run-{now.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
            run_dir = output_root / run_id
            try:
                run_dir.mkdir(mode=0o700)
                break
            except FileExistsError:
                continue
        else:
            raise StoreError("could not allocate a unique run directory")
        meta = RunMetadata(
            run_id=run_id,
            event_id=event_id,
            project_id=project_id,
            provider=provider,
            model=model,
            config_fingerprint=config_fingerprint,
            project_root=os.path.relpath(project_root.resolve(), output_root),
            created_at=now,
            updated_at=now,
        )
        store = cls(output_root, run_id, meta)
        (store.dir / "evidence").mkdir(mode=0o700)
        store.write_json("event.json", event_payload)
        store.save_meta()
        return store

    @classmethod
    def load(cls, output_root: Path, run_id: str) -> "RunStore":
        if not RUN_ID_RE.fullmatch(run_id):
            raise InputError("run id has an unsupported shape")
        output_root = output_root.resolve()
        run_dir = output_root / run_id
        if not run_dir.is_dir():
            raise InputError(f"run not found: {run_id}")
        payload = read_json(run_dir / "run.json")
        try:
            meta = RunMetadata.model_validate(payload)
        except ValidationError as exc:
            raise StoreError("run metadata is corrupt", detail=str(exc)) from exc
        if meta.run_id != run_id:
            raise StoreError("run metadata does not match its directory")
        return cls(output_root, run_id, meta)

    def project_root(self) -> Path:
        candidate = Path(self.meta.project_root)
        if candidate.is_absolute():
            # Older checkpoints may contain an absolute path; accept it locally
            # but never render it in reports or handoff artifacts.
            return candidate.resolve()
        return (self.output_root / candidate).resolve()

    # -- metadata -----------------------------------------------------------
    def save_meta(self) -> None:
        self.meta.updated_at = utcnow()
        self.write_json("run.json", self.meta.model_dump(mode="json"))

    def set_status(self, status: RunStatus, *, interruption: str | None = None) -> None:
        self.meta.status = status
        self.meta.interruption = interruption
        self.save_meta()

    # -- artifacts ----------------------------------------------------------
    def _artifact_path(self, name: str) -> Path:
        if name not in ARTIFACT_NAMES and not name.startswith(
            ("evidence/snippets/", "evidence/objects/", "evidence/outcomes/")
        ):
            raise StoreError(f"artifact name is not allowed: {name}")
        path = self.dir / name
        if self.dir.resolve() not in path.resolve().parents:
            raise StoreError("artifact path escapes the run directory")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self._artifact_path(name)
        atomic_write_text(path, text)
        return path

    def write_json(self, name: str, payload: Any) -> Path:
        path = self._artifact_path(name)
        atomic_write_json(path, payload)
        return path

    def read_json(self, name: str) -> Any:
        return read_json(self._artifact_path(name))

    def exists(self, name: str) -> bool:
        return self._artifact_path(name).exists()

    def append_trace(self, record: TraceRecord) -> None:
        path = self._artifact_path("trace.jsonl")
        line = json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n"
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def write_trace(self, records: list[TraceRecord]) -> None:
        text = "".join(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n" for record in records
        )
        self.write_text("trace.jsonl", text)

    def read_trace(self) -> list[TraceRecord]:
        path = self._artifact_path("trace.jsonl")
        if not path.exists():
            return []
        records: list[TraceRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(TraceRecord.model_validate_json(line))
            except ValidationError as exc:
                raise StoreError("trace contains an invalid record", detail=str(exc)) from exc
        return records

    # -- checkpoint ---------------------------------------------------------
    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        self.write_json("checkpoint.json", checkpoint.model_dump(mode="json"))

    def load_checkpoint(self) -> Checkpoint | None:
        if not self.exists("checkpoint.json"):
            return None
        payload = self.read_json("checkpoint.json")
        try:
            return Checkpoint.model_validate(payload)
        except ValidationError as exc:
            raise StoreError("checkpoint is corrupt", detail=str(exc)) from exc

    # -- lock ---------------------------------------------------------------
    def acquire_lock(self) -> "RunLock":
        return RunLock(self.dir / ".lock").acquire()

    # -- context ------------------------------------------------------------
    def load_context(self) -> ContextPacket | None:
        if not self.exists("context.json"):
            return None
        payload = self.read_json("context.json")
        try:
            return ContextPacket.model_validate(payload)
        except ValidationError as exc:
            raise StoreError("context packet is corrupt", detail=str(exc)) from exc


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.owned = False

    def acquire(self) -> "RunLock":
        for attempt in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if attempt == 0 and self._break_if_stale():
                    continue
                raise StoreError("another process is writing this run")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"{os.getpid()}\n{time.time()}\n")
            self.owned = True
            return self
        raise StoreError("another process is writing this run")

    def _break_if_stale(self) -> bool:
        try:
            first = self.path.read_text(encoding="utf-8").splitlines()[0]
            pid = int(first)
        except (OSError, ValueError, IndexError):
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            try:
                self.path.unlink()
            except OSError:
                return False
            return True
        except PermissionError:
            return False
        return False

    def release(self) -> None:
        if self.owned:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self.owned = False

    def __enter__(self) -> "RunLock":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class EvidenceIndex(StrictModel):
    schema_version: Literal[1, 2] = 2
    evidence: list[Evidence] = Field(default_factory=list, max_length=2000)


class EvidenceStore:
    def __init__(self, run: RunStore) -> None:
        self.run = run
        self.path = run.dir / "evidence" / "index.json"
        self._cache: dict[str, Evidence] | None = None

    def _load(self) -> dict[str, Evidence]:
        if self._cache is not None:
            return self._cache
        if not self.path.exists():
            self._cache = {}
            return self._cache
        payload = read_json(self.path)
        try:
            index = EvidenceIndex.model_validate(payload)
        except ValidationError as exc:
            raise StoreError("evidence index is corrupt", detail=str(exc)) from exc
        self._cache = {item.evidence_id: item for item in index.evidence}
        return self._cache

    def _save(self) -> None:
        items = list(self._load().values())
        atomic_write_json(self.path, EvidenceIndex(evidence=items).model_dump(mode="json"))

    def add(self, evidence: Evidence) -> Evidence:
        current = self._load()
        existing = current.get(evidence.evidence_id)
        if existing is not None:
            return existing
        snippet_path = self.run.dir / "evidence" / "snippets" / f"{evidence.evidence_id}.txt"
        atomic_write_text(snippet_path, evidence.snippet)
        # Publish the index last. A failed index write must not leave a cached
        # reference that another tool could mistake for a committed capture.
        updated = {**current, evidence.evidence_id: evidence}
        atomic_write_json(self.path, EvidenceIndex(evidence=list(updated.values())).model_dump(mode="json"))
        self._cache = updated
        return evidence

    def add_capture(
        self, evidence: Evidence, text: str, *, start_byte: int,
        capture_scope: str, complete_within_scope: bool,
        notes: list[str] | None = None,
    ) -> Evidence:
        raw = text.encode("utf-8")
        if len(raw) > 8 * 1024 * 1024:
            raise StoreError("capture exceeds the 8 MiB archive limit")
        end = start_byte + len(evidence.snippet.encode("utf-8"))
        if raw[start_byte:end] != evidence.snippet.encode("utf-8"):
            raise StoreError("evidence snippet is not an exact slice of its capture")
        digest = hashlib.sha256(raw).hexdigest()
        path = self.run._artifact_path(f"evidence/objects/{digest}.txt")
        if path.exists():
            if path.read_bytes() != raw:
                raise StoreError("existing capture object is corrupt")
        else:
            atomic_write_text(path, text)
        captured = Evidence.model_validate({
            **evidence.model_dump(mode="json"), "schema_version": 2,
            "snapshot_sha256": digest, "snapshot_bytes": len(raw),
            "snapshot_start_byte": start_byte, "snapshot_end_byte": end,
            "snapshot_line_start": (evidence.line_start - raw[:start_byte].count(b"\n")) if evidence.line_start else None,
            "capture_scope": capture_scope, "complete_within_scope": complete_within_scope,
            "capture_notes": [note[:500] for note in (notes or [])],
        })
        return self.add(captured)

    def capture_text(self, evidence_id: str) -> str:
        item = self.require(evidence_id)
        if item.snapshot_sha256 is None:
            # Legacy records contain only this snippet. Never re-open a live
            # source to manufacture the missing historical content.
            return item.snippet
        path = self.run._artifact_path(f"evidence/objects/{item.snapshot_sha256}.txt")
        try:
            if path.stat().st_size != item.snapshot_bytes:
                raise StoreError("capture object length mismatch")
            raw = path.read_bytes()
        except OSError as exc:
            raise StoreError("capture object is missing or unreadable") from exc
        if hashlib.sha256(raw).hexdigest() != item.snapshot_sha256:
            raise StoreError("capture object hash mismatch")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise StoreError("capture object is not UTF-8") from exc
        if raw[item.snapshot_start_byte:item.snapshot_end_byte] != item.snippet.encode("utf-8"):
            raise StoreError("capture range does not match the stored snippet")
        return text

    def get(self, evidence_id: str) -> Evidence | None:
        return self._load().get(evidence_id)

    def require(self, evidence_id: str) -> Evidence:
        item = self.get(evidence_id)
        if item is None:
            raise InputError(f"unknown evidence id: {evidence_id}")
        return item

    def all(self) -> list[Evidence]:
        return list(self._load().values())

    def ids(self) -> list[str]:
        return list(self._load().keys())

    def summaries(self) -> list[EvidenceSummary]:
        return [
            EvidenceSummary(
                evidence_id=item.evidence_id,
                kind=item.kind,
                path=item.path,
                line_start=item.line_start,
                line_end=item.line_end,
                url=item.url,
                sha256=item.sha256,
                captured_at=item.captured_at,
                truncated=item.truncated,
                stale=item.stale,
            )
            for item in self.all()
        ]

    def source_hashes(self) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for item in self.all():
            if item.kind == EvidenceKind.LOCAL_FILE and item.path and not item.stale:
                hashes[item.path] = item.sha256
        return hashes

    def latest_for_path(self, path: str) -> Evidence | None:
        matches = [item for item in self.all() if item.path == path]
        return matches[-1] if matches else None

    def refresh_staleness(self, project_root: Path) -> list[Evidence]:
        changed: list[Evidence] = []
        current_map = self._load()
        for item in list(current_map.values()):
            if item.kind != EvidenceKind.LOCAL_FILE or not item.path:
                continue
            target = project_root / item.path
            try:
                current = hash_file(target)
            except OSError:
                reason = "source_missing"
                if not item.stale or item.stale_reason != reason:
                    updated = self._replace(item, stale=True, stale_reason=reason)
                    changed.append(updated)
                continue
            if current != item.sha256:
                reason = f"source_changed:{item.sha256[:12]}->{current[:12]}"
                if not item.stale or item.stale_reason != reason:
                    updated = self._replace(item, stale=True, stale_reason=reason)
                    changed.append(updated)
            elif item.stale:
                updated = self._replace(item, stale=False, stale_reason=None)
                changed.append(updated)
        if changed:
            self._save()
        return changed

    def mark_stale(self, evidence_id: str, reason: str) -> Evidence:
        item = self.require(evidence_id)
        updated = self._replace(item, stale=True, stale_reason=reason)
        self._save()
        return updated

    def mark_source_stale(self, source: Evidence, reason: str) -> None:
        """Publish one version's stale flag for all of its captured ranges."""
        current = self._load()
        updated: dict[str, Evidence] = {}
        changed = False
        for key, item in current.items():
            same_source = (item.kind, item.path, item.url, item.sha256) == (
                source.kind, source.path, source.url, source.sha256)
            if same_source and (not item.stale or item.stale_reason != reason):
                item = Evidence.model_validate({**item.model_dump(mode="json"),
                                                "stale": True, "stale_reason": reason})
                changed = True
            updated[key] = item
        if changed:
            atomic_write_json(self.path, EvidenceIndex(evidence=list(updated.values())).model_dump(mode="json"))
            self._cache = updated

    def _replace(self, item: Evidence, **changes: Any) -> Evidence:
        updated = Evidence.model_validate({**item.model_dump(mode="json"), **changes})
        self._load()[item.evidence_id] = updated
        return updated


# ---------------------------------------------------------------------------
# Lessons
# ---------------------------------------------------------------------------


def _lesson_scope(lesson: Lesson, rule_version: str | None) -> str:
    if lesson.status != LessonStatus.VERIFIED:
        return "candidate"
    if lesson.schema_version == 1 or not lesson.applicable_versions:
        return "legacy"
    if rule_version is not None and rule_version not in lesson.applicable_versions:
        return "out_of_scope"
    return "current"


class LessonFile(StrictModel):
    schema_version: Literal[1] = 1
    lessons: list[Lesson] = Field(default_factory=list, max_length=2000)


class LessonStore:
    def __init__(self, output_root: Path) -> None:
        self.path = output_root.resolve() / "lessons.json"

    def load(self) -> list[Lesson]:
        if not self.path.exists():
            return []
        payload = read_json(self.path)
        try:
            return LessonFile.model_validate(payload).lessons
        except ValidationError as exc:
            raise StoreError("lesson store is corrupt; refusing to reset it", detail=str(exc)) from exc

    def save(self, lessons: list[Lesson]) -> None:
        atomic_write_json(self.path, LessonFile(lessons=lessons).model_dump(mode="json"))

    def upsert(self, lesson: Lesson) -> None:
        lessons = {item.lesson_id: item for item in self.load()}
        existing = lessons.get(lesson.lesson_id)
        if existing is not None and existing.status == LessonStatus.VERIFIED and lesson.status == LessonStatus.CANDIDATE:
            # Never downgrade a verified lesson implicitly.
            return
        lessons[lesson.lesson_id] = lesson
        self.save(list(lessons.values()))

    def search(self, query: str, limit: int = 5, *, rule_version: str | None = None) -> list[LessonSummary]:
        words = {word for word in re.findall(r"[A-Za-z_][A-Za-z0-9_.]{2,}", query.lower())}
        scored: list[tuple[int, Lesson]] = []
        for lesson in self.load():
            haystack = f"{lesson.symptom_pattern} {lesson.applicability} {lesson.guidance}".lower()
            score = sum(1 for word in words if word in haystack)
            if words and score == 0:
                continue
            if lesson.status == LessonStatus.VERIFIED:
                score += 100
            elif lesson.status == LessonStatus.DEPRECATED:
                score -= 50
            scored.append((score, lesson))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            LessonSummary(
                lesson_id=lesson.lesson_id,
                status=lesson.status,
                symptom_pattern=lesson.symptom_pattern[:500],
                applicability=lesson.applicability[:500],
                guidance=lesson.guidance[:500],
                verified=lesson.status == LessonStatus.VERIFIED,
                applicable_versions=lesson.applicable_versions[:10],
                scope_status=_lesson_scope(lesson, rule_version),
                verification_contract=(
                    lesson.verification.contract
                    if lesson.verification is not None and lesson.verification.contract
                    else ("v1" if lesson.schema_version == 1 else "v2")
                ),
            )
            for _, lesson in scored[:limit]
        ]


def copy_tree_fixture(source: Path, destination: Path) -> None:
    """Copy a synthetic fixture tree; used by demo and tests only."""
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
