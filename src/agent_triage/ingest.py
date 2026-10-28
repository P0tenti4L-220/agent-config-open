"""Event ingestion, normalization and fingerprints."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .contracts import Event
from .errors import InputError

TEXT_SUFFIXES = {".log", ".txt", ".md"}
EVENT_SUFFIXES = {".json"}
MAX_INPUT_BYTES = 2 * 1024 * 1024
ERROR_TYPE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))\b")
TRACEBACK_RE = re.compile(r"^Traceback \(most recent call last\):?$")
EXCEPTION_LINE_RE = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning)(?::.*)?$"
)
ERROR_LEVEL_RE = re.compile(r"(?:^|\s|\[)(?:ERROR|FATAL|CRITICAL)(?:\]|\b)")
WHITESPACE_RE = re.compile(r"\s+")
HEX_RE = re.compile(r"\b[0-9a-f]{8,}\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+\b")


def load_event(path: Path) -> Event:
    if path.suffix.lower() not in EVENT_SUFFIXES:
        raise InputError(
            "event input must be JSON (.json)",
            detail=f"got suffix {path.suffix or '<none>'}",
        )
    raw = _read_bounded(path)
    try:
        return Event.model_validate_json(raw)
    except ValidationError as exc:
        parts = []
        for error in exc.errors()[:10]:
            location = ".".join(str(item) for item in error["loc"]) or "<root>"
            parts.append(f"{location}: {error['msg']}")
        raise InputError("event does not match the expected shape", detail="; ".join(parts)) from exc


def event_from_symptom_file(
    path: Path,
    project_root: Path,
    *,
    goal: str,
    project_id: str,
    rule_version: str,
    now: datetime | None = None,
) -> Event:
    """Create a minimal event from a UTF-8 log/text file.

    The file must be inside the project root; its relative path becomes
    ``error_files``.  The symptom is selected in this order: the exception
    summary in the last traceback block, the last standalone exception
    summary, the last ERROR/FATAL/CRITICAL line, and finally the first
    non-empty line.  The result is bounded, and no root cause or environment
    value is guessed.
    """
    if path.suffix.lower() not in TEXT_SUFFIXES:
        raise InputError(
            "symptom input must be .log, .txt or .md",
            detail=f"got suffix {path.suffix or '<none>'}",
        )
    root = project_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise InputError("symptom file must be inside the project root") from exc
    raw = _read_bounded(path)
    if b"\x00" in raw[:8192]:
        raise InputError("symptom file looks binary; expected UTF-8 text")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise InputError("symptom file is not valid UTF-8 text") from exc
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise InputError("symptom file contains no non-empty line")
    symptom = _pick_symptom(lines)
    timestamp = now or datetime.now(timezone.utc)
    digest = hashlib.sha256(f"{goal}\x00{symptom}\x00{relative.as_posix()}".encode()).hexdigest()
    return Event(
        schema_version=1,
        event_id=f"evt-{digest[:16]}",
        occurred_at=timestamp,
        project_id=project_id,
        rule_version=rule_version,
        goal=goal,
        symptom=symptom,
        error_files=[relative.as_posix()],
        related_files=[],
        attempts=[],
        source={"kind": "symptom_text"},
        note=(
            "event generated from a local text file; symptom is the last exception summary, "
            "else the last ERROR/FATAL line, else the first non-empty line; the log is kept "
            "as-is and no root cause was inferred"
        ),
    )


def _pick_symptom(lines: list[str]) -> str:
    """Pick a bounded symptom without claiming a root cause.

    ``lines`` are stripped, non-empty lines in their original order.
    """
    # A traceback gives the clearest short summary.  Chained exceptions can
    # contain several blocks; the last block carries the exception that escaped.
    traceback_starts = [index for index, line in enumerate(lines) if TRACEBACK_RE.match(line)]
    if traceback_starts:
        start = traceback_starts[-1]
        summary = None
        for line in lines[start + 1 :]:
            if TRACEBACK_RE.match(line):
                break
            if EXCEPTION_LINE_RE.match(line):
                summary = line
        if summary:
            return summary[:2000]

    # Logs may record an exception summary without a traceback.
    for line in reversed(lines):
        if EXCEPTION_LINE_RE.match(line):
            return line[:2000]

    # Fall back to an error-level record before using arbitrary first-line text.
    for line in reversed(lines):
        if ERROR_LEVEL_RE.search(line):
            return line[:2000]

    return lines[0][:2000]


def _read_bounded(path: Path) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise InputError(f"cannot stat input file: {path.name}", detail=str(exc)) from exc
    if size > MAX_INPUT_BYTES:
        raise InputError(
            "input file exceeds 2 MiB; select a smaller excerpt instead of reading it whole",
            detail=f"{path.name} is {size} bytes",
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise InputError(f"cannot read input file: {path.name}", detail=str(exc)) from exc


def normalize_symptom(symptom: str) -> str:
    text = WHITESPACE_RE.sub(" ", symptom.strip().lower())
    text = HEX_RE.sub("<hex>", text)
    text = NUMBER_RE.sub("<n>", text)
    return text[:1000]


def error_type_of(symptom: str) -> str:
    match = ERROR_TYPE_RE.search(symptom)
    return match.group(1) if match else "unknown"


def primary_component(event: Event) -> str:
    candidates = [*event.error_files, *event.related_files]
    if not candidates:
        return "unknown"
    name = candidates[0].rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0] or name


def fingerprint(event: Event) -> dict[str, str]:
    """Stable, project-independent problem fingerprint.

    A fingerprint is a *candidate* signal only.  It never proves that two
    events share a root cause.
    """
    payload = {
        "algorithm": "fp1",
        "symptom": normalize_symptom(event.symptom),
        "error_type": error_type_of(event.symptom),
        "component": primary_component(event).lower(),
    }
    canonical = repr(sorted(payload.items())).encode("utf-8")
    return {"algorithm": "fp1", "value": hashlib.sha256(canonical).hexdigest()}


def keywords_for_event(event: Event) -> list[str]:
    words: list[str] = []
    for text in (event.symptom, event.goal):
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.]{2,}", text.lower()):
            if token not in words:
                words.append(token)
    return words[:12]


def event_to_jsonable(event: Event) -> dict[str, Any]:
    return event.model_dump(mode="json")
