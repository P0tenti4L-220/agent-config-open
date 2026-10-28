"""Provider-request snapshots and evidence display provenance.

The runner used to mark evidence as "displayed" as soon as a tool returned it.
That is not the same thing as the model seeing the body: the next context build
may drop the whole tool call/result pair, or truncate the evidence layer.  This
module records the actual payload handed to the provider, derives the exact
evidence fragments present in it, and only turns those into display records
after a response was received.

A request that raised before a response is kept as ``attempted``/``failed`` and
never yields display proof.  The fragment itself is stored (bounded) so the
recorded hash can be recomputed later instead of trusting an opaque digest.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import Field, ValidationError, model_validator

from .contracts import (
    DisplayedEvidence,
    DisplaySource,
    Evidence,
    EvidenceId,
    Sha256,
    StrictInt,
    StrictModel,
)
from .errors import InputError, ValidationFailure
from .store import atomic_write_json, read_json, utcnow

REQUEST_ID_RE = re.compile(r"^req-[0-9]{6}$")
MAX_FRAGMENT_CHARS = 8000
MAX_SNAPSHOT_BYTES = 512 * 1024
MIN_TRUNCATED_FRAGMENT = 16
_JSON_STRING_RE = re.compile(r'"((?:[^"\\]|\\.){16,})"')


class EvidenceView(StrictModel):
    """One exact fragment of an evidence item that was in the request."""

    evidence_id: EvidenceId
    source_sha256: Sha256
    source: DisplaySource
    mode: Literal["full", "truncated"]
    visible_chars: StrictInt = Field(ge=0, le=1_000_000)
    visible_sha256: Sha256
    fragment: str = Field(min_length=1, max_length=MAX_FRAGMENT_CHARS)

    @model_validator(mode="after")
    def _hash_matches_fragment(self) -> "EvidenceView":
        if self.visible_chars != len(self.fragment):
            raise ValueError("visible_chars must equal the stored fragment length")
        if self.visible_sha256 != _sha256_text(self.fragment):
            raise ValueError("visible_sha256 must hash the stored fragment")
        return self

    def to_displayed(self, *, turn: int, request_id: str) -> DisplayedEvidence:
        return DisplayedEvidence(
            schema_version=2,
            evidence_id=self.evidence_id,
            turn=turn,
            source=self.source,
            mode=self.mode,
            visible_chars=self.visible_chars,
            visible_sha256=self.visible_sha256,
            request_id=request_id,
            source_sha256=self.source_sha256,
        )


class RequestView(StrictModel):
    schema_version: Literal[2] = 2
    request_id: str
    run_id: str
    turn: StrictInt = Field(ge=0)
    provider: str
    status: Literal["attempted", "response_received", "failed"]
    payload_sha256: Sha256
    payload_chars: StrictInt = Field(ge=0)
    payload_complete: bool = True
    system: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_ids: list[str] = Field(default_factory=list, max_length=200)
    tool_result_ids: list[str] = Field(default_factory=list, max_length=200)
    evidence_views: list[EvidenceView] = Field(default_factory=list, max_length=400)
    created_at: datetime
    responded_at: datetime | None = None

    @model_validator(mode="after")
    def _response_shape(self) -> "RequestView":
        if self.status == "response_received" and self.responded_at is None:
            raise ValueError("a received response must record responded_at")
        if self.status != "response_received" and self.responded_at is not None:
            raise ValueError("only received responses record responded_at")
        return self


class RequestViewStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.directory = self.run_dir / "request-views"

    def begin(
        self,
        *,
        run_id: str,
        turn: int,
        provider: str,
        system: str,
        messages: list[dict[str, Any]],
        evidence: list[Evidence],
    ) -> RequestView:
        self.directory.mkdir(parents=True, exist_ok=True)
        existing = [
            int(path.stem.split("-", 1)[1])
            for path in self.directory.glob("req-*.json")
            if REQUEST_ID_RE.fullmatch(path.stem)
        ]
        sequence = max(existing, default=0) + 1
        request_id = f"req-{sequence:06d}"
        payload = {"system": system, "messages": messages}
        payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        payload_bytes = payload_text.encode("utf-8")
        complete = len(payload_bytes) <= MAX_SNAPSHOT_BYTES
        snapshot_system = system
        snapshot_messages = messages
        if not complete:
            # Keep a bounded, honest snapshot.  The evidence fragments below are
            # still derived from the real payload before this cap was applied.
            snapshot_system = system[: MAX_SNAPSHOT_BYTES // 2]
            snapshot_messages = [
                {"role": "user", "content": [{"type": "text", "text": "[snapshot omitted: payload exceeded bounded snapshot size]"}]}
            ]
        views = derive_evidence_views(evidence, system=system, messages=messages)
        view = RequestView(
            request_id=request_id,
            run_id=run_id,
            turn=turn,
            provider=provider,
            status="attempted",
            payload_sha256=hashlib.sha256(payload_bytes).hexdigest(),
            payload_chars=len(payload_text),
            payload_complete=complete,
            system=snapshot_system,
            messages=snapshot_messages,
            tool_call_ids=_tool_call_ids(messages),
            tool_result_ids=_tool_result_ids(messages),
            evidence_views=views,
            created_at=utcnow(),
        )
        self._write(view)
        return view

    def mark_response_received(self, request_id: str) -> RequestView:
        view = self.load(request_id)
        if view.status != "attempted":
            return view
        updated = RequestView.model_validate(
            {
                **view.model_dump(mode="json"),
                "status": "response_received",
                "responded_at": utcnow().isoformat(),
            }
        )
        self._write(updated)
        return updated

    def mark_failed(self, request_id: str) -> RequestView:
        view = self.load(request_id)
        if view.status != "attempted":
            return view
        updated = RequestView.model_validate({**view.model_dump(mode="json"), "status": "failed"})
        self._write(updated)
        return updated

    def load(self, request_id: str) -> RequestView:
        path = self.directory / f"{request_id}.json"
        if not path.is_file():
            raise InputError(f"request view not found: {request_id}")
        try:
            return RequestView.model_validate(read_json(path))
        except (InputError, ValidationError) as exc:
            raise ValidationFailure(f"request view is invalid: {request_id}", detail=str(exc)) from exc

    def all(self) -> list[RequestView]:
        if not self.directory.is_dir():
            return []
        views: list[RequestView] = []
        for path in sorted(self.directory.glob("req-*.json")):
            try:
                views.append(RequestView.model_validate(read_json(path)))
            except (InputError, ValidationError) as exc:
                raise ValidationFailure(f"request view is invalid: {path.name}", detail=str(exc)) from exc
        return views

    def confirmed_displayed(self, view: RequestView) -> list[DisplayedEvidence]:
        if view.status != "response_received":
            return []
        return [item.to_displayed(turn=view.turn, request_id=view.request_id) for item in view.evidence_views]

    def _write(self, view: RequestView) -> None:
        atomic_write_json(self.directory / f"{view.request_id}.json", view.model_dump(mode="json"))


def load_request_views(run_dir: Path) -> list[RequestView]:
    return RequestViewStore(run_dir).all()


def derive_evidence_views(
    evidence: Iterable[Evidence],
    *,
    system: str,
    messages: list[dict[str, Any]],
) -> list[EvidenceView]:
    """Return only evidence bodies/fragments present in the submitted payload."""
    message_leaves = _leaf_strings(messages)
    leaves = [system, *message_leaves]
    blob = "\n".join(leaf for leaf in leaves if leaf)
    views: list[EvidenceView] = []
    for item in evidence:
        match = _match_evidence(item, system=system, blob=blob, leaves=leaves)
        if match is None:
            continue
        fragment, mode, source = match
        views.append(
            EvidenceView(
                evidence_id=item.evidence_id,
                source_sha256=item.sha256,
                source=source,
                mode=mode,
                visible_chars=len(fragment),
                visible_sha256=_sha256_text(fragment),
                fragment=fragment,
            )
        )
    return views


def verify_display_records(
    displayed: list[DisplayedEvidence] | None,
    views: list[RequestView],
    evidence: list[Evidence],
) -> tuple[list[str], Literal["none", "legacy", "request_view"]]:
    """Recompute display proof from stored request views.

    Returns problems and whether the records are legacy (v1), request-view
    (v2), or absent.  A legacy record is readable but is not proof that the
    body entered a provider request.
    """
    records = list(displayed or [])
    if not records:
        return [], "none"
    v2 = [record for record in records if record.schema_version >= 2]
    if not v2:
        return [], "legacy"
    known = {item.evidence_id: item for item in evidence}
    by_request = {item.request_id: item for item in views}
    problems: list[str] = []
    for record in v2:
        item = known.get(record.evidence_id)
        if item is None:
            problems.append(f"display record cites unknown evidence {record.evidence_id}")
            continue
        if record.source_sha256 != item.sha256:
            problems.append(
                f"display source hash for {record.evidence_id} does not match the stored evidence"
            )
        if not record.request_id:
            problems.append(f"display record for {record.evidence_id} has no request id")
            continue
        view = by_request.get(record.request_id)
        if view is None:
            problems.append(f"display record references missing request {record.request_id}")
            continue
        if view.status != "response_received":
            problems.append(
                f"request {record.request_id} for {record.evidence_id} did not receive a response"
            )
            continue
        if not view.payload_complete:
            problems.append(
                f"request {record.request_id} has an incomplete payload snapshot; "
                f"the display of {record.evidence_id} cannot be recomputed"
            )
            continue
        if view.payload_complete and view.payload_sha256 != _payload_sha256(view):
            problems.append(
                f"request {record.request_id} payload hash cannot be recomputed from its snapshot"
            )
            continue
        if record.turn != view.turn:
            problems.append(
                f"display record for {record.evidence_id} names turn {record.turn}, "
                f"but request {record.request_id} is turn {view.turn}"
            )
            continue
        matches = [
            candidate
            for candidate in view.evidence_views
            if candidate.evidence_id == record.evidence_id
            and candidate.visible_sha256 == record.visible_sha256
            and candidate.source_sha256 == record.source_sha256
            and candidate.source == record.source
            and candidate.mode == record.mode
        ]
        if not matches:
            problems.append(
                f"request {record.request_id} does not contain the recorded view of {record.evidence_id}"
            )
            continue
        candidate = matches[0]
        if candidate.visible_chars != record.visible_chars:
            problems.append(f"display width changed for {record.evidence_id}")
        if candidate.visible_sha256 != _sha256_text(candidate.fragment):
            problems.append(f"display fragment hash cannot be recomputed for {record.evidence_id}")
        if view.payload_complete and not _fragment_in_payload(candidate.fragment, view):
            problems.append(
                f"display fragment for {record.evidence_id} is not present in request {record.request_id}"
            )
    return problems, "request_view"


def _payload_sha256(view: RequestView) -> str:
    payload = {"system": view.system, "messages": view.messages}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(text)


def _fragment_in_payload(fragment: str, view: RequestView) -> bool:
    blob = "\n".join([view.system, *_leaf_strings(view.messages)])
    return _fragment_present(fragment, blob)


def _fragment_present(fragment: str, blob: str) -> bool:
    if not fragment or not blob:
        return False
    if fragment in blob:
        return True
    lines = [line for line in fragment.splitlines() if line.strip()]
    if not lines:
        return False
    cursor = 0
    for line in lines:
        position = blob.find(line, cursor)
        if position == -1:
            return False
        cursor = position + len(line)
    return True


def _match_evidence(
    item: Evidence,
    *,
    system: str,
    blob: str,
    leaves: list[str],
) -> tuple[str, Literal["full", "truncated"], DisplaySource] | None:
    snippet = item.snippet
    if not snippet.strip() or not blob:
        return None
    if _fragment_present(snippet, blob):
        fragment = _bounded(snippet)
        mode = "full" if fragment == snippet else "truncated"
        return fragment, mode, _source(fragment, system)
    # The whole snippet may be split across JSON leaves (for example the lines
    # of a tool result).  If every line is present in order, it was visible.
    lines = [line for line in snippet.splitlines() if line.strip()]
    if lines and _lines_in_order(lines, blob):
        fragment = _bounded(snippet)
        mode = "full" if fragment == snippet else "truncated"
        return fragment, mode, _source(fragment, system)
    # Otherwise take the longest fragment of the snippet that is present.  A
    # bounded system prompt truncates at the end, so a prefix is the normal
    # truncation shape; single JSON values cover issue bodies and search hits.
    candidates = sorted(_candidate_fragments(snippet), key=len, reverse=True)
    for candidate in candidates:
        if len(candidate) < MIN_TRUNCATED_FRAGMENT:
            continue
        if _fragment_present(candidate, blob):
            return _bounded(candidate), "truncated", _source(candidate, system)
    return None


def _lines_in_order(lines: list[str], blob: str) -> bool:
    cursor = 0
    for line in lines:
        position = blob.find(line, cursor)
        if position == -1:
            return False
        cursor = position + len(line)
    return True


def _candidate_fragments(snippet: str) -> set[str]:
    candidates: set[str] = set()
    for line in snippet.splitlines():
        if len(line.strip()) >= MIN_TRUNCATED_FRAGMENT:
            candidates.add(line)
    for match in _JSON_STRING_RE.finditer(snippet):
        try:
            decoded = json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, str) and len(decoded) >= MIN_TRUNCATED_FRAGMENT:
            candidates.add(decoded)
    return candidates


def _source(fragment: str, system: str) -> DisplaySource:
    return DisplaySource.CONTEXT if fragment and fragment in system else DisplaySource.TOOL_RESULT


def _bounded(text: str) -> str:
    if len(text) <= MAX_FRAGMENT_CHARS:
        return text
    return text[:MAX_FRAGMENT_CHARS]


def _leaf_strings(value: Any, *, depth: int = 0) -> list[str]:
    """Flatten strings, following JSON-in-string content used by tool results."""
    if depth > 5:
        return []
    leaves: list[str] = []
    if isinstance(value, str):
        if not value:
            return leaves
        leaves.append(value)
        stripped = value.strip()
        if len(stripped) <= 200_000 and stripped[:1] in "{[":
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if isinstance(parsed, (dict, list)):
                leaves.extend(_leaf_strings(parsed, depth=depth + 1))
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in {"system", "messages"}:
                leaves.extend(_leaf_strings(item, depth=depth + 1))
            else:
                leaves.extend(_leaf_strings(item, depth=depth + 1))
    elif isinstance(value, (list, tuple)):
        for item in value:
            leaves.extend(_leaf_strings(item, depth=depth + 1))
    return leaves


def _tool_call_ids(messages: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id"):
                ids.append(str(block["id"]))
    return ids


def _tool_result_ids(messages: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id"):
                ids.append(str(block["tool_use_id"]))
    return ids


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()