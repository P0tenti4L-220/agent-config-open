"""Deterministic lexical selection of exact, independently verifiable slices.

No stemming, decorated highlights or generated summaries enter evidence text.
Offsets always refer to UTF-8 bytes in the named original representation.
"""

from __future__ import annotations

import hashlib
import json
import re
from itertools import islice

from .contracts import Evidence, EvidenceWindow

MAX_WINDOW_BYTES = 8000  # also fits the historical 8000-character proof cap
_WORDS = re.compile(r"[A-Za-z_][A-Za-z0-9_./'-]{1,119}|[\u3400-\u9fff]{2,}")
_STOP = {"the", "and", "for", "from", "with", "this", "that", "read", "file", "what", "does", "why"}


def query_terms(*texts: str) -> list[str]:
    terms: list[str] = []
    for text in texts:
        for match in _WORDS.finditer(text):
            word = match.group()
            candidates = [word]
            if re.fullmatch(r"[\u3400-\u9fff]{2,}", word):
                candidates.extend(word[i:i + 2] for i in range(len(word) - 1))
            for value in candidates:
                value = value.lower()
                if value not in _STOP and value not in terms:
                    terms.append(value)
    return terms[:64]


def relevance(text: str, terms: list[str]) -> int:
    return sum(bool(re.search(re.escape(term), text, re.IGNORECASE)) for term in terms)


def window_identity(window: EvidenceWindow) -> str:
    fields = [window.evidence_id, window.source_sha256, window.snapshot_sha256,
              window.span_origin, window.start_byte, window.end_byte, window.text_sha256]
    encoded = json.dumps(fields, separators=(",", ":")).encode()
    return "win-" + hashlib.sha256(encoded).hexdigest()[:20]


def make_window(item: Evidence, start: int, end: int, reason: str) -> EvidenceWindow:
    raw = item.snippet.encode("utf-8")
    fragment = raw[start:end].decode("utf-8")
    line_start = (item.line_start or 1) + raw[:start].count(b"\n")
    line_end = line_start + fragment.count("\n") - int(fragment.endswith("\n"))
    if item.snippet.lstrip().startswith(("{", "[")) and (item.truncated or start > 0 or end < len(raw)):
        reason += "; partial JSON text, not a complete object"
    values = dict(
        window_id="win-" + "0" * 20, evidence_id=item.evidence_id,
        source_sha256=item.sha256, snapshot_sha256=item.snapshot_sha256,
        span_origin="evidence_snippet", start_byte=start, end_byte=end,
        line_start=line_start, line_end=line_end, text=fragment,
        text_sha256=hashlib.sha256(raw[start:end]).hexdigest(), selection_reason=reason,
    )
    window = EvidenceWindow(**values)
    return window.model_copy(update={"window_id": window_identity(window)})


def verify_window(item: Evidence, window: EvidenceWindow, snapshot: str | None = None) -> bool:
    if (window.evidence_id != item.evidence_id or window.source_sha256 != item.sha256
            or window.snapshot_sha256 != item.snapshot_sha256
            or window.window_id != window_identity(window)):
        return False
    original = item.snippet if window.span_origin == "evidence_snippet" else snapshot
    if original is None:
        return False
    raw = original.encode("utf-8")
    if not 0 <= window.start_byte < window.end_byte <= len(raw):
        return False
    try:
        raw[:window.start_byte].decode("utf-8")
        fragment = raw[window.start_byte:window.end_byte].decode("utf-8")
    except UnicodeDecodeError:
        return False
    line_start = (item.line_start or 1) + raw[:window.start_byte].count(b"\n")
    line_end = line_start + fragment.count("\n") - int(fragment.endswith("\n"))
    return (fragment == window.text and hashlib.sha256(fragment.encode()).hexdigest() == window.text_sha256
            and window.line_start == line_start and window.line_end == line_end)


def _bounded_span(text: str, anchor: int, end_anchor: int, byte_limit: int) -> tuple[int, int]:
    """Keep adjacent complete lines when they fit, otherwise a literal local slice."""
    line_start = text.rfind("\n", 0, anchor) + 1
    start = text.rfind("\n", 0, max(0, line_start - 1)) + 1
    line_end = text.find("\n", end_anchor)
    end = len(text) if line_end < 0 else line_end + 1
    next_end = text.find("\n", end)
    end = len(text) if next_end < 0 else next_end + 1
    if len(text[start:end].encode("utf-8")) <= byte_limit:
        return start, end
    # A single long line still has a precise byte position; no line is skipped.
    start = anchor
    end = min(len(text), max(end_anchor, anchor + 1))
    while len(text[start:end].encode("utf-8")) > byte_limit and end > start:
        end -= 1
    left = max(0, byte_limit - len(text[start:end].encode("utf-8"))) // 2
    while start > 0 and len(text[start - 1:anchor].encode("utf-8")) <= left:
        start -= 1
    while end < len(text) and len(text[start:end + 1].encode("utf-8")) <= byte_limit:
        end += 1
    return start, end


def select_windows(item: Evidence, terms: list[str], max_bytes: int) -> list[EvidenceWindow]:
    text = item.snippet
    if not text or max_bytes < 4:
        return []
    max_bytes = min(max_bytes, 2 * MAX_WINDOW_BYTES)
    if len(text.encode("utf-8")) <= min(max_bytes, MAX_WINDOW_BYTES):
        return [make_window(item, 0, len(text.encode("utf-8")), "complete snippet")]
    hits: list[tuple[int, int, str]] = []
    for term in terms:
        for match in islice(re.finditer(re.escape(term), text, re.IGNORECASE), 32):
            hits.append((match.start(), match.end(), term))
    reason = "query match with adjacent context" if hits else "fallback_preview; no lexical match"
    if hits:
        candidates = []
        for start, end, term in hits:
            span = _bounded_span(text, start, end, min(max_bytes, MAX_WINDOW_BYTES))
            score = relevance(text[span[0]:span[1]], terms)
            candidates.append((-score, -len(term), start, end))
        candidates.sort()
        anchors = [(candidates[0][2], candidates[0][3])]
        covered = _bounded_span(text, *anchors[0], min(max_bytes, MAX_WINDOW_BYTES))
        for _score, _specificity, start, end in candidates[1:]:
            if end <= covered[0] or start >= covered[1]:
                anchors.append((start, end))
                break
    else:
        anchors = [(0, 1), (max(0, len(text) - 1), len(text))]
    per_window = min(MAX_WINDOW_BYTES, max_bytes // len(anchors))
    spans = sorted(_bounded_span(text, start, end, per_window) for start, end in anchors)
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if end <= start:
            continue
        if merged and start <= merged[-1][1] and len(text[merged[-1][0]:end].encode()) <= MAX_WINDOW_BYTES:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return [make_window(item, len(text[:start].encode()), len(text[:end].encode()), reason)
            for start, end in merged[:2]]


def render_window(window: EvidenceWindow, *, include_cursor: bool = False) -> str:
    return (f"[{window.evidence_id}] {window.window_id} "
            f"bytes=[{window.start_byte},{window.end_byte}) "
            f"lines={window.line_start}-{window.line_end} {window.selection_reason}"
            + (f" read_evidence_cursor={window.read_cursor}" if include_cursor and window.read_cursor else "")
            + "\n" + window.text)
