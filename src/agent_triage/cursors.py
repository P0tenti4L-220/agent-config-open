"""Run/source/object-bound byte cursors. They grant no additional access."""

from __future__ import annotations

import base64
import hashlib
import json

from .contracts import Evidence
from .errors import ToolFailure


def _binding(run_id: str, item: Evidence, offset: int) -> dict:
    digest = item.snapshot_sha256 or hashlib.sha256(item.snippet.encode()).hexdigest()
    return {"v": 1, "run": run_id, "evidence": item.evidence_id,
            "source": item.sha256, "snapshot": digest, "offset": offset}


def make_cursor(run_id: str, item: Evidence, offset: int) -> str:
    return base64.urlsafe_b64encode(json.dumps(_binding(run_id, item, offset),
                                               separators=(",", ":")).encode()).decode().rstrip("=")


def read_cursor(cursor: str, run_id: str, item: Evidence, raw: bytes) -> int:
    try:
        payload = json.loads(base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True))
        offset = payload["offset"]
        if type(offset) is not int or payload != _binding(run_id, item, offset):
            raise ValueError("binding mismatch")
        if not 0 <= offset < len(raw):
            raise ValueError("offset out of range")
        raw[:offset].decode("utf-8")
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise ToolFailure("invalid_cursor", "cursor must name this run, source version, snapshot and a valid UTF-8 offset") from exc
    return offset
