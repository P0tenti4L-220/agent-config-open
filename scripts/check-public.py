#!/usr/bin/env python3
"""Check staged public content without printing matched secret values.

This is a small publishing aid, not a complete secret scanner. Run after
staging explicit paths; ignored local data is never opened by this script.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {
    "local", "raw", "archive", "backups", "state", "outbox", "work",
    ".codex", ".claude", ".agents", ".aws", ".workbuddy-ai", ".venv",
    "__pycache__", "node_modules",
}
FORBIDDEN_NAMES = {"auth.json", "credentials.json", ".credentials.json", ".netrc"}
FORBIDDEN_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".db", ".sqlite", ".log"}
PATTERNS = {
    "credential": re.compile(
        r"(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,}"
        r"|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}"
        r"|Bearer\s+[A-Za-z0-9._-]{24,})"
    ),
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "personal-path": re.compile(r"/(?:Users|home)/([^/\s`'\"<>]+)"),
    "private-host": re.compile(
        r"https?://(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+"
        r"|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|[\w.-]+\.(?:internal|local))\b"
    ),
}
# Explicit deterministic literals used by adversarial redaction fixtures.
# Do not suppress whole test files: new suspicious literals must still fail.
SYNTHETIC_CREDENTIALS = {
    "sk-abcdefghijklmnopqrstuvwxyz",
    "ghp_abcdefghijklmnopqrstuvwxyz",
    "sk-test-should-never-appear",
}
SYNTHETIC_USERS = {"example", "YOUR_USER", "<user>"}


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.PIPE)


def main() -> int:
    try:
        entries = git("ls-files", "--stage", "-z").decode("utf-8").split("\0")
    except (subprocess.CalledProcessError, UnicodeDecodeError):
        print("error: run inside an initialized Git checkout after staging explicit paths")
        return 2
    problems: list[str] = []
    checked = 0
    for entry in filter(None, entries):
        metadata, name = entry.split("\t", 1)
        mode, _, stage = metadata.split()
        path = PurePosixPath(name)
        if stage != "0" or mode not in {"100644", "100755"}:
            problems.append(f"{name}: forbidden Git entry mode or unresolved conflict")
            continue
        if (
            FORBIDDEN_PARTS.intersection(path.parts)
            or path.name in FORBIDDEN_NAMES
            or path.suffix in FORBIDDEN_SUFFIXES
            or (path.name.startswith(".env") and path.name != ".env.example")
            or path.suffix == ".env"
        ):
            problems.append(f"{name}: forbidden public file path")
            continue
        try:
            content = git("show", f":{name}").decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"{name}: binary content requires a separate publishing review")
            continue
        checked += 1
        for line_no, line in enumerate(content.splitlines(), 1):
            for label, pattern in PATTERNS.items():
                for match in pattern.finditer(line):
                    if label == "credential" and match.group() in SYNTHETIC_CREDENTIALS:
                        continue
                    if label == "personal-path" and match.group(1) in SYNTHETIC_USERS:
                        continue
                    problems.append(f"{name}:{line_no}: potential {label}")
    if not checked:
        problems.append("no staged text files found")
    if problems:
        print("Public-content check failed (matching values are withheld):")
        print("\n".join(sorted(set(problems))))
        return 1
    print(f"Checked {checked} staged text files; no configured pattern matched.")
    print("Manual review is still required; this is not a complete secret scan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
