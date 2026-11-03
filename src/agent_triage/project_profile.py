"""Bounded, sourced project profile for issue research.

Only the project's own manifests are read.  The agent never executes project
code, installs dependencies or inspects the full interpreter environment, so
every version keeps its source (declared, locked, observed) and anything that
could not be established stays ``None`` with a gap entry.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path
from typing import Any

from .contracts import (
    Event,
    IssueResearchConfig,
    ProfileDependency,
    ProjectProfile,
)
from .errors import ToolFailure
from .github_scope import normalize_repo
from .tools import Workspace

MAX_MANIFEST_BYTES = 512 * 1024
_DEP_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)(\[[^\]]*\])?")
_REQ_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?\s*(.*)$")


def build_project_profile(
    *,
    workspace: Workspace,
    config: IssueResearchConfig,
    event: Event,
    project_repo: str | None,
) -> ProjectProfile:
    gaps: list[str] = []
    source_hashes: dict[str, str] = {}
    declared: dict[str, str] = {}
    locked: dict[str, str] = {}
    language: str | None = None

    pyproject = _read_text(workspace, "pyproject.toml", source_hashes)
    if pyproject is not None:
        language = "python"
        try:
            payload = tomllib.loads(pyproject)
        except tomllib.TOMLDecodeError:
            gaps.append("pyproject.toml could not be parsed; declared dependencies are unknown")
        else:
            declared.update(_pyproject_dependencies(payload, gaps))

    for name in _requirement_files(workspace):
        text = _read_text(workspace, name, source_hashes)
        if text is None:
            continue
        language = language or "python"
        for package, spec in _requirements_entries(text):
            declared.setdefault(package, spec or "*")

    uv_lock = _read_text(workspace, "uv.lock", source_hashes)
    if uv_lock is not None:
        language = language or "python"
        try:
            payload = tomllib.loads(uv_lock)
        except tomllib.TOMLDecodeError:
            gaps.append("uv.lock could not be parsed; locked versions are unknown")
        else:
            for package in payload.get("package", []) if isinstance(payload.get("package"), list) else []:
                if not isinstance(package, dict):
                    continue
                name = package.get("name")
                version = package.get("version")
                if isinstance(name, str) and isinstance(version, str):
                    locked.setdefault(_normalize_package(name), version[:200])

    if language is None and any(workspace.root.glob("*.py")):
        language = "python"

    mappings = {item.package.lower(): item for item in config.dependencies}
    dependencies: list[ProfileDependency] = []
    all_packages = list(dict.fromkeys([*declared, *locked]))
    for package in sorted(all_packages):
        mapping = mappings.get(package)
        notes: list[str] = []
        if mapping is None and config.mode == "layered":
            gaps.append(f"dependency_repo_mapping_missing:{package}"[:500])
        if mapping is not None and mapping.package.lower() != package:
            notes.append("mapped through a case-insensitive package name")
        dependencies.append(
            ProfileDependency(
                ecosystem=mapping.ecosystem if mapping else "pypi",
                package=package[:500],
                import_names=list(mapping.import_names) if mapping else [],
                upstream_repo=mapping.repo if mapping else None,
                declared=declared.get(package),
                locked=locked.get(package),
                observed=None,
                notes=notes,
            )
        )

    if declared and not locked:
        gaps.append("lockfile missing or unreadable; installed versions are not established")
    for package in declared:
        if package in locked:
            declared_spec = declared[package]
            locked_version = locked[package]
            if declared_spec not in {"*", ""} and locked_version not in declared_spec:
                # A specifier like ">=1.2,<2" is not a single version; only record
                # the mismatch when the specifier looks like an exact pin.
                if re.fullmatch(r"==?[0-9][^\s,]*", declared_spec):
                    notes_text = f"declared/locked version disagree for {package}"
                    if notes_text not in gaps:
                        gaps.append(notes_text[:500])

    remote, remote_gap = _local_remote(workspace)
    if remote_gap:
        gaps.append(remote_gap)
    repo, repo_source = _resolve_repo(project_repo, remote, gaps)

    runtime_version = event.environment.runtime_version
    if language is None:
        gaps.append("project language could not be established from manifests")
    elif not runtime_version:
        gaps.append("runtime_version not provided by the event; interpreter version is unknown")

    return ProjectProfile(
        project_repo=repo,
        repo_source=repo_source,
        local_remote=remote,
        language=language,
        runtime_version=runtime_version,
        os_name=None,
        architecture=None,
        dependencies=dependencies,
        source_hashes=source_hashes,
        gaps=list(dict.fromkeys(gaps))[:20],
    )


def related_dependencies(
    profile: ProjectProfile,
    symptom: str,
    *,
    limit: int,
) -> list[ProfileDependency]:
    """Pick mapped dependencies mentioned by the symptom, never inventing one."""

    haystack = symptom.casefold()
    ranked: list[tuple[int, ProfileDependency]] = []
    for dependency in profile.dependencies:
        if not dependency.upstream_repo:
            continue
        names = [dependency.package, *dependency.import_names]
        score = 0
        for name in names:
            needle = name.casefold()
            if needle and needle in haystack:
                score += 2 if needle == dependency.package.casefold() else 1
        if score:
            ranked.append((-score, dependency))
    ranked.sort(key=lambda row: (row[0], row[1].package.casefold()))
    return [dependency for _score, dependency in ranked[:limit]]


def _read_text(workspace: Workspace, relative: str, hashes: dict[str, str]) -> str | None:
    path = workspace.root / relative
    if not path.is_file():
        return None
    try:
        payload, text = workspace.read_bytes(relative)
    except ToolFailure:
        return None
    if len(payload) > MAX_MANIFEST_BYTES:
        return None
    hashes[relative] = hashlib.sha256(payload).hexdigest()
    return text


def _requirement_files(workspace: Workspace) -> list[str]:
    names = ["requirements.txt"]
    for path in sorted(workspace.root.glob("requirements-*.txt")):
        if path.is_file():
            names.append(path.name)
    return names


def _pyproject_dependencies(payload: dict[str, Any], gaps: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    project = payload.get("project")
    if isinstance(project, dict):
        for entry in project.get("dependencies", []) if isinstance(project.get("dependencies"), list) else []:
            if isinstance(entry, str):
                package, spec = _parse_requirement(entry)
                if package:
                    found.setdefault(package, spec)
        optional = project.get("optional-dependencies")
        if isinstance(optional, dict):
            gaps.append("optional dependency groups are not treated as installed")
    poetry = payload.get("tool", {}).get("poetry", {}) if isinstance(payload.get("tool"), dict) else {}
    poetry_dependencies = poetry.get("dependencies") if isinstance(poetry, dict) else None
    if isinstance(poetry_dependencies, dict):
        for name, spec in poetry_dependencies.items():
            if name.lower() == "python":
                continue
            if isinstance(spec, str):
                found.setdefault(_normalize_package(str(name)), spec[:200])
            elif isinstance(spec, dict) and isinstance(spec.get("version"), str):
                found.setdefault(_normalize_package(str(name)), spec["version"][:200])
    return found


def _requirements_entries(text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "http:", "https:", "git+")):
            continue
        match = _REQ_RE.match(line)
        if not match:
            continue
        entries.append((_normalize_package(match.group(1)), match.group(2).strip()[:200] or "*"))
    return entries


def _parse_requirement(value: str) -> tuple[str | None, str]:
    value = value.split(";", 1)[0].strip()
    match = _DEP_RE.match(value)
    if not match:
        return None, "*"
    spec = value[match.end():].strip()
    return _normalize_package(match.group(1)), spec[:200] or "*"


def _normalize_package(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _local_remote(workspace: Workspace) -> tuple[str | None, str | None]:
    path: Path = workspace.root / ".git" / "config"
    if not path.is_file():
        return None, None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:256_000]
    except OSError:
        return None, "local git config could not be read; repository identity is unverified"
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            current = stripped
            continue
        if current and current.startswith('[remote "') and stripped.startswith("url ="):
            raw = stripped.split("=", 1)[1].strip()
            normalized = normalize_repo(raw)
            if normalized:
                return normalized, None
    return None, None


def _resolve_repo(
    configured: str | None,
    remote: str | None,
    gaps: list[str],
) -> tuple[str | None, str]:
    configured_repo = normalize_repo(configured)
    if not configured_repo:
        if remote:
            gaps.append("mcp.repo is not configured; local remote is not used as an implicit target")
        return None, "unresolved"
    if not remote:
        return configured_repo, "config"
    if remote.lower() == configured_repo.lower():
        return configured_repo, "config_remote_match"
    gaps.append(
        f"configured repository differs from the local remote ({remote}); "
        "searches stay on the configured repository"
    )
    return configured_repo, "config_remote_mismatch"