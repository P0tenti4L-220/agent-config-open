"""Configuration loading and non-secret fingerprinting."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

from pydantic import ValidationError

from .contracts import Config
from .errors import ConfigError

DEFAULT_CONFIG = Config()


def load_config(path: Path | None) -> Config:
    if path is None:
        return DEFAULT_CONFIG.model_copy(deep=True)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read config: {path.name}", detail=str(exc)) from exc
    if len(raw) > 256 * 1024:
        raise ConfigError("config file is larger than 256 KiB")
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError("config is not valid UTF-8 TOML", detail=str(exc)) from exc
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        raise ConfigError("config does not match the expected shape", detail=_format_validation(exc)) from exc


def config_fingerprint(config: Config) -> str:
    """Hash the effective configuration; credential values are never part of it."""
    canonical = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _format_validation(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors()[:8]:
        location = ".".join(str(item) for item in error["loc"]) or "<root>"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)