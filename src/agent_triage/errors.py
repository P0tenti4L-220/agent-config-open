"""Typed errors with stable CLI exit codes."""

from __future__ import annotations


class TriageError(Exception):
    """Base class for expected, reportable failures."""

    exit_code = 4

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def as_dict(self) -> dict[str, str]:
        payload = {"error": self.message}
        if self.detail:
            payload["detail"] = self.detail
        return payload


class InputError(TriageError):
    exit_code = 2


class ConfigError(TriageError):
    exit_code = 2


class ValidationFailure(TriageError):
    exit_code = 1


class OutcomeRejected(TriageError):
    exit_code = 1


class RunInterrupted(TriageError):
    exit_code = 3


class StoreError(TriageError):
    exit_code = 4


class ModelError(TriageError):
    exit_code = 4


class ModelOutputError(ModelError):
    """Provider returned something that cannot be parsed into the final contract."""

    def __init__(self, message: str, *, detail: str | None = None, raw_text: str | None = None) -> None:
        super().__init__(message, detail=detail)
        self.raw_text = raw_text


class ToolFailure(Exception):
    """Raised by a tool implementation; converted to a ToolResult error."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable