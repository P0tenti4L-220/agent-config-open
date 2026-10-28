"""Replay and live providers using the Anthropic messages protocol."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field, ValidationError, model_validator

from .contracts import Analysis, CountMode, StrictModel, ToolSpec
from .errors import ConfigError, ModelError, ModelOutputError


@dataclass
class ModelContext:
    """Provider-independent input for one model decision."""

    system: str
    messages: list[dict[str, Any]]
    packet: Any
    input_tokens: int
    count_mode: CountMode
    is_final_turn: bool
    displayed: list[Any] = field(default_factory=list)
    omissions: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class RemainingBudget:
    model_requests: int
    tool_attempts: int
    seconds: float


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolRequests:
    calls: list[ToolCall]
    assistant_content: list[dict[str, Any]] = field(default_factory=list)
    text: str | None = None
    usage: dict[str, int] | None = None


@dataclass
class FinalResponse:
    analysis: Analysis
    text: str | None = None
    usage: dict[str, int] | None = None


class ModelClient(Protocol):
    def next(
        self,
        context: ModelContext,
        tool_specs: list[ToolSpec],
        remaining_budget: RemainingBudget,
    ) -> ToolRequests | FinalResponse: ...


# ---------------------------------------------------------------------------
# Replay provider
# ---------------------------------------------------------------------------


class ReplayToolCall(StrictModel):
    tool: str
    arguments: dict[str, Any]
    call_id: str | None = None


class ReplayStep(StrictModel):
    type: str
    calls: list[ReplayToolCall] | None = None
    analysis: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _shape(self) -> "ReplayStep":
        if self.type == "tools":
            if not self.calls:
                raise ValueError("tools step requires at least one call")
        elif self.type == "final":
            if self.analysis is None:
                raise ValueError("final step requires an analysis object")
        else:
            raise ValueError("replay step type must be 'tools' or 'final'")
        return self


class ReplayScenario(StrictModel):
    schema_version: int = Field(default=1, strict=True)
    name: str = "replay"
    steps: list[ReplayStep] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def _version(self) -> "ReplayScenario":
        if self.schema_version != 1:
            raise ValueError("unsupported replay schema_version")
        return self


def load_replay(path: Path) -> ReplayScenario:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read replay script: {path.name}", detail=str(exc)) from exc
    try:
        return ReplayScenario.model_validate_json(raw)
    except ValidationError as exc:
        raise ConfigError("replay script does not match the expected shape", detail=str(exc)) from exc


class ReplayModel:
    """Deterministic provider used by demo, tests and eval.

    Only model responses are replayed.  Tool execution, argument validation,
    context construction and output validation still use production code.
    """

    provider = "replay"

    def __init__(self, scenario: ReplayScenario, *, position: int = 0) -> None:
        self.scenario = scenario
        self.position = position
        self.name = scenario.name

    def next(
        self,
        context: ModelContext,
        tool_specs: list[ToolSpec],
        remaining_budget: RemainingBudget,
    ) -> ToolRequests | FinalResponse:
        if self.position >= len(self.scenario.steps):
            raise ModelOutputError(
                "replay script exhausted before a final analysis",
                detail=f"position={self.position}, steps={len(self.scenario.steps)}",
            )
        step = self.scenario.steps[self.position]
        self.position += 1
        if step.type == "tools":
            calls: list[ToolCall] = []
            for index, call in enumerate(step.calls or []):
                call_id = call.call_id or f"call-{self.position}-{index}"
                calls.append(ToolCall(call_id=call_id, name=call.tool, arguments=call.arguments))
            content = [
                {
                    "type": "tool_use",
                    "id": call.call_id,
                    "name": call.name,
                    "input": call.arguments,
                }
                for call in calls
            ]
            return ToolRequests(calls=calls, assistant_content=content)
        raw = json.loads(json.dumps(step.analysis))
        resolved = resolve_evidence_aliases(raw, context.packet.evidence_index)
        try:
            analysis = Analysis.model_validate(resolved)
            from .review import sanitize_model_review

            analysis, _ = sanitize_model_review(analysis)
        except ValidationError as exc:
            details = []
            for error in exc.errors()[:10]:
                location = ".".join(str(item) for item in error["loc"]) or "<root>"
                details.append(f"{location}: {error['msg']}")
            raise ModelOutputError(
                "replay final analysis does not match the contract",
                detail="; ".join(details),
                raw_text=json.dumps(step.analysis, ensure_ascii=False),
            ) from exc
        return FinalResponse(analysis=analysis)

    def dump_scenario(self) -> dict[str, Any]:
        return self.scenario.model_dump(mode="json")


def resolve_evidence_aliases(value: Any, evidence_index: list[Any]) -> Any:
    """Resolve replay-only aliases such as ``@path:src/app.py``.

    Aliases are resolved against evidence read during this run.  If an alias
    cannot be resolved it is deliberately left untouched so contract
    validation fails instead of silently inventing an evidence id.
    """
    if isinstance(value, str):
        if value.startswith("@path:"):
            wanted = value[len("@path:") :]
            matches = [item for item in evidence_index if item.path == wanted and not item.stale]
            return matches[-1].evidence_id if matches else value
        if value.startswith("@kind:"):
            wanted = value[len("@kind:") :]
            matches = [item for item in evidence_index if item.kind.value == wanted and not item.stale]
            return matches[-1].evidence_id if matches else value
        if value.startswith("@url:"):
            wanted = value[len("@url:") :]
            matches = [item for item in evidence_index if item.url == wanted and not item.stale]
            return matches[-1].evidence_id if matches else value
        return value
    if isinstance(value, list):
        return [resolve_evidence_aliases(item, evidence_index) for item in value]
    if isinstance(value, dict):
        return {key: resolve_evidence_aliases(item, evidence_index) for key, item in value.items()}
    return value


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------


class AnthropicModel:
    """Native Anthropic tool-use provider.

    The SDK retry loop is disabled; retries are owned by the runner so every
    attempt is counted. Offline tests exercise the SDK through MockTransport.
    """

    provider = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        credential_env: str,
        base_url: str | None = None,
        max_output_tokens: int = 3000,
        disable_thinking: bool = False,
        client: Any | None = None,
    ) -> None:
        key = os.environ.get(credential_env)
        if client is None and not key:
            raise ConfigError(
                f"missing model credential in environment variable {credential_env}; "
                "live analysis remains not_run"
            )
        self.model = model
        self.disable_thinking = disable_thinking
        self.max_output_tokens = max(256, min(int(max_output_tokens), 64_000))
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise ConfigError("anthropic SDK is not installed") from exc
            client_options: dict[str, Any] = {}
            if base_url is not None:
                client_options["base_url"] = base_url
            client = Anthropic(api_key=key, max_retries=0, timeout=45.0, **client_options)
        self.client = client
        self.name = model

    def next(
        self,
        context: ModelContext,
        tool_specs: list[ToolSpec],
        remaining_budget: RemainingBudget,
    ) -> ToolRequests | FinalResponse:
        messages = context.messages
        if not messages:
            messages = [{"role": "user", "content": [{"type": "text", "text": context.system}]}]
        tools = [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
            }
            for spec in tool_specs
        ]
        options: dict[str, Any] = {}
        if self.disable_thinking:
            options["thinking"] = {"type": "disabled"}
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                temperature=0,
                system=context.system,
                messages=messages,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
                timeout=remaining_budget.seconds,
                **options,
            )
        except Exception as exc:  # noqa: BLE001 - converted into a typed model error
            raise ModelError(
                "model request failed",
                detail=f"{type(exc).__name__}: {str(exc)[:300]}",
            ) from exc
        usage = {
            "input_tokens": int(response.usage.input_tokens),
            "output_tokens": int(response.usage.output_tokens),
        }
        blocks: list[dict[str, Any]] = []
        tool_calls: list[ToolCall] = []
        texts: list[str] = []
        for block in response.content:
            payload = block.model_dump(mode="json")
            blocks.append(payload)
            if payload.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        call_id=str(payload.get("id")),
                        name=str(payload.get("name")),
                        arguments=payload.get("input") if isinstance(payload.get("input"), dict) else {},
                    )
                )
            elif payload.get("type") == "text":
                texts.append(str(payload.get("text") or ""))
        text = "\n".join(texts)
        if tool_calls:
            return ToolRequests(calls=tool_calls, assistant_content=blocks, text=text, usage=usage)
        analysis = parse_analysis_text(text)
        return FinalResponse(analysis=analysis, text=text, usage=usage)


class DeepSeekModel(AnthropicModel):
    """DeepSeek Flash via its official Anthropic-compatible endpoint.

    Start with non-thinking mode so the existing bounded tool loop keeps its
    output budget for tool calls and the final structured analysis.
    """

    provider = "deepseek"

    def __init__(
        self,
        *,
        model: str = "deepseek-flash",
        credential_env: str = "DEEPSEEK_API_KEY",
        base_url: str = "https://api.deepseek.com/anthropic",
        max_output_tokens: int = 3000,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            model=model,
            credential_env=credential_env,
            base_url=base_url,
            max_output_tokens=max_output_tokens,
            disable_thinking=True,
            client=client,
        )


def parse_analysis_text(text: str) -> Analysis:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ModelOutputError("model final response contains no JSON object", raw_text=text[:20_000])
    payload = cleaned[start : end + 1]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ModelOutputError(
            "model final response is not valid JSON", detail=str(exc), raw_text=text[:20_000]
        ) from exc
    try:
        analysis = Analysis.model_validate(data)
        from .review import sanitize_model_review

        return sanitize_model_review(analysis)[0]
    except ValidationError as exc:
        details = []
        for error in exc.errors()[:10]:
            location = ".".join(str(item) for item in error["loc"]) or "<root>"
            details.append(f"{location}: {error['msg']}")
        raise ModelOutputError(
            "model final response does not match the analysis contract",
            detail="; ".join(details),
            raw_text=text[:20_000],
        ) from exc


def build_model(
    provider: str,
    *,
    model: str,
    credential_env: str,
    base_url: str | None = None,
    replay_scenario: ReplayScenario | None = None,
    replay_position: int = 0,
    max_output_tokens: int = 3000,
) -> ModelClient:
    if provider == "replay":
        if replay_scenario is None:
            raise ConfigError("replay provider requires a replay script")
        return ReplayModel(replay_scenario, position=replay_position)
    if provider == "anthropic":
        return AnthropicModel(
            model=model,
            credential_env=credential_env,
            base_url=base_url,
            max_output_tokens=max_output_tokens,
        )
    if provider == "deepseek":
        return DeepSeekModel(
            model=model,
            credential_env=credential_env,
            base_url=base_url or "https://api.deepseek.com/anthropic",
            max_output_tokens=max_output_tokens,
        )
    raise ConfigError(f"unsupported provider: {provider}")
