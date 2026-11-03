"""Provider-boundary contract tests with an injected HTTP transport.

These tests never open a socket.  They exercise the real Anthropic SDK request
serialization and the local parsing/error mapping, including retry and timeout
behavior, without a live credential.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from anthropic import Anthropic

from agent_triage.contracts import CountMode
from agent_triage.errors import ModelError, ModelOutputError
from agent_triage.models import AnthropicModel, FinalResponse, ModelContext, RemainingBudget, ToolRequests
from agent_triage.tools import tool_specs


def _context() -> ModelContext:
    return ModelContext(
        system="You are a test harness.",
        messages=[{"role": "user", "content": [{"type": "text", "text": "read the failure"}]}],
        packet=None,
        input_tokens=8,
        count_mode=CountMode.ESTIMATED,
        is_final_turn=False,
    )


def _model(handler) -> tuple[AnthropicModel, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    http_client = httpx.Client(transport=httpx.MockTransport(wrapped))
    client = Anthropic(api_key="test-key", base_url="http://test.invalid", http_client=http_client, max_retries=0)
    return AnthropicModel(model="synthetic-model", credential_env="UNUSED", client=client), seen


def _message(content: list[dict[str, Any]], *, stop_reason: str = "tool_use") -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "synthetic-model",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 7},
    }


def _analysis_text() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "summary": "synthetic summary",
            "attribution": "unknown",
            "facts": [],
            "hypotheses": [],
            "missing_evidence": ["need a real input"],
            "suggested_checks": [],
            "duplicate_candidates": [],
            "analysis_status": "partial",
            "stop_reason": "insufficient_evidence",
        }
    )


@pytest.mark.parametrize("tool_name, arguments", [
    ("read_file", {"path": "src/app.py"}),
    ("read_evidence", {"evidence_id": "ev-0123456789ab", "cursor": "bound-cursor", "max_bytes": 1024}),
])
def test_tool_use_round_trip_and_request_shape(tool_name: str, arguments: dict) -> None:
    body = _message(
        [
            {"type": "text", "text": "I will read the file."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": tool_name,
                "input": arguments,
            },
        ]
    )
    model, seen = _model(lambda request: httpx.Response(200, json=body))
    response = model.next(_context(), tool_specs(), RemainingBudget(2, 3, 10.0))
    assert isinstance(response, ToolRequests)
    assert [call.name for call in response.calls] == [tool_name]
    assert response.calls[0].call_id == "toolu_1"
    assert response.calls[0].arguments == arguments
    assert response.assistant_content[0]["type"] == "text"
    assert response.usage == {"input_tokens": 12, "output_tokens": 7}
    request = seen[0]
    payload = json.loads(request.content)
    assert payload["model"] == "synthetic-model"
    assert payload["system"] == "You are a test harness."
    assert payload["temperature"] == 0
    assert payload["messages"][0]["role"] == "user"
    assert {tool["name"] for tool in payload["tools"]} == {spec.name for spec in tool_specs()}


def test_text_only_final_response_is_parsed() -> None:
    body = _message([{"type": "text", "text": _analysis_text()}], stop_reason="end_turn")
    model, _ = _model(lambda request: httpx.Response(200, json=body))
    response = model.next(_context(), tool_specs(), RemainingBudget(2, 3, 10.0))
    assert isinstance(response, FinalResponse)
    assert response.analysis.analysis_status.value == "partial"
    assert response.analysis.missing_evidence


def test_invalid_final_json_raises_output_error() -> None:
    body = _message([{"type": "text", "text": "not json at all"}], stop_reason="end_turn")
    model, _ = _model(lambda request: httpx.Response(200, json=body))
    with pytest.raises(ModelOutputError):
        model.next(_context(), tool_specs(), RemainingBudget(2, 3, 10.0))


def test_truncated_final_json_raises_output_error() -> None:
    body = _message(
        [{"type": "text", "text": '{"schema_version": 1, "summary": "cut'}],
        stop_reason="max_tokens",
    )
    model, _ = _model(lambda request: httpx.Response(200, json=body))
    with pytest.raises(ModelOutputError):
        model.next(_context(), tool_specs(), RemainingBudget(2, 3, 10.0))


def test_rate_limit_is_one_attempt_and_typed_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
            headers={"retry-after": "1"},
        )

    model, seen = _model(handler)
    with pytest.raises(ModelError, match="model request failed"):
        model.next(_context(), tool_specs(), RemainingBudget(2, 3, 10.0))
    assert len(seen) == 1


def test_provider_timeout_is_typed_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    model, seen = _model(handler)
    with pytest.raises(ModelError):
        model.next(_context(), tool_specs(), RemainingBudget(2, 3, 10.0))
    assert len(seen) == 1


def test_server_error_is_typed_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "boom"}})

    model, seen = _model(handler)
    with pytest.raises(ModelError):
        model.next(_context(), tool_specs(), RemainingBudget(2, 3, 10.0))
    assert len(seen) == 1
