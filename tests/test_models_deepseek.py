"""DeepSeek uses the real Anthropic SDK with an offline HTTP transport."""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
import httpx
import pytest
from pydantic import ValidationError

from agent_triage import cli
from agent_triage.config import Config, config_fingerprint
from agent_triage.demo import demo_event, write_demo_project
from agent_triage.doctor import run_doctor
from agent_triage.models import DeepSeekModel, FinalResponse, ModelError, RemainingBudget, ToolRequests, build_model
from agent_triage.tools import tool_specs
from test_models_anthropic import _analysis_text, _context, _message

SECRET = "deepseek-synthetic-key"


def _transport(monkeypatch, handler) -> list[httpx.Request]:
    seen: list[httpx.Request] = []
    constructor = anthropic.Anthropic

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    def client(**kwargs):
        return constructor(http_client=httpx.Client(transport=httpx.MockTransport(wrapped)), **kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", client)
    monkeypatch.setenv("DEEPSEEK_API_KEY", SECRET)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://wrong-provider.invalid")
    return seen


def test_deepseek_effective_config_and_endpoint_fingerprint(monkeypatch) -> None:
    config = Config(provider="deepseek")
    assert config.model == "deepseek-flash"
    assert config.credential_env == "DEEPSEEK_API_KEY"
    assert config.base_url == "https://api.deepseek.com/anthropic"
    other = Config(provider="deepseek", base_url="https://proxy.invalid/anthropic")
    assert config_fingerprint(config) != config_fingerprint(other)
    original = config_fingerprint(config)
    monkeypatch.setenv("DEEPSEEK_API_KEY", SECRET)
    assert config_fingerprint(config) == original
    assert SECRET not in config.model_dump_json()


@pytest.mark.parametrize(
    "endpoint",
    ["https://key@api.invalid", "https://api.invalid?api_key=secret", "https://api.invalid#secret", "file:///tmp/x"],
)
def test_endpoint_rejects_credentials_and_non_http_urls(endpoint: str) -> None:
    with pytest.raises(ValidationError):
        Config(provider="deepseek", base_url=endpoint)


def test_deepseek_tool_result_and_final_json_round_trip(monkeypatch) -> None:
    replies = [
        _message([{"type": "tool_use", "id": "toolu_ds", "name": "read_file", "input": {"path": "src/app.py"}}]),
        _message([{"type": "text", "text": _analysis_text()}], stop_reason="end_turn"),
    ]
    seen = _transport(monkeypatch, lambda request: httpx.Response(200, json=replies.pop(0)))
    config = Config(provider="deepseek")
    model = build_model(
        config.provider, model=config.model, credential_env=config.credential_env, base_url=config.base_url
    )
    assert isinstance(model, DeepSeekModel)
    assert model.provider == "deepseek"
    context = _context()
    response = model.next(context, tool_specs(), RemainingBudget(3, 4, 10.0))
    assert isinstance(response, ToolRequests)
    assert response.calls[0].call_id == "toolu_ds"
    assert response.calls[0].arguments == {"path": "src/app.py"}
    context.messages.extend(
        [
            {"role": "assistant", "content": response.assistant_content},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_ds", "content": "source evidence"}]},
        ]
    )
    final = model.next(context, tool_specs(), RemainingBudget(2, 3, 9.0))
    assert isinstance(final, FinalResponse)
    assert final.analysis.analysis_status.value == "partial"
    assert len(seen) == 2
    for request in seen:
        assert str(request.url) == "https://api.deepseek.com/anthropic/v1/messages"
        assert request.headers["x-api-key"] == SECRET
        payload = json.loads(request.content)
        assert payload["model"] == "deepseek-flash"
        assert payload["thinking"] == {"type": "disabled"}
        assert payload["tools"]
    second = json.loads(seen[1].content)
    assert second["messages"][-1]["content"][0]["tool_use_id"] == "toolu_ds"
    assert second["messages"][-1]["content"][0]["content"] == "source evidence"


def test_existing_anthropic_endpoint_environment_still_works(monkeypatch) -> None:
    reply = _message([{"type": "text", "text": _analysis_text()}], stop_reason="end_turn")
    seen = _transport(monkeypatch, lambda request: httpx.Response(200, json=reply))
    model = build_model("anthropic", model="synthetic-model", credential_env="DEEPSEEK_API_KEY")
    model.next(_context(), tool_specs(), RemainingBudget(2, 3, 10.0))
    assert str(seen[0].url) == "https://wrong-provider.invalid/v1/messages"
    assert "thinking" not in json.loads(seen[0].content)


@pytest.mark.parametrize("status", [401, 429, 500])
def test_deepseek_errors_are_one_counted_request(monkeypatch, status: int) -> None:
    seen = _transport(
        monkeypatch,
        lambda request: httpx.Response(status, json={"type": "error", "error": {"type": "api_error", "message": "failed"}}),
    )
    model = DeepSeekModel()
    with pytest.raises(ModelError, match="model request failed"):
        model.next(_context(), tool_specs(), RemainingBudget(2, 3, 10.0))
    assert len(seen) == 1


def test_deepseek_cli_uses_config_endpoint_and_records_actual_provider(tmp_path: Path, monkeypatch, capsys) -> None:
    reply = _message([{"type": "text", "text": _analysis_text()}], stop_reason="end_turn")
    seen = _transport(monkeypatch, lambda request: httpx.Response(200, json=reply))
    project = tmp_path / "project"
    write_demo_project(project)
    event = tmp_path / "event.json"
    event.write_text(demo_event().model_dump_json(), encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text('provider = "deepseek"\nbase_url = "https://proxy.invalid/anthropic"\n', encoding="utf-8")
    output = tmp_path / "runs"
    assert cli.main(
        ["analyze", "--event", str(event), "--project-root", str(project), "--config", str(config), "--output", str(output)]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    run = output / result["run_id"]
    assert str(seen[0].url) == "https://proxy.invalid/anthropic/v1/messages"
    metadata = json.loads((run / "run.json").read_text())
    assert metadata["provider"] == "deepseek"
    assert metadata["model"] == "deepseek-flash"
    recorded_config = (run / "config.json").read_text()
    assert SECRET not in recorded_config
    assert json.loads(recorded_config)["credential_env"] == "DEEPSEEK_API_KEY"
    assert (run / "request-views").is_dir()


def test_deepseek_missing_key_does_not_create_a_run(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    project = tmp_path / "project"
    write_demo_project(project)
    event = tmp_path / "event.json"
    event.write_text(demo_event().model_dump_json(), encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text('provider = "deepseek"\n', encoding="utf-8")
    output = tmp_path / "runs"
    assert cli.main(
        ["analyze", "--event", str(event), "--project-root", str(project), "--config", str(config), "--output", str(output)]
    ) == 2
    captured = capsys.readouterr()
    assert "DEEPSEEK_API_KEY" in captured.err
    assert not list(output.glob("run-*"))


def test_doctor_recognizes_deepseek_credentials(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('provider = "deepseek"\n', encoding="utf-8")
    report = run_doctor(
        profile="connected", output=tmp_path / "doctor", config_path=config, env={"DEEPSEEK_API_KEY": SECRET}
    )
    checks = {check.check_id: check for check in report.checks}
    assert checks["provider_credential"].status == "pass"
    assert checks["model_name"].status == "pass"
    assert checks["mcp_config"].status == "blocked"
    assert SECRET not in report.model_dump_json()
    assert report.network == report.model_call == "not_run"
