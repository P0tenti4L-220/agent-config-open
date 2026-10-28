"""Service layer shared by the CLI, demo and tests."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from .config import Config, config_fingerprint
from .context import ContextBuilder
from .contracts import Event
from .errors import InputError, ValidationFailure
from .ingest import event_to_jsonable, fingerprint
from .mcp_client import GitHubMCPBackend
from .models import ReplayScenario, build_model
from .runner import RunResult, Runner
from .store import Checkpoint, EvidenceStore, LessonStore, RegistryStore, RunStore
from .tools import GithubBackend, Workspace


def build_versions(config: Config, event: Event) -> dict[str, str]:
    from . import __version__

    from .context import PROMPT_HASH
    from .playbooks import catalog_manifest
    from .policy import (
        PLAYBOOK_CATALOG_VERSION,
        POLICY_HASH,
        POLICY_VERSION,
        PROMPT_VERSION,
        TOOL_SCHEMA_VERSION,
    )

    return {
        "package": __version__,
        "context": "v1",
        "rules": event.rule_version,
        "provider": config.provider,
        "model": config.model,
        "policy_version": POLICY_VERSION,
        "policy_hash": POLICY_HASH,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": PROMPT_HASH,
        "playbook_catalog_version": PLAYBOOK_CATALOG_VERSION,
        **catalog_manifest(),
        "tool_schema_version": TOOL_SCHEMA_VERSION,
    }


def run_new_analysis(
    *,
    event: Event,
    project_root: Path,
    output_root: Path,
    config: Config,
    replay_scenario: ReplayScenario | None,
    clock: Callable[[], float] = time.monotonic,
    interrupt_after_rounds: int | None = None,
) -> RunResult:
    fingerprint_info = fingerprint(event)
    # Validate the workspace and provider before creating a run directory, so a
    # bad config or credential does not leave an empty registered run behind.
    workspace = Workspace(project_root)
    model = build_model(
        config.provider,
        model=config.model,
        credential_env=config.credential_env,
        base_url=config.base_url,
        replay_scenario=replay_scenario,
        max_output_tokens=config.context.max_output_tokens,
    )
    store = RunStore.create(
        output_root,
        event_payload=event_to_jsonable(event),
        event_id=event.event_id,
        project_id=event.project_id,
        project_root=project_root,
        provider=config.provider,
        model=config.model,
        config_fingerprint=config_fingerprint(config),
    )
    store.write_json("config.json", config.model_dump(mode="json"))
    if replay_scenario is not None:
        store.write_json("replay.json", replay_scenario.model_dump(mode="json"))
    registry = RegistryStore(output_root)
    occurrence, duplicate, _ = registry.register(
        event_id=event.event_id,
        project_id=event.project_id,
        fingerprint=fingerprint_info["value"],
        fingerprint_algorithm=fingerprint_info["algorithm"],
        symptom=event.symptom,
        run_id=store.run_id,
        provider=config.provider,
    )
    evidence = EvidenceStore(store)
    lessons = LessonStore(output_root)
    backend = _start_backend(config)
    try:
        builder = ContextBuilder(config.context, versions=build_versions(config, event))
        runner = Runner(
            store=store,
            event=event,
            config=config,
            model=model,
            context_builder=builder,
            evidence_store=evidence,
            lesson_store=lessons,
            workspace=workspace,
            github=backend,
            clock=clock,
            interrupt_after_rounds=interrupt_after_rounds,
            duplicate_event=duplicate,
            occurrence=occurrence,
        )
        result = runner.run()
        result.duplicate_event = duplicate
        result.occurrence = occurrence
        return result
    finally:
        if backend is not None:
            backend.close()


def resume_analysis(
    *,
    store: RunStore,
    config: Config,
    replay_scenario: ReplayScenario | None,
    accept_changed_sources: bool = False,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult:
    checkpoint = store.load_checkpoint()
    if checkpoint is None:
        raise InputError("run has no checkpoint; resume requires an interrupted run")
    if config_fingerprint(config) != store.meta.config_fingerprint:
        raise ValidationFailure(
            "configuration differs from the configuration recorded for this run"
        )
    event = Event.model_validate(store.read_json("event.json"))
    project_root = store.project_root()
    workspace = Workspace(project_root)
    evidence = EvidenceStore(store)
    lessons = LessonStore(store.output_root)
    backend = _start_backend(config)
    try:
        model = build_model(
            config.provider,
            model=config.model,
            credential_env=config.credential_env,
            base_url=config.base_url,
            replay_scenario=replay_scenario,
            replay_position=checkpoint.replay_position,
            max_output_tokens=config.context.max_output_tokens,
        )
        builder = ContextBuilder(config.context, versions=build_versions(config, event))
        runner = Runner(
            store=store,
            event=event,
            config=config,
            model=model,
            context_builder=builder,
            evidence_store=evidence,
            lesson_store=lessons,
            workspace=workspace,
            github=backend,
            checkpoint=checkpoint,
            clock=clock,
            accept_changed_sources=accept_changed_sources,
        )
        return runner.run()
    finally:
        if backend is not None:
            backend.close()


def _start_backend(config: Config) -> GithubBackend | None:
    if not config.mcp.enabled:
        return None
    backend = GitHubMCPBackend(config.mcp)
    backend.start()
    return backend
