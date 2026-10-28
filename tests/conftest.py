from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import (  # noqa: E402
    base_analysis,
    final_step,
    make_event,
    make_scenario,
    run_replay,
    tool_step,
    write_sample_project,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="opt in to tests marked live; credentials alone are not enough",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-live"):
        return
    skip = pytest.mark.skip(reason="live tests require the explicit --run-live opt-in")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    return write_sample_project(tmp_path / "project")


@pytest.fixture
def event():
    return make_event()


@pytest.fixture
def replay_run(tmp_path: Path):
    return run_replay(tmp_path)


__all__ = [
    "base_analysis",
    "final_step",
    "make_event",
    "make_scenario",
    "run_replay",
    "tool_step",
    "write_sample_project",
]