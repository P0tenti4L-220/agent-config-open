from __future__ import annotations

from pathlib import Path

from agent_triage.ingest import event_from_symptom_file


def _event(tmp_path: Path, text: str):
    project = tmp_path / "project"
    (project / "logs").mkdir(parents=True)
    log = project / "logs" / "failure.log"
    log.write_text(text, encoding="utf-8")
    return event_from_symptom_file(
        log,
        project,
        goal="定位错误并整理验证材料",
        project_id="sample-project",
        rule_version="v0.1.0",
    )


def test_symptom_prefers_last_traceback_exception(tmp_path: Path) -> None:
    event = _event(
        tmp_path,
        "\n".join(
            [
                "2026-10-24T20:00:00+08:00 INFO loading check records from data/checks.json",
                "2026-10-24T20:00:00+08:00 ERROR failed to read check records",
                "Traceback (most recent call last):",
                '  File "src/check_reader.py", line 7, in load_checks',
                '    return payload["items"]',
                "KeyError: 'items'",
                "",
            ]
        ),
    )
    assert event.symptom == "KeyError: 'items'"
    assert "first line" not in (event.note or "")


def test_symptom_uses_last_exception_when_tracebacks_are_chained(tmp_path: Path) -> None:
    event = _event(
        tmp_path,
        "\n".join(
            [
                "Traceback (most recent call last):",
                '  File "src/reader.py", line 3, in read',
                "ValueError: inner failure",
                "",
                "During handling of the above exception, another exception occurred:",
                "",
                "Traceback (most recent call last):",
                '  File "src/reader.py", line 9, in read',
                "KeyError: 'items'",
            ]
        ),
    )
    assert event.symptom == "KeyError: 'items'"


def test_symptom_falls_back_to_error_line(tmp_path: Path) -> None:
    event = _event(
        tmp_path,
        "\n".join(
            [
                "2026-10-24T20:00:00+08:00 INFO loading check records",
                "2026-10-24T20:00:01+08:00 ERROR failed to read check records",
                "2026-10-24T20:00:02+08:00 INFO retrying",
            ]
        ),
    )
    assert event.symptom == "2026-10-24T20:00:01+08:00 ERROR failed to read check records"


def test_symptom_falls_back_to_first_non_empty_line(tmp_path: Path) -> None:
    event = _event(tmp_path, "\n\n  first line is the only clue\nsecond line\n")
    assert event.symptom == "first line is the only clue"
