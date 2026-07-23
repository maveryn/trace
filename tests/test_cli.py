from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import trace_tasks.cli as cli
from trace_tasks.cli import export_main, generate_main, list_main


def test_list_json(capsys) -> None:
    assert list_main(["--domain", "icons", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 50
    assert {row["domain"] for row in payload["tasks"]} == {"icons"}


def test_generate_dry_run(capsys, tmp_path) -> None:
    assert (
        generate_main(
            [
                "--task",
                "task_geometry__graph_paper__polygon_area_value",
                "--seed",
                "42",
                "--output",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["task_count"] == 1
    assert payload["instance_count"] == 1
    assert payload["sampling_seed"] == 42


def test_export_keeps_stdout_machine_readable(capsys, monkeypatch, tmp_path) -> None:
    output_path = tmp_path / "out.jsonl"

    def fake_export(*args, **kwargs):
        print("progress message")
        return SimpleNamespace(
            output_path=output_path,
            row_count=1,
            output_format="jsonl",
            prompt_variant="active",
            image_path_mode="relative",
        )

    monkeypatch.setattr(cli, "export_trace_dataset_to_rlvr", fake_export)
    assert export_main([str(Path("dataset")), "--output", str(output_path)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["row_count"] == 1
    assert "progress message" in captured.err
