from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from trace_tasks import __version__

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"


def _run_example(name: str, *args: object) -> dict[str, object]:
    env = dict(os.environ)
    source_root = str(REPO_ROOT / "src")
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{current_pythonpath}"
        if current_pythonpath
        else source_root
    )
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_ROOT / name), *(str(arg) for arg in args)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, dict)
    return parsed


def test_python_examples_run_end_to_end(tmp_path: Path) -> None:
    generated = _run_example(
        "generate_and_validate.py",
        "--output",
        tmp_path / "generated",
        "--count",
        1,
    )
    assert generated["instance_count"] == 1
    assert generated["total_errors"] == 0
    dataset_root = Path(str(generated["dataset_root"]))
    build_report = json.loads(
        (dataset_root / "build_report.json").read_text(encoding="utf-8")
    )
    assert build_report["code_provenance"] == {
        "code_hash": f"trace-tasks-{__version__}",
        "identity_input": True,
    }

    replayed = _run_example("replay_and_score.py", dataset_root)
    scores = replayed["scores"]
    assert isinstance(scores, dict)
    assert scores["answer_reward"] == 1.0

    output_path = tmp_path / "trace-rlvr.jsonl"
    exported = _run_example(
        "export_dataset.py",
        dataset_root,
        "--output",
        output_path,
    )
    assert exported["format"] == "jsonl"
    assert exported["prompt_variant"] == "answer_only"
    assert exported["row_count"] == 1
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 1
