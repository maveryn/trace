"""Cross-interpreter determinism regressions for canonical review capture."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = """
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "src"))

from trace_tasks.review.recipe import _default_generator, prepare_output

task_id = sys.argv[1]
output = _default_generator(task_id, int(sys.argv[2]), json.loads(sys.argv[3]), 100)
print(json.dumps(prepare_output(output, task_id=task_id).hashes.to_dict(), sort_keys=True))
"""


def _hashes_for_seed(
    *, task_id: str, instance_seed: int, params: dict[str, object], hash_seed: int
) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(hash_seed)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            PROBE,
            task_id,
            str(instance_seed),
            json.dumps(params, sort_keys=True),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(json.loads(completed.stdout))


@pytest.mark.parametrize(
    ("task_id", "instance_seed", "params"),
    (
        (
            "task_charts__multiseries__category_total_extremum_label",
            4236176091273391,
            {"_sample_cursor": 0, "query_id": "largest_category_total_label"},
        ),
        (
            "task_graph__graph_options__contained_subgraph_label",
            7873567924254169,
            {"_sample_cursor": 0, "query_id": "single"},
        ),
        (
            "task_pages__calendar_event_grid__weekday_event_count",
            2489519197383255,
            {"_sample_cursor": 2, "query_id": "single"},
        ),
    ),
)
def test_review_capture_is_stable_across_python_hash_seeds(
    task_id: str,
    instance_seed: int,
    params: dict[str, object],
) -> None:
    assert _hashes_for_seed(
        task_id=task_id,
        instance_seed=instance_seed,
        params=params,
        hash_seed=3,
    ) == _hashes_for_seed(
        task_id=task_id,
        instance_seed=instance_seed,
        params=params,
        hash_seed=9,
    )
