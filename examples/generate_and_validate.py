#!/usr/bin/env python3
"""Generate one small canonical Trace dataset and inspect its validation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from trace_tasks import BuildConfig, BuildTaskConfig, __version__, build_dataset

DEFAULT_TASK_ID = "task_geometry__graph_paper__polygon_area_value"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/python-api-example")
    )
    parser.add_argument("--task", default=DEFAULT_TASK_ID)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--code-hash",
        default=f"trace-tasks-{__version__}",
        help="Source revision or package provenance recorded in instance identity",
    )
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be at least 1")

    config = BuildConfig(
        output_root=str(args.output),
        dataset_name="python-api-example",
        instance_version="v0",
        image_format="png",
        tasks=[BuildTaskConfig(task_id=args.task, count=args.count)],
        max_attempts_per_instance=100,
        sampling_seed=args.seed,
        workers=1,
    )
    dataset_root = build_dataset(config, code_hash=args.code_hash)
    validation_report = json.loads(
        (dataset_root / "validation_report.json").read_text(encoding="utf-8")
    )
    result = {
        "dataset_root": str(dataset_root),
        "instance_count": args.count,
        "task": args.task,
        "total_errors": int(validation_report["total_errors"]),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["total_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
