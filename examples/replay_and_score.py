#!/usr/bin/env python3
"""Replay stored verifier inputs and score a model response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from trace_tasks import score_trace_response


def _read_record(source: Path, row_index: int) -> dict[str, Any]:
    records_path = source / "train_instances.jsonl" if source.is_dir() else source
    if records_path.name != "train_instances.jsonl":
        raise ValueError("source must be a dataset root or train_instances.jsonl")
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if row_index < 0 or row_index >= len(records):
        raise IndexError(
            f"row index {row_index} is outside a {len(records)}-row dataset"
        )
    record = records[row_index]
    if not isinstance(record, dict):
        raise TypeError("training rows must be JSON objects")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument(
        "--response",
        help="Model response; defaults to a ground-truth answer scorer smoke test",
    )
    args = parser.parse_args(argv)

    try:
        record = _read_record(args.source, args.row)
    except (OSError, ValueError, IndexError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    response = args.response
    if response is None:
        response = json.dumps(
            {"answer": record["answer_gt"]["value"]},
            ensure_ascii=False,
        )
    scores = score_trace_response(
        response=response,
        answer_gt=record["answer_gt"],
        annotation_gt=record["annotation_gt"],
        reward_contract=record["reward_contract"],
        answer_weight=1.0,
        annotation_weight=0.0,
        trace_reward_mode="answer",
    )
    print(
        json.dumps(
            {
                "instance_id": record["instance_id"],
                "response": response,
                "scores": scores,
                "task": record["task"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
