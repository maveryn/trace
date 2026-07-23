#!/usr/bin/env python3
"""Export a canonical Trace dataset to RLVR-ready JSONL or Parquet."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from typing import Sequence

from trace_tasks import export_trace_dataset_to_rlvr


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/python-api-example/trace-rlvr.jsonl"),
    )
    parser.add_argument("--format", choices=("jsonl", "parquet"))
    parser.add_argument(
        "--prompt-variant",
        choices=("answer", "answer_only", "annotation", "answer_and_annotation"),
        default="answer",
    )
    args = parser.parse_args(argv)

    # Export progress is useful to humans, while stdout stays machine-readable.
    with redirect_stdout(sys.stderr):
        result = export_trace_dataset_to_rlvr(
            args.source,
            args.output,
            output_format=args.format,
            prompt_variant=args.prompt_variant,
            image_path_mode="relative",
        )
    print(
        json.dumps(
            {
                "format": result.output_format,
                "output_path": str(result.output_path),
                "prompt_variant": result.prompt_variant,
                "row_count": result.row_count,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
