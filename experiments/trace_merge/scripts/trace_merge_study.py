#!/usr/bin/env python3
"""Entry point: python experiments/trace_merge/scripts/trace_merge_study.py <command> --config ... (see trace_merge.cli)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_merge.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
