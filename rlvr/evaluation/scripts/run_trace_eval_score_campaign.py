#!/usr/bin/env python3
"""Score saved TRACE evaluation responses with the pinned benchmark routes."""

from __future__ import annotations

import sys

from run_trace_eval_official_score_campaign import main as _run_score_campaign


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--suite" in arguments:
        raise SystemExit("trace_eval_v1 is fixed; do not pass --suite")
    _run_score_campaign(["--suite", "trace_eval_v1", *arguments])


if __name__ == "__main__":
    main()
