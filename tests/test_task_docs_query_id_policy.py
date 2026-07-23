"""Regression tests for public query-id documentation policy."""

from __future__ import annotations

from pathlib import Path
import re


_QUERY_ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|")
_SUPPORTED_PATTERNS = (
    re.compile(r"Supported `query_id` values:\s*((?:`[^`]+`(?:,\s*)?)+)"),
    re.compile(r"Supported sampled `query_id`:\s*((?:`[^`]+`(?:,\s*)?)+)"),
)


def test_single_query_task_docs_use_single_sentinel() -> None:
    """Task docs with one public query must not use objective names as query ids."""

    offenders: list[str] = []
    for path in sorted(Path("docs/tasks").glob("*/*/task_*.md")):
        text = path.read_text(encoding="utf-8")
        rows = [
            match.group(1)
            for line in text.splitlines()
            if (match := _QUERY_ROW_RE.match(line))
        ]
        if len(set(rows)) == 1 and rows and rows[0] != "single":
            offenders.append(f"{path}: query-details row uses `{rows[0]}`")

        query_id_lines: list[str] = []
        for line in text.splitlines():
            if "Query id:" in line:
                query_id_lines.extend(re.findall(r"`([^`]+)`", line))
            for pattern in _SUPPORTED_PATTERNS:
                match = pattern.search(line)
                if match is None:
                    continue
                values = re.findall(r"`([^`]+)`", match.group(1))
                if len(values) == 1 and values[0] != "single":
                    offenders.append(f"{path}: supported query line uses `{values[0]}`")

        if len(set(query_id_lines)) == 1 and query_id_lines[0] != "single":
            offenders.append(f"{path}: query-id line uses `{query_id_lines[0]}`")

    assert offenders == []
