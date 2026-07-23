"""Shared test utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, List, Mapping


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read one JSONL file into a list of dictionaries."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_prompt_json_example(prompt: str) -> Dict[str, Any]:
    """Extract the trailing example JSON payload from one rendered prompt."""

    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def assert_counter_support_within(
    counter: Mapping[Hashable, int],
    expected_keys: Iterable[Hashable],
    *,
    expected_per_key: int,
    tolerance: int,
) -> None:
    """Assert deterministic balanced sampling covers support within a bounded window."""

    expected = set(expected_keys)
    actual = {key for key, value in counter.items() if int(value) > 0}
    assert actual == expected
    low = max(0, int(expected_per_key) - int(tolerance))
    high = int(expected_per_key) + int(tolerance)
    for key in sorted(expected, key=str):
        value = int(counter.get(key, 0))
        assert low <= value <= high, f"{key!r}: expected {low}..{high}, got {value}"
