"""Shared helpers for deterministic JSON prompt examples."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def _is_point_coordinate(value: Any) -> bool:
    """Return true when value is one numeric 2D point."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    return all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)


def _example_answer_value(answer_type: str) -> Any:
    """Return one canonical answer example by answer type."""
    answer_kind = str(answer_type)
    if answer_kind == "integer":
        return 8
    if answer_kind == "number":
        return 42.3
    if answer_kind == "string":
        return "Ava"
    if answer_kind == "pi_expression":
        return "12π"
    if answer_kind == "option_letter":
        return "B"
    return "value"


def _canonical_point_examples(count: int) -> List[List[int]] | None:
    """Return one deterministic, non-degenerate point layout when available."""
    layouts: Dict[int, List[List[int]]] = {
        1: [[0, 0]],
        2: [[0, 0], [4, 0]],
        3: [[0, 0], [4, 0], [0, 3]],
        4: [[0, 0], [4, 0], [4, 2], [0, 2]],
        5: [[0, 0], [4, 0], [5, 2], [2, 4], [-1, 2]],
        6: [[0, 0], [3, 0], [6, 0], [0, 3], [3, 3], [6, 3]],
        7: [[0, 0], [3, 0], [6, 0], [0, 3], [3, 3], [6, 3], [0, 6]],
        8: [[0, 0], [3, 0], [6, 0], [0, 3], [3, 3], [6, 3], [0, 6], [3, 6]],
    }
    return layouts.get(int(count))


def _mapping_of_points(value: Any) -> Mapping[str, Any] | None:
    """Return mapping when all values are points."""
    if not isinstance(value, Mapping) or not value:
        return None
    if all(_is_point_coordinate(item) for item in value.values()):
        return value
    return None


def _sequence_of_points(value: Any) -> Sequence[Any] | None:
    """Return sequence when all items are points."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return None
    if all(_is_point_coordinate(item) for item in value):
        return value
    return None


def _sequence_of_strings(value: Any) -> Sequence[str] | None:
    """Return sequence when all items are plain strings."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return None
    if all(isinstance(item, str) for item in value):
        return [str(item) for item in value]
    return None


def _sequence_of_assignment_strings(value: Any) -> Sequence[str] | None:
    """Return sequence when all items are simple assignment-like strings."""
    string_seq = _sequence_of_strings(value)
    if string_seq is None:
        return None
    if all("=" in item and item.split("=", 1)[0].strip() for item in string_seq):
        return list(string_seq)
    return None


def _example_like(value: Any, *, index: int) -> Any:
    """Build one lightweight placeholder that preserves JSON shape."""
    point_map = _mapping_of_points(value)
    if point_map is not None:
        layout = _canonical_point_examples(len(point_map))
        if layout is not None:
            out: Dict[str, Any] = {}
            for offset, key in enumerate(point_map.keys()):
                out[str(key)] = list(layout[offset])
            return out
    point_seq = _sequence_of_points(value)
    if point_seq is not None:
        layout = _canonical_point_examples(len(point_seq))
        if layout is not None:
            return [list(point) for point in layout]
    string_seq = _sequence_of_strings(value)
    if string_seq is not None:
        assignment_seq = _sequence_of_assignment_strings(value)
        if assignment_seq is not None:
            out: List[str] = []
            for offset, item in enumerate(assignment_seq):
                key = str(item).split("=", 1)[0].strip()
                out.append(f"{key}={int(index + offset + 2)}")
            return out
        alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        count = min(len(alphabet), len(string_seq))
        start = 1 if count > 1 else 0
        return [str(alphabet[(start + index) % len(alphabet)]) for index in range(count)]
    if _is_point_coordinate(value):
        return [int(120 + 30 * index), int(140 + 30 * index)]
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(index + 2)
    if isinstance(value, float):
        rounded = round(float(value))
        if abs(float(value) - float(rounded)) <= 1e-9:
            return int(index + 2)
        return float(index + 2.5)
    if isinstance(value, str):
        text = str(value).strip()
        if text.endswith("π"):
            return f"{int(index + 2)}π"
        return str(index + 2)
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for offset, key in enumerate(value.keys()):
            out[str(key)] = _example_like(value[key], index=int(index + offset))
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        out_list: List[Any] = []
        for offset, item in enumerate(value):
            out_list.append(_example_like(item, index=int(index + offset)))
        return out_list
    return str(index + 2)


def build_prompt_json_examples(*, annotation_value: Any, answer_type: str) -> Tuple[str, str]:
    """Build deterministic `answer_and_annotation` and `answer_only` JSON examples."""
    answer_value = _example_answer_value(str(answer_type))
    example_answer_only = {"answer": answer_value}
    example_answer_and_annotation = {
        "annotation": _example_like(annotation_value, index=0),
        "answer": answer_value,
    }
    return (
        json.dumps(example_answer_and_annotation, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
        json.dumps(example_answer_only, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
    )


def dump_prompt_json_examples(
    *,
    annotation: Any,
    answer: Any,
    ensure_ascii: bool = True,
) -> Tuple[str, str]:
    """Dump explicit prompt JSON examples with the canonical compact formatting."""
    return (
        json.dumps(
            {"annotation": annotation, "answer": answer},
            ensure_ascii=bool(ensure_ascii),
            allow_nan=False,
            separators=(",", ":"),
        ),
        json.dumps(
            {"answer": answer},
            ensure_ascii=bool(ensure_ascii),
            allow_nan=False,
            separators=(",", ":"),
        ),
    )


def build_keyed_point_prompt_json_examples(
    *,
    annotation_keys: Sequence[str],
    answer: Any,
    ensure_ascii: bool = True,
) -> Tuple[str, str]:
    """Build compact examples for keyed pixel-point annotation contracts."""

    annotation: Dict[str, List[int]] = {}
    for index, key in enumerate(annotation_keys):
        row = int(index) // 4
        col = int(index) % 4
        annotation[str(key)] = [
            int(140 + col * 90 + (row % 2) * 20),
            int(160 + row * 72),
        ]
    return dump_prompt_json_examples(
        annotation=annotation,
        answer=answer,
        ensure_ascii=bool(ensure_ascii),
    )


def resolve_prompt_json_examples(
    prompt_defaults: Mapping[str, Any],
    *,
    annotation_value: Any,
    answer_type: str,
) -> Tuple[str, str]:
    """Resolve configured prompt examples with deterministic generated fallback."""
    json_example = prompt_defaults.get("json_example")
    json_example_answer_only = prompt_defaults.get("json_example_answer_only")
    if isinstance(json_example, str) and json_example.strip() and isinstance(json_example_answer_only, str) and json_example_answer_only.strip():
        return str(json_example), str(json_example_answer_only)
    generated_json_example, generated_json_example_answer_only = build_prompt_json_examples(
        annotation_value=annotation_value,
        answer_type=str(answer_type),
    )
    if not (isinstance(json_example, str) and json_example.strip()):
        json_example = str(generated_json_example)
    if not (isinstance(json_example_answer_only, str) and json_example_answer_only.strip()):
        json_example_answer_only = str(generated_json_example_answer_only)
    return str(json_example), str(json_example_answer_only)


__all__ = [
    "build_prompt_json_examples",
    "build_keyed_point_prompt_json_examples",
    "dump_prompt_json_examples",
    "resolve_prompt_json_examples",
]
