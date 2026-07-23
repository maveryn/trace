"""Shared Trace answer/annotation reward scoring."""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Callable, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

from .reward_contracts import resolve_annotation_reward_contract_id


_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_FINAL_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```\s*\Z", re.DOTALL | re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

_TRACE_ANSWER_SCORING_MODES = {"exact_json", "legacy_strict"}
_TRACE_ANNOTATION_REWARD_FORMULAS = {"gated", "additive"}
_POINT_MATCH_FALLBACK_HALF_LIFE_PX = 32.0
_POINT_MATCH_HALF_LIFE_FRACTION = 0.035
_POINT_MATCH_HALF_LIFE_MIN_PX = 20.0
_POINT_MATCH_HALF_LIFE_MAX_PX = 80.0
_TRACE_OUTPUT_MODE_ANSWER = "answer"
_TRACE_OUTPUT_MODE_ANSWER_AND_ANNOTATION = "answer_and_annotation"
_DEFAULT_TRACE_OUTPUT_MODE = _TRACE_OUTPUT_MODE_ANSWER
_TRACE_OUTPUT_MODE_ALIASES = {
    "answer": _TRACE_OUTPUT_MODE_ANSWER,
    "answer_only": _TRACE_OUTPUT_MODE_ANSWER,
    "annotation": _TRACE_OUTPUT_MODE_ANSWER_AND_ANNOTATION,
    "answer_and_annotation": _TRACE_OUTPUT_MODE_ANSWER_AND_ANNOTATION,
}
TRACE_ANNOTATION_LOG_TYPES = (
    "bbox",
    "bbox_sequence",
    "bbox_set",
    "bbox_map",
    "bbox_set_map",
    "point_map",
    "point_set_map",
    "point",
    "segment",
    "segment_set",
    "point_sequence",
    "point_set",
)


def _normalize_trace_output_mode(trace_output_mode: str | None) -> str:
    raw_mode = (trace_output_mode or _DEFAULT_TRACE_OUTPUT_MODE).strip().lower()
    if raw_mode in {"", "auto"}:
        return _DEFAULT_TRACE_OUTPUT_MODE
    normalized_mode = _TRACE_OUTPUT_MODE_ALIASES.get(raw_mode)
    if normalized_mode is None:
        raise ValueError(
            "Trace output mode must be one of {'answer', 'answer_only', 'answer_and_annotation'}; "
            "shorthand alias 'annotation' is also accepted. "
            f"got {trace_output_mode!r}"
        )
    return normalized_mode


def _is_non_string_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _parse_json_like(value: Any) -> Any | None:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            try:
                return ast.literal_eval(text)
            except Exception:
                return None
    return value


def _extract_brace_object_spans(text: str) -> list[tuple[int, int, str]]:
    objects: list[tuple[int, int, str]] = []
    depth = 0
    start_index: int | None = None
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue
        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_index is not None:
                objects.append((start_index, index + 1, text[start_index : index + 1]))
                start_index = None

    return objects


def _extract_brace_objects(text: str) -> list[str]:
    return [obj for _, _, obj in _extract_brace_object_spans(text)]


def _extract_final_json_object(response: str) -> tuple[str | None, bool]:
    stripped = response.rstrip()
    if not stripped:
        return None, False
    final_code_block = _FINAL_CODE_BLOCK_RE.search(stripped)
    if final_code_block is not None:
        content = final_code_block.group(1).strip()
        if content:
            return content, True
    for start, end, candidate in reversed(_extract_brace_object_spans(stripped)):
        if end != len(stripped):
            continue
        prefix = stripped[:start].rstrip()
        if prefix.endswith("```"):
            return candidate.strip(), True
        return candidate.strip(), True
    return None, False


def _collect_json_candidates(response: str) -> list[str]:
    candidates: list[str] = []
    for match in _ANSWER_TAG_RE.findall(response):
        if match.strip():
            candidates.append(match.strip())
    for match in _CODE_BLOCK_RE.findall(response):
        if match.strip():
            candidates.append(match.strip())
    for match in _extract_brace_objects(response):
        if match.strip():
            candidates.append(match.strip())
    if response.strip():
        candidates.append(response.strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _normalize_trace_reward_mode(trace_reward_mode: str | None, *, trace_output_mode: str | None = None) -> str:
    raw_mode = (trace_reward_mode or "").strip().lower()
    if raw_mode in {"", "auto"}:
        return _normalize_trace_output_mode(trace_output_mode)
    return _normalize_trace_output_mode(trace_reward_mode)


def _normalize_trace_answer_scoring(trace_answer_scoring: str | None) -> str:
    normalized = str(trace_answer_scoring or "exact_json").strip().lower()
    aliases = {
        "exact": "exact_json",
        "exact_json": "exact_json",
        "legacy": "legacy_strict",
        "legacy_strict": "legacy_strict",
        "strict": "legacy_strict",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        raise ValueError(
            f"Trace answer scoring must be one of {sorted(_TRACE_ANSWER_SCORING_MODES)!r}; "
            f"aliases {sorted(aliases)!r} are accepted. got {trace_answer_scoring!r}"
        )
    return resolved


def _normalize_trace_annotation_reward_formula(trace_annotation_reward_formula: str | None) -> str:
    normalized = str(trace_annotation_reward_formula or "gated").strip().lower()
    aliases = {
        "answer_gated": "gated",
        "gated": "gated",
        "gated_annotation": "gated",
        "separate": "additive",
        "additive": "additive",
        "linear": "additive",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        raise ValueError(
            "Trace annotation reward formula must be one of "
            f"{sorted(_TRACE_ANNOTATION_REWARD_FORMULAS)!r}; aliases {sorted(aliases)!r} are accepted. "
            f"got {trace_annotation_reward_formula!r}"
        )
    return resolved


def _required_trace_answer_keys(trace_reward_mode: str) -> set[str]:
    if trace_reward_mode == "answer":
        return {"answer"}
    return {"answer", "annotation"}


def _extract_trace_sections(response: str) -> tuple[str | None, str | None, bool]:
    answer_text, structure_ok = _extract_final_json_object(response)
    if not structure_ok or not answer_text:
        return None, None, False
    return None, answer_text, True


def _parse_trace_answer_payload(answer_block: str) -> dict[str, Any] | None:
    parsed = _parse_json_like(answer_block)
    if isinstance(parsed, dict):
        return parsed
    for candidate in _collect_json_candidates(answer_block):
        parsed = _parse_json_like(candidate)
        if isinstance(parsed, dict):
            return parsed
    return None


def evaluate_trace_response_format(
    response: str,
    *,
    trace_reward_mode: str = "answer_and_annotation",
) -> dict[str, Any]:
    normalized_mode = _normalize_trace_reward_mode(trace_reward_mode)
    think_text, answer_block, structure_ok = _extract_trace_sections(response)
    if not structure_ok:
        return {
            "format": 0.0,
            "structure_ok": False,
            "json_ok": False,
            "schema_ok": False,
            "think_text": None,
            "answer_block": None,
            "payload": None,
        }

    payload = _parse_json_like(answer_block or "")
    if not isinstance(payload, dict):
        return {
            "format": 0.0,
            "structure_ok": True,
            "json_ok": False,
            "schema_ok": False,
            "think_text": think_text,
            "answer_block": answer_block,
            "payload": None,
        }

    schema_ok = set(payload.keys()) == _required_trace_answer_keys(normalized_mode)
    return {
        "format": 1.0 if schema_ok else 0.0,
        "structure_ok": True,
        "json_ok": True,
        "schema_ok": schema_ok,
        "think_text": think_text,
        "answer_block": answer_block,
        "payload": payload,
    }


def _normalize_scalar_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        as_float = float(value)
        if not np.isfinite(as_float):
            return None
        if as_float.is_integer():
            return int(as_float)
        return as_float
    return None


def _normalize_image_size(value: Any) -> tuple[float, float] | None:
    parsed = _parse_json_like(value)
    if isinstance(parsed, dict):
        for key in ("source_image_size", "image_size", "image_sizes", "canvas_size", "img_size"):
            if key in parsed:
                nested = _normalize_image_size(parsed.get(key))
                if nested is not None:
                    return nested
        width = None
        height = None
        for key in ("width", "w", "image_width", "canvas_width"):
            if key in parsed:
                width = _normalize_scalar_number(parsed.get(key))
                break
        for key in ("height", "h", "image_height", "canvas_height"):
            if key in parsed:
                height = _normalize_scalar_number(parsed.get(key))
                break
        if width is not None and height is not None and float(width) > 0.0 and float(height) > 0.0:
            return float(width), float(height)
        return None

    if _is_non_string_sequence(parsed):
        items = list(parsed)
        if len(items) == 2:
            width = _normalize_scalar_number(items[0])
            height = _normalize_scalar_number(items[1])
            if width is not None and height is not None and float(width) > 0.0 and float(height) > 0.0:
                return float(width), float(height)
        for item in items:
            nested = _normalize_image_size(item)
            if nested is not None:
                return nested
    return None


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(min(max(float(value), float(lower)), float(upper)))


def _resolve_point_half_life_px(
    *,
    point_half_life_px: float | None,
    image_size: Any | None = None,
    image_sizes: Any | None = None,
    metadata: Any | None = None,
    extra_info: Any | None = None,
) -> float:
    if point_half_life_px is not None:
        resolved = float(point_half_life_px)
        if resolved <= 0.0:
            raise ValueError(f"Trace point_half_life_px must be positive, got {point_half_life_px}")
        return resolved

    for candidate in (image_size, image_sizes, metadata, extra_info):
        normalized = _normalize_image_size(candidate)
        if normalized is None:
            continue
        width, height = normalized
        diagonal = math.hypot(float(width), float(height))
        if diagonal > 0.0:
            return _clamp(
                _POINT_MATCH_HALF_LIFE_FRACTION * diagonal,
                _POINT_MATCH_HALF_LIFE_MIN_PX,
                _POINT_MATCH_HALF_LIFE_MAX_PX,
            )
    return float(_POINT_MATCH_FALLBACK_HALF_LIFE_PX)


def _canonical_scalar_symbol(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (Integral, Real)) and not isinstance(value, bool):
        normalized = _normalize_scalar_number(value)
        return str(normalized)
    if isinstance(value, str):
        return value.strip().lower()
    if value is Ellipsis:
        return "..."
    try:
        return json.dumps(_canonical_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _canonical_jsonable(value: Any) -> Any:
    if value is Ellipsis:
        return "..."
    if isinstance(value, bool):
        return value
    if isinstance(value, np.generic):
        return _canonical_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _canonical_jsonable(value.tolist())
    if isinstance(value, (Integral, Real)) and not isinstance(value, bool):
        return _normalize_scalar_number(value)
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, dict):
        items: list[tuple[str, Any]] = []
        for key, item_value in value.items():
            items.append((_canonical_scalar_symbol(key), _canonical_jsonable(item_value)))
        return {key: item_value for key, item_value in sorted(items, key=lambda kv: kv[0])}
    if isinstance(value, (set, frozenset)):
        normalized_items = [_canonical_jsonable(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
    if _is_non_string_sequence(value):
        return [_canonical_jsonable(item) for item in list(value)]
    return value


def _serialize_candidate(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is Ellipsis:
        return "..."
    try:
        return json.dumps(
            _canonical_jsonable(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return str(value)


def _normalize_coord_pair(value: Any) -> tuple[float | int, float | int] | None:
    if not _is_non_string_sequence(value):
        return None
    items = list(value)
    if len(items) != 2:
        return None
    x = _normalize_scalar_number(items[0])
    y = _normalize_scalar_number(items[1])
    if x is None or y is None:
        return None
    return (x, y)


def _normalize_point(value: Any) -> tuple[float, float] | None:
    parsed = _parse_json_like(value)
    point = _normalize_coord_pair(parsed)
    if point is None:
        return None
    return (float(point[0]), float(point[1]))


def _normalize_point_list(value: Any) -> list[tuple[float, float]] | None:
    parsed = _parse_json_like(value)
    single = _normalize_coord_pair(parsed)
    if single is not None:
        return [(float(single[0]), float(single[1]))]
    if not _is_non_string_sequence(parsed):
        return None
    normalized: list[tuple[float, float]] = []
    for item in list(parsed):
        coord = _normalize_coord_pair(item)
        if coord is None:
            return None
        normalized.append((float(coord[0]), float(coord[1])))
    return normalized


def _normalize_segment(value: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    parsed = _parse_json_like(value)
    if not _is_non_string_sequence(parsed):
        return None
    endpoints = list(parsed)
    if len(endpoints) != 2:
        return None
    left = _normalize_coord_pair(endpoints[0])
    right = _normalize_coord_pair(endpoints[1])
    if left is None or right is None:
        return None
    return ((float(left[0]), float(left[1])), (float(right[0]), float(right[1])))


def _normalize_segment_list(value: Any) -> list[tuple[tuple[float, float], tuple[float, float]]] | None:
    parsed = _parse_json_like(value)
    if not _is_non_string_sequence(parsed):
        return None
    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for item in list(parsed):
        if not _is_non_string_sequence(item):
            return None
        endpoints = list(item)
        if len(endpoints) != 2:
            return None
        left = _normalize_coord_pair(endpoints[0])
        right = _normalize_coord_pair(endpoints[1])
        if left is None or right is None:
            return None
        pairs.append(((float(left[0]), float(left[1])), (float(right[0]), float(right[1]))))
    return pairs


def _normalize_bbox(value: Any) -> list[float] | None:
    if not _is_non_string_sequence(value):
        return None
    items = list(value)
    if len(items) != 4:
        return None
    coords: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, Real):
            return None
        coord = float(item)
        if not np.isfinite(coord):
            return None
        coords.append(coord)
    x0, y0, x1, y1 = coords
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    if left == right or top == bottom:
        return None
    return [left, top, right, bottom]


def _normalize_bbox_scalar(value: Any) -> list[float] | None:
    parsed = _parse_json_like(value)
    return _normalize_bbox(parsed)


def _normalize_bbox_set(value: Any) -> list[list[float]] | None:
    parsed = _parse_json_like(value)
    if parsed is None:
        return None
    single = _normalize_bbox(parsed)
    if single is not None:
        return [single]
    if isinstance(parsed, dict) and "bboxes" in parsed:
        parsed = parsed.get("bboxes")
    if not _is_non_string_sequence(parsed):
        return None
    out: list[list[float]] = []
    for item in list(parsed):
        bbox = _normalize_bbox(item)
        if bbox is None:
            return None
        out.append(bbox)
    return out


def _normalize_map_key(key: Any) -> str | None:
    normalized = str(key).strip()
    if not normalized:
        return None
    return normalized


def _normalize_point_map(value: Any) -> dict[str, tuple[float, float]] | None:
    parsed = _parse_json_like(value)
    if not isinstance(parsed, dict):
        return None
    out: dict[str, tuple[float, float]] = {}
    for key, item in parsed.items():
        normalized_key = _normalize_map_key(key)
        if normalized_key is None or normalized_key in out:
            return None
        point = _normalize_coord_pair(item)
        if point is None:
            return None
        out[normalized_key] = (float(point[0]), float(point[1]))
    return out


def _normalize_point_set_map(value: Any) -> dict[str, list[tuple[float, float]]] | None:
    parsed = _parse_json_like(value)
    if not isinstance(parsed, dict):
        return None
    out: dict[str, list[tuple[float, float]]] = {}
    for key, item in parsed.items():
        normalized_key = _normalize_map_key(key)
        if normalized_key is None or normalized_key in out:
            return None
        points = _normalize_point_list(item)
        if points is None:
            return None
        out[normalized_key] = points
    return out


def _normalize_bbox_map(value: Any) -> dict[str, list[float]] | None:
    parsed = _parse_json_like(value)
    if not isinstance(parsed, dict):
        return None
    out: dict[str, list[float]] = {}
    for key, item in parsed.items():
        normalized_key = _normalize_map_key(key)
        if normalized_key is None or normalized_key in out:
            return None
        bbox = _normalize_bbox(item)
        if bbox is None:
            return None
        out[normalized_key] = bbox
    return out


def _normalize_bbox_set_map(value: Any) -> dict[str, list[list[float]]] | None:
    parsed = _parse_json_like(value)
    if not isinstance(parsed, dict):
        return None
    out: dict[str, list[list[float]]] = {}
    for key, item in parsed.items():
        normalized_key = _normalize_map_key(key)
        if normalized_key is None or normalized_key in out:
            return None
        bboxes = _normalize_bbox_set(item)
        if bboxes is None:
            return None
        out[normalized_key] = bboxes
    return out


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _greedy_assignment(iou_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if iou_matrix.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    remaining_rows = set(range(iou_matrix.shape[0]))
    remaining_cols = set(range(iou_matrix.shape[1]))
    selected_rows: list[int] = []
    selected_cols: list[int] = []
    while remaining_rows and remaining_cols:
        best_pair: tuple[int, int] | None = None
        best_iou = -1.0
        for row in remaining_rows:
            for col in remaining_cols:
                iou = float(iou_matrix[row, col])
                if iou > best_iou:
                    best_iou = iou
                    best_pair = (row, col)
        if best_pair is None:
            break
        row, col = best_pair
        selected_rows.append(row)
        selected_cols.append(col)
        remaining_rows.remove(row)
        remaining_cols.remove(col)
    return np.array(selected_rows, dtype=np.int64), np.array(selected_cols, dtype=np.int64)


def _score_bbox_set_soft_iou(
    pred_boxes: list[list[float]],
    gt_boxes: list[list[float]],
) -> tuple[float, int, float]:
    if not pred_boxes and not gt_boxes:
        return 1.0, 0, 0.0
    if not pred_boxes or not gt_boxes:
        return 0.0, 0, 0.0

    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    for row, pred_box in enumerate(pred_boxes):
        for col, gt_box in enumerate(gt_boxes):
            iou_matrix[row, col] = _bbox_iou(pred_box, gt_box)

    if linear_sum_assignment is None:
        row_ind, col_ind = _greedy_assignment(iou_matrix)
    else:
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matched_ious: list[float] = []
    for row, col in zip(row_ind, col_ind):
        iou = float(iou_matrix[row, col])
        matched_ious.append(iou)
    if not matched_ious:
        return 0.0, 0, 0.0
    score = float(sum(matched_ious) / max(len(pred_boxes), len(gt_boxes), 1))
    mean_iou = float(sum(matched_ious) / len(matched_ious))
    return score, len(matched_ious), mean_iou


def _score_bbox_sequence_soft_iou(
    pred_boxes: list[list[float]],
    gt_boxes: list[list[float]],
) -> tuple[float, int, float]:
    if not pred_boxes and not gt_boxes:
        return 1.0, 0, 0.0
    if not pred_boxes or not gt_boxes:
        return 0.0, 0, 0.0
    ious = [_bbox_iou(pred_box, gt_box) for pred_box, gt_box in zip(pred_boxes, gt_boxes)]
    if not ious:
        return 0.0, 0, 0.0
    score = float(sum(ious) / max(len(pred_boxes), len(gt_boxes), 1))
    mean_iou = float(sum(ious) / len(ious))
    return score, len(ious), mean_iou


def _score_bbox_map_soft_iou(
    pred_boxes: dict[str, list[float]],
    gt_boxes: dict[str, list[float]],
) -> tuple[float, int, float]:
    if not pred_boxes and not gt_boxes:
        return 1.0, 0, 0.0
    key_union = set(pred_boxes) | set(gt_boxes)
    if not key_union:
        return 1.0, 0, 0.0
    shared_keys = sorted(set(pred_boxes) & set(gt_boxes))
    ious = [_bbox_iou(pred_boxes[key], gt_boxes[key]) for key in shared_keys]
    if not ious:
        return 0.0, 0, 0.0
    score = float(sum(ious) / len(key_union))
    mean_iou = float(sum(ious) / len(ious))
    return score, len(ious), mean_iou


def _score_bbox_set_map_soft_iou(
    pred_box_sets: dict[str, list[list[float]]],
    gt_box_sets: dict[str, list[list[float]]],
) -> tuple[float, int, float, int]:
    if not pred_box_sets and not gt_box_sets:
        return 1.0, 0, 0.0, 0
    key_union = set(pred_box_sets) | set(gt_box_sets)
    if not key_union:
        return 1.0, 0, 0.0, 0
    shared_keys = sorted(set(pred_box_sets) & set(gt_box_sets))
    key_scores: list[float] = []
    iou_means: list[float] = []
    total_assigned = 0
    for key in shared_keys:
        key_score, assigned_count, assigned_iou_mean = _score_bbox_set_soft_iou(
            pred_box_sets[key],
            gt_box_sets[key],
        )
        key_scores.append(float(key_score))
        total_assigned += int(assigned_count)
        if int(assigned_count) > 0:
            iou_means.append(float(assigned_iou_mean))
    if not key_scores:
        return 0.0, 0, 0.0, 0
    score = float(sum(key_scores) / len(key_union))
    mean_iou = float(sum(iou_means) / len(iou_means)) if iou_means else 0.0
    return score, len(shared_keys), mean_iou, total_assigned


def _point_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return float(((float(point_a[0]) - float(point_b[0])) ** 2 + (float(point_a[1]) - float(point_b[1])) ** 2) ** 0.5)


def _point_soft_similarity(distance: float, *, half_life_px: float) -> float:
    half_life = max(float(half_life_px), 1e-6)
    return float(math.exp(-math.log(2.0) * ((float(distance) / half_life) ** 2)))


def _score_point_set_soft_distance(
    pred_points: list[tuple[float, float]],
    gt_points: list[tuple[float, float]],
    *,
    half_life_px: float,
) -> tuple[float, int, float]:
    if not pred_points and not gt_points:
        return 1.0, 0, 0.0
    if not pred_points or not gt_points:
        return 0.0, 0, 0.0

    distance_matrix = np.zeros((len(pred_points), len(gt_points)), dtype=np.float32)
    for row, pred_point in enumerate(pred_points):
        for col, gt_point in enumerate(gt_points):
            distance_matrix[row, col] = _point_distance(pred_point, gt_point)

    if linear_sum_assignment is None:
        row_ind, col_ind = _greedy_assignment(-distance_matrix)
    else:
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

    similarities = [
        _point_soft_similarity(float(distance_matrix[row, col]), half_life_px=half_life_px)
        for row, col in zip(row_ind, col_ind)
    ]
    if not similarities:
        return 0.0, 0, 0.0
    score = float(sum(similarities) / max(len(pred_points), len(gt_points), 1))
    mean_similarity = float(sum(similarities) / len(similarities))
    return score, len(similarities), mean_similarity


def _score_point_sequence_soft_distance(
    pred_points: list[tuple[float, float]],
    gt_points: list[tuple[float, float]],
    *,
    half_life_px: float,
) -> tuple[float, int, float]:
    if not pred_points and not gt_points:
        return 1.0, 0, 0.0
    if not pred_points or not gt_points:
        return 0.0, 0, 0.0
    similarities = []
    for pred_point, gt_point in zip(pred_points, gt_points):
        distance = _point_distance(pred_point, gt_point)
        similarities.append(_point_soft_similarity(float(distance), half_life_px=half_life_px))
    if not similarities:
        return 0.0, 0, 0.0
    score = float(sum(similarities) / max(len(pred_points), len(gt_points), 1))
    mean_similarity = float(sum(similarities) / len(similarities))
    return score, len(similarities), mean_similarity


def _score_point_map_soft_distance(
    pred_points: dict[str, tuple[float, float]],
    gt_points: dict[str, tuple[float, float]],
    *,
    half_life_px: float,
) -> tuple[float, int, float]:
    if not pred_points and not gt_points:
        return 1.0, 0, 0.0
    key_union = set(pred_points) | set(gt_points)
    if not key_union:
        return 1.0, 0, 0.0
    shared_keys = sorted(set(pred_points) & set(gt_points))
    similarities = [
        _point_soft_similarity(
            _point_distance(pred_points[key], gt_points[key]),
            half_life_px=half_life_px,
        )
        for key in shared_keys
    ]
    if not similarities:
        return 0.0, 0, 0.0
    score = float(sum(similarities) / len(key_union))
    mean_similarity = float(sum(similarities) / len(similarities))
    return score, len(similarities), mean_similarity


def _score_point_set_map_soft_distance(
    pred_point_sets: dict[str, list[tuple[float, float]]],
    gt_point_sets: dict[str, list[tuple[float, float]]],
    *,
    half_life_px: float,
) -> tuple[float, int, float, int]:
    if not pred_point_sets and not gt_point_sets:
        return 1.0, 0, 0.0, 0
    key_union = set(pred_point_sets) | set(gt_point_sets)
    if not key_union:
        return 1.0, 0, 0.0, 0
    shared_keys = sorted(set(pred_point_sets) & set(gt_point_sets))
    key_scores: list[float] = []
    similarity_means: list[float] = []
    total_assigned = 0
    for key in shared_keys:
        key_score, assigned_count, assigned_similarity_mean = _score_point_set_soft_distance(
            pred_point_sets[key],
            gt_point_sets[key],
            half_life_px=half_life_px,
        )
        key_scores.append(float(key_score))
        total_assigned += int(assigned_count)
        if int(assigned_count) > 0:
            similarity_means.append(float(assigned_similarity_mean))
    if not key_scores:
        return 0.0, 0, 0.0, 0
    score = float(sum(key_scores) / len(key_union))
    mean_similarity = float(sum(similarity_means) / len(similarity_means)) if similarity_means else 0.0
    return score, len(shared_keys), mean_similarity, total_assigned


def _segment_distance(
    pred_pair: tuple[tuple[float, float], tuple[float, float]],
    gt_pair: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    direct = max(
        _point_distance(pred_pair[0], gt_pair[0]),
        _point_distance(pred_pair[1], gt_pair[1]),
    )
    flipped = max(
        _point_distance(pred_pair[0], gt_pair[1]),
        _point_distance(pred_pair[1], gt_pair[0]),
    )
    return float(min(direct, flipped))


def _score_segment_set_soft_distance(
    pred_pairs: list[tuple[tuple[float, float], tuple[float, float]]],
    gt_pairs: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    half_life_px: float,
) -> tuple[float, int, float]:
    if not pred_pairs and not gt_pairs:
        return 1.0, 0, 0.0
    if not pred_pairs or not gt_pairs:
        return 0.0, 0, 0.0

    distance_matrix = np.zeros((len(pred_pairs), len(gt_pairs)), dtype=np.float32)
    for row, pred_pair in enumerate(pred_pairs):
        for col, gt_pair in enumerate(gt_pairs):
            distance_matrix[row, col] = _segment_distance(pred_pair, gt_pair)

    if linear_sum_assignment is None:
        row_ind, col_ind = _greedy_assignment(-distance_matrix)
    else:
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

    similarities = [
        _point_soft_similarity(float(distance_matrix[row, col]), half_life_px=half_life_px)
        for row, col in zip(row_ind, col_ind)
    ]
    if not similarities:
        return 0.0, 0, 0.0
    score = float(sum(similarities) / max(len(pred_pairs), len(gt_pairs), 1))
    mean_similarity = float(sum(similarities) / len(similarities))
    return score, len(similarities), mean_similarity


def _score_trace_answer(
    answer_value: Any,
    answer_gt: dict[str, Any],
    *,
    trace_answer_scoring: str = "exact_json",
    legacy_strict_scorer: Callable[..., tuple[Any, ...]] | None = None,
) -> tuple[float, bool]:
    if not isinstance(answer_gt, dict) or "value" not in answer_gt:
        return 0.0, False
    normalized_scoring = _normalize_trace_answer_scoring(trace_answer_scoring)
    if normalized_scoring == "legacy_strict":
        if legacy_strict_scorer is None:
            raise ValueError("Trace legacy_strict answer scoring requires a legacy_strict_scorer callback")
        answer_text = _serialize_candidate(answer_value)
        score, extracted, _, _ = legacy_strict_scorer(response=answer_text, ground_truth=answer_gt.get("value"))
        return float(score), bool(extracted)

    answer_type = str(answer_gt.get("type", "")).strip()
    if isinstance(answer_value, str) and answer_type == "string":
        parsed_answer_value = answer_value
    else:
        parsed_answer_value = _parse_json_like(answer_value) if isinstance(answer_value, str) else answer_value
        if parsed_answer_value is None and isinstance(answer_value, str):
            parsed_answer_value = answer_value
    normalized_pred = _canonical_jsonable(parsed_answer_value)
    normalized_gt = _canonical_jsonable(answer_gt.get("value"))
    return (1.0 if normalized_pred == normalized_gt else 0.0), True


def _score_trace_annotation(
    annotation_value: Any,
    annotation_gt: dict[str, Any],
    reward_contract: dict[str, Any],
    *,
    point_half_life_px: float,
) -> tuple[float, bool, dict[str, Any]]:
    if not isinstance(annotation_gt, dict) or "type" not in annotation_gt:
        return 0.0, False, {"reason": "missing_annotation_gt"}
    if not isinstance(reward_contract, dict):
        return 0.0, False, {"reason": "missing_reward_contract"}

    annotation_type = str(annotation_gt.get("type", "")).strip()
    annotation_gt_value = annotation_gt.get("value")
    annotation_contract = reward_contract.get("annotation") if isinstance(reward_contract.get("annotation"), dict) else {}
    annotation_contract_id = str(annotation_contract.get("id", "")).strip()
    annotation_contract_type = str(annotation_contract.get("type", "")).strip()
    try:
        expected_contract_id = resolve_annotation_reward_contract_id(annotation_type)
    except ValueError:
        return 0.0, False, {"reason": f"unsupported_annotation_type:{annotation_type}"}
    if annotation_contract_type and annotation_contract_type != annotation_type:
        return 0.0, False, {"reason": "annotation_contract_type_mismatch"}
    if annotation_contract_id != expected_contract_id:
        return 0.0, False, {"reason": "annotation_contract_id_mismatch"}

    if annotation_type == "point":
        pred = _normalize_point(annotation_value)
        gt = _normalize_point(annotation_gt_value)
        if pred is None or gt is None:
            return 0.0, False, {"reason": "point_parse_failed"}
        distance = _point_distance(pred, gt)
        similarity = _point_soft_similarity(distance, half_life_px=point_half_life_px)
        return similarity, True, {
            "pred_size": 1,
            "gt_size": 1,
            "assigned_count": 1,
            "assigned_similarity_mean": float(similarity),
            "point_half_life_px": float(point_half_life_px),
        }

    if annotation_type == "point_set":
        pred = _normalize_point_list(annotation_value)
        gt = _normalize_point_list(annotation_gt_value)
        if pred is None or gt is None:
            return 0.0, False, {"reason": "point_set_parse_failed"}
        score, assigned_count, assigned_similarity_mean = _score_point_set_soft_distance(
            pred,
            gt,
            half_life_px=point_half_life_px,
        )
        return score, True, {
            "pred_size": len(pred),
            "gt_size": len(gt),
            "assigned_count": int(assigned_count),
            "assigned_similarity_mean": float(assigned_similarity_mean),
            "point_half_life_px": float(point_half_life_px),
        }

    if annotation_type == "point_sequence":
        pred = _normalize_point_list(annotation_value)
        gt = _normalize_point_list(annotation_gt_value)
        if pred is None or gt is None:
            return 0.0, False, {"reason": "point_sequence_parse_failed"}
        score, assigned_count, assigned_similarity_mean = _score_point_sequence_soft_distance(
            pred,
            gt,
            half_life_px=point_half_life_px,
        )
        return score, True, {
            "pred_len": len(pred),
            "gt_len": len(gt),
            "assigned_count": int(assigned_count),
            "assigned_similarity_mean": float(assigned_similarity_mean),
            "point_half_life_px": float(point_half_life_px),
        }

    if annotation_type == "segment":
        pred = _normalize_segment(annotation_value)
        gt = _normalize_segment(annotation_gt_value)
        if pred is None or gt is None:
            return 0.0, False, {"reason": "segment_parse_failed"}
        distance = _segment_distance(pred, gt)
        similarity = _point_soft_similarity(distance, half_life_px=point_half_life_px)
        return similarity, True, {
            "pred_size": 1,
            "gt_size": 1,
            "assigned_count": 1,
            "assigned_similarity_mean": float(similarity),
            "point_half_life_px": float(point_half_life_px),
        }

    if annotation_type == "segment_set":
        pred = _normalize_segment_list(annotation_value)
        gt = _normalize_segment_list(annotation_gt_value)
        if pred is None or gt is None:
            return 0.0, False, {"reason": "segment_set_parse_failed"}
        score, assigned_count, assigned_similarity_mean = _score_segment_set_soft_distance(
            pred,
            gt,
            half_life_px=point_half_life_px,
        )
        return score, True, {
            "pred_size": len(pred),
            "gt_size": len(gt),
            "assigned_count": int(assigned_count),
            "assigned_similarity_mean": float(assigned_similarity_mean),
            "point_half_life_px": float(point_half_life_px),
        }

    if annotation_type == "bbox":
        pred = _normalize_bbox_scalar(annotation_value)
        gt = _normalize_bbox_scalar(annotation_gt_value)
        if pred is None or gt is None:
            return 0.0, False, {"reason": "bbox_parse_failed"}
        iou = _bbox_iou(pred, gt)
        return iou, True, {
            "pred_size": 1,
            "gt_size": 1,
            "assigned_count": 1,
            "assigned_iou_mean": float(iou),
        }

    if annotation_type == "bbox_set":
        pred = _normalize_bbox_set(annotation_value)
        gt = _normalize_bbox_set(annotation_gt_value)
        if pred is None or gt is None:
            return 0.0, False, {"reason": "bbox_parse_failed"}
        score, assigned_count, assigned_iou_mean = _score_bbox_set_soft_iou(pred, gt)
        return score, True, {
            "pred_size": len(pred),
            "gt_size": len(gt),
            "assigned_count": assigned_count,
            "assigned_iou_mean": assigned_iou_mean,
        }

    if annotation_type == "bbox_sequence":
        pred = _normalize_bbox_set(annotation_value)
        gt = _normalize_bbox_set(annotation_gt_value)
        if pred is None or gt is None:
            return 0.0, False, {"reason": "bbox_sequence_parse_failed"}
        score, assigned_count, assigned_iou_mean = _score_bbox_sequence_soft_iou(pred, gt)
        return score, True, {
            "pred_len": len(pred),
            "gt_len": len(gt),
            "assigned_count": assigned_count,
            "assigned_iou_mean": assigned_iou_mean,
        }

    if annotation_type == "point_map":
        pred = _normalize_point_map(annotation_value)
        gt = _normalize_point_map(annotation_gt_value)
        if pred is None or gt is None:
            return 0.0, False, {"reason": "point_map_parse_failed"}
        score, assigned_count, assigned_similarity_mean = _score_point_map_soft_distance(
            pred,
            gt,
            half_life_px=point_half_life_px,
        )
        return score, True, {
            "pred_size": len(pred),
            "gt_size": len(gt),
            "assigned_count": int(assigned_count),
            "shared_key_count": int(assigned_count),
            "missing_key_count": int(len(set(gt) - set(pred))),
            "extra_key_count": int(len(set(pred) - set(gt))),
            "assigned_similarity_mean": float(assigned_similarity_mean),
            "point_half_life_px": float(point_half_life_px),
        }

    if annotation_type == "point_set_map":
        pred = _normalize_point_set_map(annotation_value)
        gt = _normalize_point_set_map(annotation_gt_value)
        if pred is None or gt is None:
            return 0.0, False, {"reason": "point_set_map_parse_failed"}
        score, shared_key_count, assigned_similarity_mean, assigned_point_count = _score_point_set_map_soft_distance(
            pred,
            gt,
            half_life_px=point_half_life_px,
        )
        return score, True, {
            "pred_size": len(pred),
            "gt_size": len(gt),
            "assigned_count": int(assigned_point_count),
            "shared_key_count": int(shared_key_count),
            "missing_key_count": int(len(set(gt) - set(pred))),
            "extra_key_count": int(len(set(pred) - set(gt))),
            "assigned_similarity_mean": float(assigned_similarity_mean),
            "point_half_life_px": float(point_half_life_px),
        }

    if annotation_type == "bbox_map":
        pred = _normalize_bbox_map(annotation_value)
        gt = _normalize_bbox_map(annotation_gt_value)
        if pred is None or gt is None:
            return 0.0, False, {"reason": "bbox_map_parse_failed"}
        score, assigned_count, assigned_iou_mean = _score_bbox_map_soft_iou(pred, gt)
        return score, True, {
            "pred_size": len(pred),
            "gt_size": len(gt),
            "assigned_count": int(assigned_count),
            "shared_key_count": int(assigned_count),
            "missing_key_count": int(len(set(gt) - set(pred))),
            "extra_key_count": int(len(set(pred) - set(gt))),
            "assigned_iou_mean": float(assigned_iou_mean),
        }

    if annotation_type == "bbox_set_map":
        pred = _normalize_bbox_set_map(annotation_value)
        gt = _normalize_bbox_set_map(annotation_gt_value)
        if pred is None or gt is None:
            return 0.0, False, {"reason": "bbox_set_map_parse_failed"}
        score, shared_key_count, assigned_iou_mean, assigned_bbox_count = _score_bbox_set_map_soft_iou(pred, gt)
        return score, True, {
            "pred_size": len(pred),
            "gt_size": len(gt),
            "assigned_count": int(assigned_bbox_count),
            "shared_key_count": int(shared_key_count),
            "missing_key_count": int(len(set(gt) - set(pred))),
            "extra_key_count": int(len(set(pred) - set(gt))),
            "assigned_iou_mean": float(assigned_iou_mean),
        }

    return 0.0, False, {"reason": f"unsupported_annotation_contract:{annotation_contract_id}"}


def extract_trace_prediction(response: str) -> tuple[Any | None, Any | None, bool]:
    _, answer_block, structure_ok = _extract_trace_sections(response)
    if structure_ok and answer_block is not None:
        payload = _parse_trace_answer_payload(answer_block)
        if isinstance(payload, dict) and ("answer" in payload or "annotation" in payload):
            return payload.get("answer"), payload.get("annotation"), True
    for candidate in reversed(_collect_json_candidates(response)):
        parsed = _parse_json_like(candidate)
        if not isinstance(parsed, dict):
            continue
        if "answer" in parsed or "annotation" in parsed:
            return parsed.get("answer"), parsed.get("annotation"), True
    return None, None, False


def extract_trace_answer_for_scoring(response: str) -> str | None:
    answer_value, _, json_found = extract_trace_prediction(response)
    if not json_found or answer_value is None:
        return None
    return _serialize_candidate(answer_value)


def is_trace_reward_input(reward_input: dict[str, Any]) -> bool:
    return all(key in reward_input for key in ("answer_gt", "annotation_gt", "reward_contract"))


def score_trace_response(
    *,
    response: str,
    answer_gt: dict[str, Any],
    annotation_gt: dict[str, Any],
    reward_contract: dict[str, Any],
    bbox_iou_threshold: float | None = None,
    point_half_life_px: float | None = None,
    image_size: Any | None = None,
    image_sizes: Any | None = None,
    metadata: Any | None = None,
    extra_info: Any | None = None,
    answer_weight: float = 0.5,
    annotation_weight: float = 0.5,
    trace_reward_mode: str = "answer_and_annotation",
    trace_answer_scoring: str = "exact_json",
    trace_annotation_reward_formula: str = "gated",
    format_weight: float = 0.0,
    legacy_strict_scorer: Callable[..., tuple[Any, ...]] | None = None,
) -> dict[str, float]:
    if format_weight < 0.0 or format_weight > 1.0:
        raise ValueError(f"Trace format_weight must be in [0, 1], got {format_weight}")
    resolved_point_half_life_px = _resolve_point_half_life_px(
        point_half_life_px=point_half_life_px,
        image_size=image_size,
        image_sizes=image_sizes,
        metadata=metadata,
        extra_info=extra_info,
    )

    normalized_mode = _normalize_trace_reward_mode(trace_reward_mode)
    normalized_answer_scoring = _normalize_trace_answer_scoring(trace_answer_scoring)
    normalized_annotation_formula = _normalize_trace_annotation_reward_formula(trace_annotation_reward_formula)
    annotation_type = str(annotation_gt.get("type", "")).strip() if isinstance(annotation_gt, dict) else ""
    format_details = evaluate_trace_response_format(response, trace_reward_mode=normalized_mode)
    payload = format_details.get("payload") if isinstance(format_details.get("payload"), dict) else None

    if payload is not None and ("answer" in payload or "annotation" in payload):
        answer_value = payload.get("answer")
        annotation_value = payload.get("annotation")
        json_found = True
    else:
        answer_value, annotation_value, json_found = extract_trace_prediction(response)

    if answer_value is None:
        answer_value = response

    answer_score, answer_parse_ok = _score_trace_answer(
        answer_value,
        answer_gt,
        trace_answer_scoring=normalized_answer_scoring,
        legacy_strict_scorer=legacy_strict_scorer,
    )
    annotation_score = 0.0
    annotation_parse_ok = False
    annotation_details: dict[str, Any] = {}
    if annotation_value is not None:
        annotation_score, annotation_parse_ok, annotation_details = _score_trace_annotation(
            annotation_value,
            annotation_gt,
            reward_contract,
            point_half_life_px=resolved_point_half_life_px,
        )

    total_weight = float(answer_weight + annotation_weight)
    if total_weight <= 0.0:
        raise ValueError("Trace reward weights must sum to a positive value")
    normalized_answer_weight = float(answer_weight / total_weight)
    normalized_annotation_weight = float(annotation_weight / total_weight)
    if normalized_mode == "answer":
        raw_task_reward = float(answer_score)
    elif normalized_annotation_formula == "additive":
        raw_task_reward = float(
            (normalized_answer_weight * answer_score) + (normalized_annotation_weight * annotation_score)
        )
    else:
        raw_task_reward = float(answer_score * (normalized_answer_weight + (normalized_annotation_weight * annotation_score)))

    # Keep format as a logged diagnostic by default. Callers can still opt into
    # additive format reward by passing a positive format_weight.
    effective_task_reward = raw_task_reward
    overall = float(((1.0 - format_weight) * effective_task_reward) + (format_weight * float(format_details["format"])))
    result = {
        "overall": overall,
        "format": float(format_details["format"]),
        "accuracy": float(answer_score),
        "answer_reward": float(answer_score),
        "annotation_reward": float(annotation_score),
        "task_reward_raw": float(raw_task_reward),
        "task_reward_effective": float(effective_task_reward),
        "task_reward_gated": float(effective_task_reward),
        "format_weight": float(format_weight),
        "trace_reward_mode_answer": 1.0 if normalized_mode == "answer" else 0.0,
        "trace_reward_mode_answer_only": 1.0 if normalized_mode == "answer" else 0.0,
        "trace_reward_mode_answer_and_annotation": 1.0 if normalized_mode == "answer_and_annotation" else 0.0,
        "trace_answer_scoring_exact_json": 1.0 if normalized_answer_scoring == "exact_json" else 0.0,
        "trace_answer_scoring_legacy_strict": 1.0 if normalized_answer_scoring == "legacy_strict" else 0.0,
        "trace_annotation_reward_formula_gated": 1.0 if normalized_annotation_formula == "gated" else 0.0,
        "trace_annotation_reward_formula_additive": 1.0 if normalized_annotation_formula == "additive" else 0.0,
        "trace_answer_weight": float(normalized_answer_weight),
        "trace_annotation_weight": float(normalized_annotation_weight),
        "format_structure_ok": 1.0 if format_details["structure_ok"] else 0.0,
        "format_json_ok": 1.0 if format_details["json_ok"] else 0.0,
        "format_schema_ok": 1.0 if format_details["schema_ok"] else 0.0,
        "answer_parse_ok": 1.0 if answer_parse_ok else 0.0,
        "annotation_parse_ok": 1.0 if annotation_parse_ok else 0.0,
        "json_found": 1.0 if json_found else 0.0,
        # Keep zero_reward aligned with task correctness semantics. If callers
        # opt into additive format reward, overall can be positive even when the
        # answer is wrong, so overall <= 0 is not a stable zero-solve signal.
        "zero_reward": 1.0 if effective_task_reward <= 0.0 else 0.0,
        "trace_reward": 1.0,
    }
    for logged_annotation_type in TRACE_ANNOTATION_LOG_TYPES:
        result[f"annotation_type_{logged_annotation_type}"] = 1.0 if annotation_type == logged_annotation_type else 0.0
    for key, value in annotation_details.items():
        if isinstance(value, (int, float)):
            result[f"annotation_{key}"] = float(value)
    return result


__all__ = [
    "TRACE_ANNOTATION_LOG_TYPES",
    "evaluate_trace_response_format",
    "extract_trace_answer_for_scoring",
    "extract_trace_prediction",
    "is_trace_reward_input",
    "score_trace_response",
]
