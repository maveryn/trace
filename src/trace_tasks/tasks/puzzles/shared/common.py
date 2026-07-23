"""Shared puzzle-domain helpers reused across multiple puzzle families."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ...shared.config_defaults import group_default, split_generation_rendering_prompt_defaults
from ...shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant


def get_int_param(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: int) -> int:
    """Resolve one integer parameter with task params overriding group defaults."""

    return int(params.get(str(key), group_default(defaults, str(key), int(fallback))))


def get_int_range(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
) -> Tuple[int, int]:
    """Resolve and validate one inclusive integer range from params/defaults."""

    low = get_int_param(params, defaults, str(min_key), int(fallback_min))
    high = get_int_param(params, defaults, str(max_key), int(fallback_max))
    if int(low) > int(high):
        raise ValueError(f"{min_key} must be <= {max_key}")
    return int(low), int(high)


def load_puzzle_task_defaults(
    scene_id_defaults: Mapping[str, Any],
    *,
    task_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, float]]:

    defaults = scene_id_defaults if isinstance(scene_id_defaults, Mapping) else {}
    gen_defaults, render_defaults, prompt_defaults = split_generation_rendering_prompt_defaults(
        defaults,
        task_id=str(task_id),
    )
    return gen_defaults, render_defaults, prompt_defaults, {}


def decouple_axis_sampling(
    params: Mapping[str, Any],
    *,
    preceding_axis_size: int,
    explicit_key: str,
) -> Mapping[str, Any]:
    """No-op hook for puzzle helper call sites."""

    _ = int(preceding_axis_size)
    _ = str(explicit_key)
    return params


def resolve_puzzle_axis_variant(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    supported_variants: Sequence[str],
    task_id: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    axis_namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one semantic or visual puzzle axis with deterministic balancing."""

    rng = spawn_rng(int(instance_seed), f"{task_id}.{axis_namespace}")
    selected_variant, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=[str(item) for item in supported_variants],
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    balanced = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        selected_variant=str(selected_variant),
        variant_probabilities=probabilities,
        supported_variants=[str(item) for item in supported_variants],
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=f"{task_id}:{axis_namespace}",
    )
    return str(balanced), {str(key): float(value) for key, value in probabilities.items()}


def projected_puzzle_bbox_annotation(
    bbox_map: Mapping[str, Sequence[float]],
    item_ids: Sequence[str],
) -> Dict[str, Any]:
    """Project ordered puzzle item ids into prompt-facing `bbox_set` annotation."""

    return {
        "bbox_set": [
            list(bbox_map[str(item_id)])
            for item_id in [str(item) for item in item_ids]
            if str(item_id) in bbox_map
        ]
    }


def projected_puzzle_keyed_bbox_annotation(
    bbox_map: Mapping[str, Sequence[float]],
    role_item_ids: Mapping[str, str],
) -> Dict[str, Any]:
    """Project role-bound puzzle item ids into prompt-facing `bbox_map` annotation."""

    keyed_bboxes: Dict[str, list[float]] = {}
    for role, item_id in role_item_ids.items():
        key = str(item_id)
        if key not in bbox_map:
            raise RuntimeError(f"missing bbox annotation for role {role!r}: item id {key!r}")
        keyed_bboxes[str(role)] = list(bbox_map[key])
    return {
        "bbox_map": {str(role): list(bbox) for role, bbox in keyed_bboxes.items()},
        "pixel_bbox_map": {str(role): list(bbox) for role, bbox in keyed_bboxes.items()},
    }


def projected_puzzle_keyed_bbox_set_annotation(
    bbox_map: Mapping[str, Sequence[float]],
    role_item_ids: Mapping[str, Sequence[str]],
) -> Dict[str, Any]:
    """Project role-bound puzzle item id sets into `bbox_set_map` annotation."""

    keyed_bbox_sets: Dict[str, list[list[float]]] = {}
    for role, item_ids in role_item_ids.items():
        role_bboxes: list[list[float]] = []
        for item_id in item_ids:
            key = str(item_id)
            if key not in bbox_map:
                raise RuntimeError(f"missing bbox annotation for role {role!r}: item id {key!r}")
            role_bboxes.append(list(bbox_map[key]))
        keyed_bbox_sets[str(role)] = role_bboxes
    return {
        "type": "bbox_set_map",
        "bbox_set_map": {
            str(role): [list(bbox) for bbox in bboxes]
            for role, bboxes in keyed_bbox_sets.items()
        },
        "pixel_bbox_set_map": {
            str(role): [list(bbox) for bbox in bboxes]
            for role, bboxes in keyed_bbox_sets.items()
        },
    }


__all__ = [
    "decouple_axis_sampling",
    "get_int_param",
    "get_int_range",
    "load_puzzle_task_defaults",
    "projected_puzzle_bbox_annotation",
    "projected_puzzle_keyed_bbox_annotation",
    "projected_puzzle_keyed_bbox_set_annotation",
    "resolve_puzzle_axis_variant",
]
