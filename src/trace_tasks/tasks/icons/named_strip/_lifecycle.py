"""Shared output lifecycle for named-strip icon tasks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...shared.config_defaults import group_default
from ...shared.deterministic_sampling import uniform_probability_map
from ...shared.output_metadata import default_task_versions
from ..shared.icon_task_rendering import resolve_icon_cell_render_params
from ..shared.procedural_named_icons import procedural_named_icon_display_name
from .shared.annotations import selected_run_bbox_set_annotation
from .shared.defaults import DEFAULT_RENDER, SCENE_ID
from .shared.output import named_strip_render_map, named_strip_render_spec
from .shared.prompts import render_named_strip_prompt_artifacts
from .shared.rendering import render_named_strip_scene, serialize_named_strip_icon
from .shared.sampling import (
    build_named_strip_icon_plans,
    named_strip_fill_style_probabilities,
    named_strip_fill_style_support,
    named_strip_shape_support,
    target_run_lengths,
)
from .shared.state import NamedStripScenePayload


@dataclass(frozen=True)
class NamedStripCommonSample:
    """Common sampled operands shared by named-strip objectives."""

    target_shape_id: str
    target_shape_name: str
    answer: int
    strip_length: int
    answer_probabilities: Mapping[str, float]
    strip_length_probabilities: Mapping[str, float]
    shape_support: Tuple[str, ...]
    shape_probabilities: Mapping[str, float]
    fill_style_support: Tuple[str, ...]
    fill_style_probabilities: Mapping[str, float]


@dataclass(frozen=True)
class NamedStripOutputPlan:
    """Task-owned symbolic plan needed for shared named-strip output binding."""

    query_id: str
    prompt_query_key: str
    target_shape_id: str
    target_shape_name: str
    answer: int
    strip_length: int
    shape_ids: Tuple[str, ...]
    selected_indices: Tuple[int, ...]
    target_runs: Tuple[Tuple[int, int], ...]
    query_probabilities: Mapping[str, float]
    answer_probabilities: Mapping[str, float]
    strip_length_probabilities: Mapping[str, float]
    shape_probabilities: Mapping[str, float]
    fill_style_support: Tuple[str, ...]
    fill_style_probabilities: Mapping[str, float]
    scene_kind: str
    row_rule: str
    question_format: str
    selected_indices_key: str
    selected_instance_ids_key: str
    query_params_extra: Mapping[str, Any] = field(default_factory=dict)


def int_bounds_from_defaults(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    low_key: str,
    high_key: str,
    fallback_low: int,
    fallback_high: int,
) -> Tuple[int, int]:
    """Resolve one integer min/max range from params, config, and fallbacks."""

    low = int(params.get(low_key, group_default(gen_defaults, low_key, fallback_low)))
    high = int(params.get(high_key, group_default(gen_defaults, high_key, fallback_high)))
    if low < 0 or high < low:
        raise ValueError(f"invalid {low_key}/{high_key} bounds")
    return int(low), int(high)


def sample_named_strip_common_fields(
    *,
    rng,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    answer_support: Sequence[int],
    explicit_answer_keys: Sequence[str],
    min_strip_length: int,
    fallback_strip_length_min: int,
    fallback_strip_length_max: int,
    min_strip_length_by_answer: Mapping[int, int] | None = None,
) -> NamedStripCommonSample:
    """Sample target shape, answer, strip length, and visual style operands."""

    support = named_strip_shape_support(params, gen_defaults)
    explicit_shape = params.get("shape_id", params.get("target_shape_id"))
    if explicit_shape is not None:
        target_shape_id = str(explicit_shape)
        if target_shape_id not in set(support):
            raise ValueError(f"target shape must be one of {support}")
        shape_probabilities = {str(value): (1.0 if str(value) == str(target_shape_id) else 0.0) for value in support}
    else:
        target_shape_id = str(rng.choice(support))
        probability = 1.0 / float(len(support))
        shape_probabilities = {str(value): float(probability) for value in support}

    explicit_answer = None
    for key in explicit_answer_keys:
        if key in params:
            explicit_answer = params[key]
            break
    if explicit_answer is not None:
        answer = int(explicit_answer)
        if int(answer) not in set(int(value) for value in answer_support):
            raise ValueError("explicit answer is outside configured support")
        answer_probabilities = uniform_probability_map(answer_support, selected=int(answer))
    else:
        answer = int(rng.choice(tuple(int(value) for value in answer_support)))
        answer_probabilities = uniform_probability_map(answer_support)

    strip_min, strip_max = int_bounds_from_defaults(
        params,
        gen_defaults,
        low_key="strip_length_min",
        high_key="strip_length_max",
        fallback_low=int(fallback_strip_length_min),
        fallback_high=int(fallback_strip_length_max),
    )
    min_required = int(min_strip_length)
    if min_strip_length_by_answer is not None and int(answer) in set(int(key) for key in min_strip_length_by_answer):
        min_required = int(min_strip_length_by_answer[int(answer)])
    strip_min = max(int(strip_min), int(min_required))
    if int(strip_min) > int(strip_max):
        raise ValueError("strip length range cannot support the requested named-strip query")
    strip_support = tuple(range(int(strip_min), int(strip_max) + 1))
    explicit_strip_length = params.get("strip_length")
    if explicit_strip_length is not None:
        strip_length = int(explicit_strip_length)
        if int(strip_length) not in set(strip_support):
            raise ValueError("explicit strip_length is outside configured support")
        strip_length_probabilities = uniform_probability_map(strip_support, selected=int(strip_length))
    else:
        strip_length = int(rng.choice(strip_support))
        strip_length_probabilities = uniform_probability_map(strip_support)

    fill_style_support = named_strip_fill_style_support(params, gen_defaults)
    fill_style_probabilities = named_strip_fill_style_probabilities(params, gen_defaults, fill_style_support)
    return NamedStripCommonSample(
        target_shape_id=str(target_shape_id),
        target_shape_name=procedural_named_icon_display_name(str(target_shape_id)),
        answer=int(answer),
        strip_length=int(strip_length),
        answer_probabilities=dict(answer_probabilities),
        strip_length_probabilities=dict(strip_length_probabilities),
        shape_support=tuple(str(value) for value in support),
        shape_probabilities=dict(shape_probabilities),
        fill_style_support=tuple(str(value) for value in fill_style_support),
        fill_style_probabilities=dict(fill_style_probabilities),
    )


def _render_named_strip_with_retries(
    *,
    task_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    render_defaults: Mapping[str, Any],
    plan: NamedStripOutputPlan,
) -> tuple[Mapping[str, Any], NamedStripScenePayload]:
    """Render one named-strip plan, retrying only visual placement/style failures."""

    render_params = resolve_icon_cell_render_params(
        params=params,
        render_defaults=render_defaults,
        fallback_defaults=DEFAULT_RENDER,
        instance_seed=int(instance_seed),
    )
    if int(render_params["cell_box_width_min_px"]) > int(render_params["cell_box_width_max_px"]):
        raise ValueError("cell_box_width_min_px must be <= cell_box_width_max_px")
    if int(render_params["cell_box_height_min_px"]) > int(render_params["cell_box_height_max_px"]):
        raise ValueError("cell_box_height_min_px must be <= cell_box_height_max_px")

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            scene_rng = spawn_rng(int(instance_seed), f"{task_id}:scene", int(attempt))
            icon_plans, sampled_palette_rgb = build_named_strip_icon_plans(
                shape_ids=plan.shape_ids,
                fill_style_support=plan.fill_style_support,
                fill_style_probabilities=plan.fill_style_probabilities,
                instance_seed=int(instance_seed),
                render_params=render_params,
                rng=scene_rng,
            )
            scene = render_named_strip_scene(
                strip_length=int(plan.strip_length),
                target_shape_id=str(plan.target_shape_id),
                selected_run_indices=plan.selected_indices,
                plans=icon_plans,
                render_params=render_params,
                rng=scene_rng,
                sampled_palette_rgb=sampled_palette_rgb,
            )
            return render_params, scene
        except Exception as exc:  # pragma: no cover - exercised by smoke tests.
            last_error = exc
    raise RuntimeError(f"could not generate {task_id}: {last_error}") from last_error


def build_named_strip_task_output(
    *,
    task_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    plan: NamedStripOutputPlan,
    shape_support: Sequence[str],
) -> TaskOutput:
    """Bind shared named-strip rendering, prompt, trace, answer, and annotation."""

    render_params, scene = _render_named_strip_with_retries(
        task_id=str(task_id),
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
        render_defaults=render_defaults,
        plan=plan,
    )
    annotation_payload = selected_run_bbox_set_annotation(scene.icons, expected_count=int(plan.answer))
    selected_instance_ids = tuple(
        str(icon.instance_id)
        for icon in scene.icons
        if bool(icon.is_selected_run_member)
    )
    shape_counts = dict(Counter(str(icon.shape_id) for icon in scene.icons))

    prompt_defaults_required, prompt_artifacts = render_named_strip_prompt_artifacts(
        instance_seed=int(instance_seed),
        prompt_defaults=prompt_defaults,
        prompt_query_key=str(plan.prompt_query_key),
        target_shape_name=str(plan.target_shape_name),
    )

    serialized_icons = [serialize_named_strip_icon(icon) for icon in scene.icons]
    run_lengths = target_run_lengths(plan.target_runs)
    render_map = named_strip_render_map(icons=scene.icons, selected_instance_ids=selected_instance_ids)
    render_map[str(plan.selected_instance_ids_key)] = list(selected_instance_ids)
    query_params = {
        "target_shape_id": str(plan.target_shape_id),
        "target_shape_name": str(plan.target_shape_name),
        "answer": int(plan.answer),
        "strip_length": int(plan.strip_length),
        "query_id_probabilities": dict(plan.query_probabilities),
        "answer_probabilities": dict(plan.answer_probabilities),
        "strip_length_probabilities": dict(plan.strip_length_probabilities),
        "shape_id_support": [str(value) for value in shape_support],
        "shape_probabilities": dict(plan.shape_probabilities),
        "named_icon_fill_style_support": list(plan.fill_style_support),
        "fill_style_probabilities": dict(plan.fill_style_probabilities),
        **dict(plan.query_params_extra),
    }
    target_run_records = [
        {"start_index": int(start), "end_index": int(end), "length": int(end) - int(start) + 1}
        for start, end in plan.target_runs
    ]
    selected_indices = [int(index) for index in plan.selected_indices]
    trace_payload = {
        "scene_ir": {
            "scene_kind": str(plan.scene_kind),
            "scene_id": SCENE_ID,
            "entities": [
                *[dict(cell) for cell in scene.cells],
                *serialized_icons,
            ],
            "relations": {
                "row_rule": str(plan.row_rule),
                "target_shape_id": str(plan.target_shape_id),
                "target_shape_name": str(plan.target_shape_name),
                "shape_counts": {str(key): int(value) for key, value in shape_counts.items()},
                "target_runs": target_run_records,
                str(plan.selected_indices_key): selected_indices,
            },
            "frames": {
                "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                "panels": dict(scene.panel_geometry),
            },
        },
        "query_spec": {
            "query_id": str(plan.query_id),
            "template_id": str(prompt_defaults_required["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": query_params,
        },
        "render_spec": named_strip_render_spec(
            render_params=render_params,
            panel_geometry=scene.panel_geometry,
            sampled_palette_rgb=scene.sampled_palette_rgb,
            cell_box_width_px=int(scene.cell_box_width_px),
            cell_box_height_px=int(scene.cell_box_height_px),
            fill_style_support=plan.fill_style_support,
        ),
        "render_map": render_map,
        "execution_trace": {
            "scene_variant": "single_panel_named_strip_row",
            "query_id": str(plan.query_id),
            "question_format": str(plan.question_format),
            "target_shape_id": str(plan.target_shape_id),
            "target_shape_name": str(plan.target_shape_name),
            "strip_length": int(plan.strip_length),
            "shape_ids_by_cell": [str(value) for value in plan.shape_ids],
            "target_runs": target_run_records,
            "target_run_lengths": [int(value) for value in run_lengths],
            str(plan.selected_indices_key): selected_indices,
            str(plan.selected_instance_ids_key): list(selected_instance_ids),
            "answer": int(plan.answer),
        },
        "witness_symbolic": {
            "query_id": str(plan.query_id),
            "target_shape_id": str(plan.target_shape_id),
            "target_shape_name": str(plan.target_shape_name),
            "answer": int(plan.answer),
            str(plan.selected_indices_key): selected_indices,
            str(plan.selected_instance_ids_key): list(selected_instance_ids),
        },
        "projected_annotation": dict(annotation_payload["projected_annotation"]),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=TypedValue(type="integer", value=int(plan.answer)),
        annotation_gt=TypedValue(type=str(annotation_payload["annotation_type"]), value=list(annotation_payload["annotation_value"])),
        image=scene.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(plan.query_id),
        prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
    )


__all__ = [
    "NamedStripCommonSample",
    "NamedStripOutputPlan",
    "build_named_strip_task_output",
    "int_bounds_from_defaults",
    "sample_named_strip_common_fields",
]
