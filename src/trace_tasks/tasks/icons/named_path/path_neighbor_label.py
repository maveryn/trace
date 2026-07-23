"""Select the labeled icon adjacent to a named icon along a marked path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import (
    group_default,
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ..shared.annotation import icon_bbox_annotation
from ..shared.icon_task_rendering import icon_render_style_trace

from .shared.defaults import NamedPathDefaults, OPTION_LABELS, SCENE_ID
from .shared.rendering import render_named_path_scene, serialize_path_icon
from .shared.sampling import (
    display_shape_name,
    fill_style_probability_map,
    fill_style_support,
    int_support_value,
    sample_icon_plans,
    sample_target_positions,
    shape_support,
    string_probability_map,
)
from .shared.state import PathScenePayload
from .shared.styles import named_path_style_trace, resolve_named_path_render_params


TASK_ID = "task_icons__named_path__path_neighbor_label"
DOMAIN = "icons"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    "after_first_shape_label",
    "before_first_shape_label",
    "after_last_shape_label",
    "before_last_shape_label",
    "after_second_shape_label",
    "before_second_shape_label",
)
NOISE_NAMESPACE = "icons.named_path.path_stop"


@dataclass(frozen=True)
class _SampleSpec:
    """Task-owned symbolic state for one path-neighbor question."""

    query_id: str
    answer_label: str
    target_shape_id: str
    target_shape_name: str
    target_occurrence_count: int
    stop_count: int
    distractor_count: int
    target_positions: Tuple[int, ...]
    query_position_index: int
    answer_position_index: int
    neighbor_direction: str
    option_positions: Tuple[int, ...]
    labels_by_position: Dict[int, str]
    query_probabilities: Dict[str, float]
    answer_label_probabilities: Dict[str, float]
    target_occurrence_count_probabilities: Dict[str, float]
    distractor_count_probabilities: Dict[str, float]
    shape_probabilities: Dict[str, float]
    fill_style_support: Tuple[str, ...]
    fill_style_probabilities: Dict[str, float]


_DEFAULTS = NamedPathDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


def _select_query(instance_seed: int, params: Mapping[str, Any]) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
    """Select and validate one public path-neighbor query."""

    query_params = dict(params)
    legacy_query = query_params.pop("path_neighbor_query", None)
    if legacy_query is not None:
        existing = query_params.get("query_id")
        if existing is not None and str(existing) != str(legacy_query):
            raise ValueError("query_id conflicts with path_neighbor_query")
        query_params["query_id"] = str(legacy_query)
    return select_task_query_id(
        instance_seed=int(instance_seed),
        params=query_params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id=SUPPORTED_QUERY_IDS[0],
        task_id=TASK_ID,
        namespace=f"{TASK_ID}.query",
    )


def _answer_label(rng, params: Mapping[str, Any]) -> Tuple[str, Dict[str, float]]:
    """Resolve the selected visible option label."""

    explicit_label = params.get("answer_label")
    if explicit_label is not None:
        value = str(explicit_label).strip().upper()
        if value not in set(OPTION_LABELS):
            raise ValueError(f"answer_label must be one of {OPTION_LABELS}")
        return value, string_probability_map(OPTION_LABELS, selected=value)
    explicit_index = params.get("answer_index")
    if explicit_index is not None:
        index = int(explicit_index)
        if index < 0 or index >= len(OPTION_LABELS):
            raise ValueError("answer_index must be in 0..5")
        value = str(OPTION_LABELS[index])
        return value, string_probability_map(OPTION_LABELS, selected=value)
    value = str(rng.choice(OPTION_LABELS))
    return value, string_probability_map(OPTION_LABELS)


def _occurrence_rank_for_query(query_id: str, occurrence_count: int) -> int:
    """Return the zero-based target occurrence selected by one public query."""

    if "second" in str(query_id):
        return 1
    if "last" in str(query_id):
        return int(occurrence_count) - 1
    return 0


def _direction_for_query(query_id: str) -> str:
    """Return before/after direction for one public query."""

    return "after" if str(query_id).startswith("after_") else "before"


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any]) -> _SampleSpec:
    """Sample task-owned symbolic state before rendering."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:sample")
    query_id, query_probabilities, task_params = _select_query(int(instance_seed), params)
    answer_label, answer_label_probabilities = _answer_label(rng, task_params)
    candidate_count = int(
        task_params.get("candidate_count", group_default(_GEN_DEFAULTS, "candidate_count", _DEFAULTS.candidate_count))
    )
    if int(candidate_count) != len(OPTION_LABELS):
        raise ValueError("named-path neighbor task requires candidate_count=6")

    distractor_count, distractor_count_probabilities = int_support_value(
        rng,
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        low_key="distractor_count_min",
        high_key="distractor_count_max",
        explicit_key="distractor_count",
        fallback_low=_DEFAULTS.distractor_count_min,
        fallback_high=_DEFAULTS.distractor_count_max,
    )
    target_occurrence_count, target_occurrence_count_probabilities = int_support_value(
        rng,
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        low_key="target_occurrence_count_min",
        high_key="target_occurrence_count_max",
        explicit_key="target_occurrence_count",
        fallback_low=_DEFAULTS.target_occurrence_count_min,
        fallback_high=_DEFAULTS.target_occurrence_count_max,
    )
    if int(target_occurrence_count) < 2:
        raise ValueError("named-path neighbor task requires at least two target occurrences")

    shapes = shape_support(task_params, _GEN_DEFAULTS)
    explicit_shape = task_params.get("shape_id", task_params.get("target_shape_id"))
    if explicit_shape is not None:
        target_shape_id = str(explicit_shape)
        if target_shape_id not in set(shapes):
            raise ValueError(f"unsupported target shape: {target_shape_id}")
    else:
        target_shape_id = str(rng.choice(shapes))

    fill_values = fill_style_support(task_params, _GEN_DEFAULTS)
    fill_probabilities = fill_style_probability_map(task_params, _GEN_DEFAULTS, fill_values)
    stop_count = int(candidate_count) + int(distractor_count) + int(target_occurrence_count)
    occurrence_rank = _occurrence_rank_for_query(str(query_id), int(target_occurrence_count))
    direction = _direction_for_query(str(query_id))
    target_positions, query_position, answer_position = sample_target_positions(
        rng,
        stop_count=int(stop_count),
        target_occurrence_count=int(target_occurrence_count),
        occurrence_rank=int(occurrence_rank),
        neighbor_direction=str(direction),
    )

    non_target_positions = [index for index in range(int(stop_count)) if index not in set(target_positions)]
    endpoint_positions = {0, int(stop_count) - 1}
    other_option_positions = [
        index
        for index in non_target_positions
        if int(index) != int(answer_position) and int(index) not in endpoint_positions
    ]
    rng.shuffle(other_option_positions)
    option_positions = tuple(sorted([int(answer_position), *[int(value) for value in other_option_positions[:5]]]))
    if len(option_positions) != len(OPTION_LABELS):
        raise RuntimeError("failed to assign six path option positions")

    remaining_labels = [str(label) for label in OPTION_LABELS if str(label) != str(answer_label)]
    rng.shuffle(remaining_labels)
    labels_by_position: Dict[int, str] = {}
    for position in option_positions:
        labels_by_position[int(position)] = (
            str(answer_label) if int(position) == int(answer_position) else str(remaining_labels.pop())
        )

    return _SampleSpec(
        query_id=str(query_id),
        answer_label=str(answer_label),
        target_shape_id=str(target_shape_id),
        target_shape_name=display_shape_name(str(target_shape_id)),
        target_occurrence_count=int(target_occurrence_count),
        stop_count=int(stop_count),
        distractor_count=int(distractor_count),
        target_positions=tuple(int(value) for value in target_positions),
        query_position_index=int(query_position),
        answer_position_index=int(answer_position),
        neighbor_direction=str(direction),
        option_positions=tuple(int(value) for value in option_positions),
        labels_by_position=dict(labels_by_position),
        query_probabilities=dict(query_probabilities),
        answer_label_probabilities=dict(answer_label_probabilities),
        target_occurrence_count_probabilities=dict(target_occurrence_count_probabilities),
        distractor_count_probabilities=dict(distractor_count_probabilities),
        shape_probabilities=string_probability_map(
            shapes,
            selected=str(target_shape_id) if explicit_shape is not None else None,
        ),
        fill_style_support=tuple(fill_values),
        fill_style_probabilities=dict(fill_probabilities),
    )


def _prompt_question_key(query_id: str) -> str:
    return f"question_text_{query_id}"


def _prompt_artifacts(*, sample: _SampleSpec, prompt_defaults: Mapping[str, Any], instance_seed: int):
    """Render both prompt output variants."""

    question_key = _prompt_question_key(str(sample.query_id))
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(prompt_defaults["object_description"]),
            "question_text": str(prompt_defaults[question_key]).format(
                target_shape_name=str(sample.target_shape_name)
            ),
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "annotation_hint": str(prompt_defaults["annotation_hint"]),
            "answer_hint": str(prompt_defaults["answer_hint"]),
            "json_example": str(prompt_defaults["json_example"]),
            "json_example_answer_only": str(prompt_defaults["json_example_answer_only"]),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


def _scene_payload(
    *,
    sample: _SampleSpec,
    params: Mapping[str, Any],
    render_params: Mapping[str, Any],
    instance_seed: int,
    scene_rng,
) -> PathScenePayload:
    """Sample icon semantics and render one complete path scene."""

    plans, sampled_palette_rgb = sample_icon_plans(
        rng=scene_rng,
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        render_params=render_params,
        stop_count=int(sample.stop_count),
        target_shape_id=str(sample.target_shape_id),
        target_positions=tuple(sample.target_positions),
        selected_position=int(sample.query_position_index),
        answer_position=int(sample.answer_position_index),
        labels_by_position=dict(sample.labels_by_position),
        fill_style_values=tuple(sample.fill_style_support),
        fill_style_probabilities=dict(sample.fill_style_probabilities),
        noise_namespace=NOISE_NAMESPACE,
    )
    return render_named_path_scene(
        rng=scene_rng,
        plans=tuple(plans),
        answer_label=str(sample.answer_label),
        target_shape_id=str(sample.target_shape_id),
        target_shape_name=str(sample.target_shape_name),
        target_occurrence_count=int(sample.target_occurrence_count),
        stop_count=int(sample.stop_count),
        distractor_count=int(sample.distractor_count),
        selected_position=int(sample.query_position_index),
        answer_position=int(sample.answer_position_index),
        neighbor_direction=str(sample.neighbor_direction),
        target_positions=tuple(sample.target_positions),
        option_positions=tuple(sample.option_positions),
        labels_by_position=dict(sample.labels_by_position),
        sampled_palette_rgb=tuple(sampled_palette_rgb),
        render_params=render_params,
    )


@register_task
class IconsNamedPathPathNeighborLabelTask:
    """Select the labeled icon immediately before/after a named shape along a path."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'topology')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic named-path neighbor instance."""

        render_params = resolve_named_path_render_params(
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            fallback_defaults=_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        sample: _SampleSpec | None = None
        scene_payload: PathScenePayload | None = None
        last_error: Exception | None = None
        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params)
                scene_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:scene", int(attempt))
                scene_payload = _scene_payload(
                    sample=sample,
                    params=params,
                    render_params=render_params,
                    instance_seed=int(instance_seed),
                    scene_rng=scene_rng,
                )
                break
            except Exception as exc:
                last_error = exc
                sample = None
                scene_payload = None
                continue
        if sample is None or scene_payload is None:
            raise RuntimeError(f"failed to generate {TASK_ID} instance") from last_error

        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "object_description",
                _prompt_question_key(str(sample.query_id)),
                "annotation_hint",
                "answer_hint",
                "json_example",
                "json_example_answer_only",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        prompt_artifacts = _prompt_artifacts(
            sample=sample,
            prompt_defaults=prompt_defaults,
            instance_seed=int(instance_seed),
        )

        icons_by_position = {int(icon.position_index): icon for icon in scene_payload.icons}
        queried_icon = icons_by_position[int(scene_payload.query_position_index)]
        answer_icon = icons_by_position[int(scene_payload.answer_position_index)]
        if str(answer_icon.label) != str(scene_payload.answer_label):
            raise RuntimeError("rendered answer icon label does not match answer label")
        annotation_artifacts = icon_bbox_annotation(answer_icon.bbox_xyxy)
        serialized_icons = [serialize_path_icon(icon) for icon in scene_payload.icons]
        answer_gt = TypedValue(type="option_letter", value=str(scene_payload.answer_label))
        annotation_gt = TypedValue(
            type=str(annotation_artifacts["annotation_type"]),
            value=list(annotation_artifacts["annotation_value"]),
        )

        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(sample.query_id),
            params={
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id_probabilities": dict(sample.query_probabilities),
                "target_shape_id": str(scene_payload.target_shape_id),
                "target_shape_name": str(scene_payload.target_shape_name),
                "shape_probabilities": dict(sample.shape_probabilities),
                "answer_label": str(scene_payload.answer_label),
                "answer_label_probabilities": dict(sample.answer_label_probabilities),
                "candidate_count": int(len(OPTION_LABELS)),
                "distractor_count": int(scene_payload.distractor_count),
                "distractor_count_probabilities": dict(sample.distractor_count_probabilities),
                "target_occurrence_count": int(scene_payload.target_occurrence_count),
                "target_occurrence_count_probabilities": dict(sample.target_occurrence_count_probabilities),
                "stop_count": int(scene_payload.stop_count),
                "named_icon_fill_style_support": list(sample.fill_style_support),
                "fill_style_probabilities": dict(sample.fill_style_probabilities),
            },
        )
        trace_payload = {
            "scene_ir": {
                "scene_kind": "icons_named_path_neighbor",
                "scene_id": SCENE_ID,
                "query_id": str(sample.query_id),
                "entities": list(serialized_icons),
                "relations": {
                    "target": "labeled_neighbor_of_named_shape_occurrence_along_path",
                    "query_id": str(sample.query_id),
                    "target_shape_id": str(scene_payload.target_shape_id),
                    "target_shape_name": str(scene_payload.target_shape_name),
                    "target_positions": [int(value) for value in scene_payload.target_positions],
                    "query_position_index": int(scene_payload.query_position_index),
                    "answer_position_index": int(scene_payload.answer_position_index),
                    "neighbor_direction": str(scene_payload.neighbor_direction),
                    "answer_label": str(scene_payload.answer_label),
                    "candidate_labels": [str(label) for label in OPTION_LABELS],
                },
                "frames": {
                    "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                    "panels": dict(scene_payload.panel_geometry),
                    "path": {
                        "order": "start_to_end",
                        "points_xy": [[float(x), float(y)] for x, y in scene_payload.path_points_xy],
                    },
                },
            },
            "query_spec": query_spec,
            "render_spec": {
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id": str(sample.query_id),
                "canvas_size": [int(render_params["canvas_width"]), int(render_params["canvas_height"])],
                "coord_space": "pixel",
                "panel_geometry": dict(scene_payload.panel_geometry),
                "style": {
                    **icon_render_style_trace(
                        render_params=render_params,
                        sampled_palette_rgb=tuple(scene_payload.sampled_palette_rgb),
                    ),
                    **named_path_style_trace(render_params),
                },
            },
            "render_map": {
                "image_id": "img0",
                "object_bboxes_px": {
                    str(icon.instance_id): [int(value) for value in icon.bbox_xyxy]
                    for icon in scene_payload.icons
                },
                "path_points_xy": [[float(x), float(y)] for x, y in scene_payload.path_points_xy],
                "query_occurrence": serialize_path_icon(queried_icon),
                "answer_option": serialize_path_icon(answer_icon),
                "labels_by_position": {str(key): str(value) for key, value in scene_payload.labels_by_position.items()},
            },
            "execution_trace": {
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "scene_variant": "single_panel_named_path",
                "query_id": str(sample.query_id),
                "query_id_probabilities": dict(sample.query_probabilities),
                "question_format": "select_labeled_neighbor_of_named_icon_along_start_to_end_path",
                "target_shape_id": str(scene_payload.target_shape_id),
                "target_shape_name": str(scene_payload.target_shape_name),
                "target_occurrence_count": int(scene_payload.target_occurrence_count),
                "target_positions": [int(value) for value in scene_payload.target_positions],
                "query_position_index": int(scene_payload.query_position_index),
                "answer_position_index": int(scene_payload.answer_position_index),
                "neighbor_direction": str(scene_payload.neighbor_direction),
                "answer_label": str(scene_payload.answer_label),
                "candidate_labels": [str(label) for label in OPTION_LABELS],
                "option_positions": [int(value) for value in scene_payload.option_positions],
                "labels_by_position": {str(key): str(value) for key, value in scene_payload.labels_by_position.items()},
                "stop_count": int(scene_payload.stop_count),
                "distractor_count": int(scene_payload.distractor_count),
            },
            "witness_symbolic": {
                "query_id": str(sample.query_id),
                "target_shape_id": str(scene_payload.target_shape_id),
                "target_shape_name": str(scene_payload.target_shape_name),
                "query_instance_id": str(queried_icon.instance_id),
                "query_position_index": int(scene_payload.query_position_index),
                "answer_instance_id": str(answer_icon.instance_id),
                "answer_position_index": int(scene_payload.answer_position_index),
                "answer_label": str(scene_payload.answer_label),
                "annotation_target_instance_id": str(answer_icon.instance_id),
            },
            "projected_annotation": {
                **dict(annotation_artifacts["projected_annotation"]),
                "items": [
                    {
                        "role": "answer_option",
                        "instance_id": str(answer_icon.instance_id),
                        "bbox_xyxy": list(answer_icon.bbox_xyxy),
                    },
                ],
            },
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=scene_payload.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(sample.query_id),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
        )


__all__ = ["IconsNamedPathPathNeighborLabelTask"]
