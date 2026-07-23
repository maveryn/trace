"""Select the marked icon type with the highest or lowest frequency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.sampling import uniform_choice
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults, required_group_defaults
from ...shared.deterministic_sampling import uniform_probability_map
from ...shared.fixed_query import select_task_query_id
from ...shared.labeling import LABEL_POOL_A_L
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ..shared.annotation import icon_bbox_set_annotation
from ..shared.anchor_marking import draw_anchor_marker

from .shared.annotations import bboxes_for_icon_ids, indices_for_icon_ids
from .shared.defaults import IconFieldDefaults
from .shared.rendering import sample_and_render_icon_field_scene
from .shared.state import IconFieldScenePayload, TypeFrequencySpec
from .shared.styles import icon_field_style_trace, resolve_icon_field_render_params


TASK_ID = "task_icons__icon_field__frequency_extreme_type_label"
DOMAIN = "icons"
SCENE_ID = "icon_field"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("most_frequent_type_label", "least_frequent_type_label")
DEFAULT_QUERY_ID = "most_frequent_type_label"

_DEFAULTS = IconFieldDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


@dataclass(frozen=True)
class _FrequencyExtremeSpec:
    """Resolved task-owned frequency-label objective."""

    query_id: str
    query_id_probabilities: Dict[str, float]
    params: Dict[str, Any]
    option_count: int
    option_count_probabilities: Dict[str, float]
    frequency_spec: TypeFrequencySpec
    frequencies: Tuple[int, ...]
    extremum: str


def _resolve_option_count(instance_seed: int, params: Mapping[str, Any]) -> tuple[int, Dict[str, float]]:
    """Resolve the number of marked candidate icon types."""

    support = tuple(
        int(value)
        for value in params.get(
            "option_count_support",
            group_default(_GEN_DEFAULTS, "option_count_support", (4, 6)),
        )
    )
    if not support or any(value not in {4, 6} for value in support):
        raise ValueError("option_count_support must contain only 4 and/or 6")
    explicit = params.get("option_count")
    if explicit is not None:
        option_count = int(explicit)
        if int(option_count) not in set(support):
            raise ValueError("option_count is outside configured support")
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}:option_count")
        option_count = int(rng.choice(tuple(support)))
    return int(option_count), uniform_probability_map(
        tuple(int(value) for value in support),
        selected=int(option_count) if explicit is not None else None,
    )


def _frequency_values_for_extremum(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    option_count: int,
    extremum: str,
) -> tuple[Tuple[int, ...], Dict[str, float]]:
    """Sample type frequencies with a unique requested max/min."""

    object_count_min = int(
        params.get("object_count_min", group_default(_GEN_DEFAULTS, "object_count_min", 8))
    )
    object_count_max = int(
        params.get("object_count_max", group_default(_GEN_DEFAULTS, "object_count_max", 18))
    )
    frequency_min = int(
        params.get("frequency_min", group_default(_GEN_DEFAULTS, "frequency_min", 1))
    )
    frequency_max = int(
        params.get("frequency_max", group_default(_GEN_DEFAULTS, "frequency_max", 5))
    )
    explicit_object_count = params.get("object_count")
    if int(option_count) < 2:
        raise ValueError("option_count must be at least 2")
    if object_count_min < int(option_count) or object_count_max < object_count_min:
        raise ValueError("object_count bounds are invalid for frequency-extreme labels")
    if frequency_min < 1 or frequency_max <= frequency_min:
        raise ValueError("frequency bounds are invalid for frequency-extreme labels")

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:frequency_values:{str(extremum)}")
    for _ in range(500):
        if str(extremum) == "most":
            winning_frequency = int(rng.randint(max(2, int(frequency_min) + 1), int(frequency_max)))
            values = [int(rng.randint(int(frequency_min), int(winning_frequency) - 1)) for _ in range(int(option_count) - 1)]
            values.append(int(winning_frequency))
        elif str(extremum) == "least":
            losing_frequency = int(rng.randint(int(frequency_min), int(frequency_max) - 1))
            values = [int(rng.randint(int(losing_frequency) + 1, int(frequency_max))) for _ in range(int(option_count) - 1)]
            values.append(int(losing_frequency))
        else:
            raise ValueError(f"unsupported extremum: {extremum}")
        rng.shuffle(values)
        total = int(sum(int(value) for value in values))
        if explicit_object_count is not None and int(total) != int(explicit_object_count):
            continue
        if explicit_object_count is None and not (int(object_count_min) <= int(total) <= int(object_count_max)):
            continue
        if str(extremum) == "most":
            max_value = max(int(value) for value in values)
            if sum(1 for value in values if int(value) == int(max_value)) != 1:
                continue
        else:
            min_value = min(int(value) for value in values)
            if sum(1 for value in values if int(value) == int(min_value)) != 1:
                continue
        return tuple(int(value) for value in values), {
            str(value): 1.0 if int(value) == int(total) else 0.0
            for value in range(int(object_count_min), int(object_count_max) + 1)
        }
    raise ValueError("failed to sample a feasible unique frequency-extreme vector")


def _frequency_spec_from_values(
    frequencies: Sequence[int],
    *,
    object_count_probabilities: Mapping[str, float],
    option_count_probabilities: Mapping[str, float],
) -> TypeFrequencySpec:
    """Convert concrete type frequencies to the scene renderer spec."""

    singleton_count = sum(1 for value in frequencies if int(value) == 1)
    repeated = tuple(int(value) for value in frequencies if int(value) > 1)
    return TypeFrequencySpec(
        object_count=sum(int(value) for value in frequencies),
        singleton_count=int(singleton_count),
        repeated_type_multiplicities=tuple(repeated),
        object_count_probabilities=dict(object_count_probabilities),
        target_count_probabilities=dict(option_count_probabilities),
        distinct_color_count=1,
    )


def _resolve_objective(instance_seed: int, params: Mapping[str, Any]) -> _FrequencyExtremeSpec:
    """Resolve query, option count, and unique frequency vector."""

    query_id, query_probs, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id=DEFAULT_QUERY_ID,
        task_id=TASK_ID,
        namespace=f"{TASK_ID}:query",
    )
    extremum = "most" if str(query_id) == "most_frequent_type_label" else "least"
    option_count, option_count_probs = _resolve_option_count(int(instance_seed), task_params)
    frequencies, object_count_probs = _frequency_values_for_extremum(
        instance_seed=int(instance_seed),
        params=task_params,
        option_count=int(option_count),
        extremum=str(extremum),
    )
    frequency_spec = _frequency_spec_from_values(
        frequencies,
        object_count_probabilities=object_count_probs,
        option_count_probabilities=option_count_probs,
    )
    return _FrequencyExtremeSpec(
        query_id=str(query_id),
        query_id_probabilities=dict(query_probs),
        params=dict(task_params),
        option_count=int(option_count),
        option_count_probabilities=dict(option_count_probs),
        frequency_spec=frequency_spec,
        frequencies=tuple(int(value) for value in frequencies),
        extremum=str(extremum),
    )


def _representative_entities_by_icon_id(scene_payload: IconFieldScenePayload) -> tuple[Dict[str, Dict[str, Any]], Tuple[str, ...]]:
    """Return first reading-order entity per icon type and ordered icon ids."""

    sorted_entities = sorted(
        (dict(entity) for entity in scene_payload.scene_instances),
        key=lambda entity: (
            int(entity.get("bbox_xyxy", [0, 0, 0, 0])[1]),
            int(entity.get("bbox_xyxy", [0, 0, 0, 0])[0]),
            int(entity.get("index", 0)),
        ),
    )
    representatives: Dict[str, Dict[str, Any]] = {}
    for entity in sorted_entities:
        icon_id = str(entity["icon_id"])
        if icon_id not in representatives:
            representatives[icon_id] = dict(entity)
    ordered_ids = tuple(str(entity["icon_id"]) for entity in sorted_entities if str(entity["icon_id"]) in representatives)
    unique_ordered_ids: list[str] = []
    for icon_id in ordered_ids:
        if str(icon_id) not in unique_ordered_ids:
            unique_ordered_ids.append(str(icon_id))
    return representatives, tuple(unique_ordered_ids)


def _draw_candidate_markers(
    *,
    image,
    scene_payload: IconFieldScenePayload,
    candidate_icon_ids: Sequence[str],
    candidate_labels: Sequence[str],
    render_params: Mapping[str, Any],
) -> tuple[Tuple[Dict[str, Any], ...], Dict[str, str]]:
    """Draw label markers on representative instances for candidate types."""

    representatives, _ = _representative_entities_by_icon_id(scene_payload)
    label_by_icon_id: Dict[str, str] = {}
    marker_records: list[Dict[str, Any]] = []
    content_bbox = tuple(int(value) for value in scene_payload.panel_geometry["scene_content_xyxy"])
    for icon_id, label in zip(candidate_icon_ids, candidate_labels):
        entity = representatives[str(icon_id)]
        marker_bbox = draw_anchor_marker(
            image=image,
            anchor_bbox=entity["bbox_xyxy"],
            content_bbox=content_bbox,
            highlight_padding_px=5,
            highlight_radius_px=10,
            outline_rgb=(220, 38, 38),
            label_color_rgb=(220, 38, 38),
            panel_fill_rgb=tuple(int(value) for value in render_params["panel_fill_rgb"]),
            label_font_size_px=24,
            label_text=str(label),
            label_role="icon_field_candidate_option_label",
        )
        label_by_icon_id[str(icon_id)] = str(label)
        marker_records.append(
            {
                "icon_id": str(icon_id),
                "label": str(label),
                "representative_index": int(entity["index"]),
                "representative_bbox_xyxy": [int(value) for value in entity["bbox_xyxy"]],
                "marker_bbox_xyxy": [int(value) for value in marker_bbox],
                "type_frequency": int(scene_payload.type_frequencies[str(icon_id)]),
            }
        )
    return tuple(dict(record) for record in marker_records), dict(label_by_icon_id)


def _unique_extreme_icon_id(scene_payload: IconFieldScenePayload, extremum: str) -> tuple[str, int]:
    """Return the unique icon type id and frequency for the requested extreme."""

    frequencies = {str(key): int(value) for key, value in scene_payload.type_frequencies.items()}
    if str(extremum) == "most":
        target_frequency = max(int(value) for value in frequencies.values())
    elif str(extremum) == "least":
        target_frequency = min(int(value) for value in frequencies.values())
    else:
        raise ValueError(f"unsupported extremum: {extremum}")
    winning_icon_ids = [
        str(icon_id)
        for icon_id, frequency in frequencies.items()
        if int(frequency) == int(target_frequency)
    ]
    if len(winning_icon_ids) != 1:
        raise ValueError("frequency-extreme task requires a unique winning type")
    return str(winning_icon_ids[0]), int(target_frequency)


def _balanced_candidate_labels(
    *,
    instance_seed: int,
    objective: _FrequencyExtremeSpec,
    scene_payload: IconFieldScenePayload,
    candidate_icon_ids: Sequence[str],
) -> tuple[Tuple[str, ...], Dict[str, float]]:
    """Assign a uniformly sampled option label to the winning icon type."""

    labels = tuple(str(label) for label in LABEL_POOL_A_L[: int(objective.option_count)])
    explicit_label = objective.params.get("answer_label")
    if explicit_label is not None:
        answer_label = str(explicit_label)
        if answer_label not in labels:
            raise ValueError("answer_label is outside the visible option labels")
    else:
        label_rng = spawn_rng(
            int(instance_seed),
            f"{TASK_ID}:answer_label:{objective.query_id}:{objective.option_count}",
        )
        answer_label = str(uniform_choice(label_rng, labels))
    label_probabilities = {
        str(label): (1.0 if str(label) == str(answer_label) else 0.0)
        if explicit_label is not None
        else 1.0 / float(len(labels))
        for label in labels
    }

    winner_icon_id, _ = _unique_extreme_icon_id(scene_payload, str(objective.extremum))
    if str(winner_icon_id) not in {str(icon_id) for icon_id in candidate_icon_ids}:
        raise ValueError("winning icon type is missing from candidate markers")

    shuffle_rng = spawn_rng(
        int(instance_seed),
        f"{TASK_ID}:distractor_labels:{objective.query_id}:{objective.option_count}",
    )
    remaining_labels = [str(label) for label in labels if str(label) != str(answer_label)]
    shuffle_rng.shuffle(remaining_labels)
    labels_by_icon_id: Dict[str, str] = {str(winner_icon_id): str(answer_label)}
    for icon_id in candidate_icon_ids:
        if str(icon_id) == str(winner_icon_id):
            continue
        labels_by_icon_id[str(icon_id)] = str(remaining_labels.pop())
    return tuple(str(labels_by_icon_id[str(icon_id)]) for icon_id in candidate_icon_ids), dict(label_probabilities)


def _build_output(
    *,
    instance_seed: int,
    objective: _FrequencyExtremeSpec,
    scene_payload: IconFieldScenePayload,
    image,
    render_params: Mapping[str, Any],
    pool_manifest: str,
    marker_records: Sequence[Mapping[str, Any]],
    label_by_icon_id: Mapping[str, str],
    answer_label_probabilities: Mapping[str, float],
) -> TaskOutput:
    """Build prompt, typed answer, annotation, and trace payload."""

    frequencies = {str(key): int(value) for key, value in scene_payload.type_frequencies.items()}
    if str(objective.extremum) == "most":
        question_format = "select_most_frequent_marked_icon_type"
    else:
        question_format = "select_least_frequent_marked_icon_type"
    winner_icon_id, target_frequency = _unique_extreme_icon_id(scene_payload, str(objective.extremum))
    answer_label = str(label_by_icon_id[str(winner_icon_id)])
    annotation_bboxes = bboxes_for_icon_ids(scene_payload, (str(winner_icon_id),))
    annotation_indices = indices_for_icon_ids(scene_payload, (str(winner_icon_id),))
    if len(annotation_bboxes) != int(target_frequency):
        raise ValueError("annotation bbox count does not match winning frequency")
    annotation_payload = icon_bbox_set_annotation(
        annotation_bboxes,
        clip_bbox=scene_payload.panel_geometry["scene_content_xyxy"],
    )

    resolved_prompt_defaults = required_group_defaults(
        _PROMPT_DEFAULTS,
        ("bundle_id", "scene_key", "task_key"),
        context=f"prompt defaults for {TASK_ID}",
    )
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(resolved_prompt_defaults["bundle_id"]),
        scene_key=str(resolved_prompt_defaults["scene_key"]),
        task_key=str(resolved_prompt_defaults["task_key"]),
        query_key=str(objective.query_id),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={},
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

    scene_entities = [dict(entity) for entity in scene_payload.scene_instances]
    for record in marker_records:
        representative_index = int(record["representative_index"])
        scene_entities[representative_index]["candidate_label"] = str(record["label"])
        scene_entities[representative_index]["candidate_marker_bbox_xyxy"] = [
            int(value) for value in record["marker_bbox_xyxy"]
        ]

    answer_gt = TypedValue(type="option_letter", value=str(answer_label))
    annotation_gt = TypedValue(
        type=str(annotation_payload["annotation_type"]),
        value=list(annotation_payload["annotation_value"]),
    )
    trace_payload = {
        "scene_ir": {
            "scene_kind": "icons_frequency_extreme_type_label",
            "scene_id": SCENE_ID,
            "query_id": str(objective.query_id),
            "entities": scene_entities,
            "relations": {
                "counting_rule": str(question_format),
                "extremum": str(objective.extremum),
                "candidate_type_labels": [
                    {"icon_id": str(record["icon_id"]), "label": str(record["label"])}
                    for record in marker_records
                ],
                "type_frequencies": dict(frequencies),
                "winner_icon_id": str(winner_icon_id),
                "winner_label": str(answer_label),
                "winner_frequency": int(target_frequency),
                "marked_candidate_count": int(objective.option_count),
            },
            "frames": {
                "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                "panels": dict(scene_payload.panel_geometry),
            },
        },
        "query_spec": {
            "task_id": TASK_ID,
            "scene_id": SCENE_ID,
            "query_id": str(objective.query_id),
            "template_id": str(resolved_prompt_defaults["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": {
                "task_id": TASK_ID,
                "scene_id": SCENE_ID,
                "query_id": str(objective.query_id),
                "prompt_query_key": str(objective.query_id),
                "query_id_probabilities": dict(objective.query_id_probabilities),
                "option_count": int(objective.option_count),
                "option_count_probabilities": dict(objective.option_count_probabilities),
                "answer_label_probabilities": dict(answer_label_probabilities),
                "object_count": int(scene_payload.object_count),
                "unique_type_total": int(scene_payload.distinct_type_count),
                "unique_color_total": int(scene_payload.distinct_color_count),
                "winner_icon_id": str(winner_icon_id),
                "winner_label": str(answer_label),
                "winner_frequency": int(target_frequency),
                "pool_manifest": str(pool_manifest),
            },
        },
        "render_spec": {
            "task_id": TASK_ID,
            "scene_id": SCENE_ID,
            "query_id": str(objective.query_id),
            "canvas_size": list(scene_payload.panel_geometry["canvas_size"]),
            "coord_space": "pixel",
            "panel_geometry": dict(scene_payload.panel_geometry),
            "style": icon_field_style_trace(
                render_params=render_params,
                sampled_palette_rgb=scene_payload.sampled_palette_rgb,
            ),
            "candidate_markers": [dict(record) for record in marker_records],
        },
        "render_map": {"image_id": "img0", "anchors": {}},
        "execution_trace": {
            "task_id": TASK_ID,
            "scene_id": SCENE_ID,
            "scene_variant": "single_panel_scene_with_marked_candidate_types",
            "query_id": str(objective.query_id),
            "prompt_query_key": str(objective.query_id),
            "question_format": str(question_format),
            "extremum": str(objective.extremum),
            "object_count": int(scene_payload.object_count),
            "unique_type_total": int(scene_payload.distinct_type_count),
            "unique_color_total": int(scene_payload.distinct_color_count),
            "option_count": int(objective.option_count),
            "scene_icon_ids": list(scene_payload.scene_icon_ids),
            "scene_color_keys": list(scene_payload.scene_color_keys),
            "type_frequencies": dict(frequencies),
            "candidate_markers": [dict(record) for record in marker_records],
            "winner_icon_id": str(winner_icon_id),
            "winner_label": str(answer_label),
            "answer_label_probabilities": dict(answer_label_probabilities),
            "winner_frequency": int(target_frequency),
            "annotation_indices": [int(index) for index in annotation_indices],
            "annotation_bboxes": [list(bbox) for bbox in annotation_gt.value],
        },
        "witness_symbolic": {
            "winner_icon_id": str(winner_icon_id),
            "winner_label": str(answer_label),
            "winner_frequency": int(target_frequency),
            "counted_icon_ids": [str(winner_icon_id)],
            "annotation_indices": [int(index) for index in annotation_indices],
            "annotation_bboxes": [list(bbox) for bbox in annotation_gt.value],
        },
        "projected_annotation": dict(annotation_payload["projected_annotation"]),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(objective.query_id),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


@register_task
class IconsIconFieldFrequencyExtremeTypeLabelTask:
    """Select the marked type with the unique maximum or minimum frequency."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'ranking')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one marked-type frequency-extreme option-label instance.

        The task file owns query selection, candidate-label binding, answer
        binding, and all-icons-of-selected-type annotation. Scene shared code
        only constructs and renders an identity-free frequency distribution.
        """

        objective = _resolve_objective(int(instance_seed), dict(params))
        render_params = resolve_icon_field_render_params(
            objective.params,
            render_defaults=_RENDER_DEFAULTS,
            defaults=_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        pool_manifest = str(
            objective.params.get(
                "pool_manifest",
                group_default(_GEN_DEFAULTS, "pool_manifest", _DEFAULTS.pool_manifest),
            )
        )
        scene_rng = spawn_rng(int(instance_seed), "scene")
        last_error: Exception | None = None
        for _ in range(max(1, int(max_attempts))):
            try:
                scene_payload, image = sample_and_render_icon_field_scene(
                    scene_rng,
                    instance_seed=int(instance_seed),
                    frequency_spec=objective.frequency_spec,
                    pool_manifest=str(pool_manifest),
                    render_params=render_params,
                    noise_namespace=f"icon_field_frequency_extreme:{TASK_ID}",
                )
                representatives, candidate_icon_ids = _representative_entities_by_icon_id(scene_payload)
                del representatives
                if len(candidate_icon_ids) != int(objective.option_count):
                    raise ValueError("candidate representative count does not match option count")
                candidate_labels, answer_label_probabilities = _balanced_candidate_labels(
                    instance_seed=int(instance_seed),
                    objective=objective,
                    scene_payload=scene_payload,
                    candidate_icon_ids=candidate_icon_ids,
                )
                marker_records, label_by_icon_id = _draw_candidate_markers(
                    image=image,
                    scene_payload=scene_payload,
                    candidate_icon_ids=candidate_icon_ids,
                    candidate_labels=candidate_labels,
                    render_params=render_params,
                )
                return _build_output(
                    instance_seed=int(instance_seed),
                    objective=objective,
                    scene_payload=scene_payload,
                    image=image,
                    render_params=render_params,
                    pool_manifest=str(pool_manifest),
                    marker_records=marker_records,
                    label_by_icon_id=label_by_icon_id,
                    answer_label_probabilities=answer_label_probabilities,
                )
            except Exception as exc:  # pragma: no cover - retry loop handles rare placement failures
                last_error = exc
                continue
        raise RuntimeError(f"failed to generate {TASK_ID} instance") from last_error


__all__ = ["IconsIconFieldFrequencyExtremeTypeLabelTask"]
