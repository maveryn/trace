"""Select the correctly arranged jigsaw reconstruction of a park scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults, required_group_defaults
from ...shared.annotation_artifacts import bbox_annotation_artifacts
from ...shared.output_metadata import default_task_versions
from ..shared.cutouts import (
    DEFAULT_OPTION_LABELS,
    JIGSAW_BOARD_STYLES,
    compose_jigsaw_arrangement_options,
    downscale_jigsaw_arrangement_artifacts,
    piece_crops,
    sample_style,
    style_trace,
)
from ..shared.canvas_profiles import (
    MAX_RECONSTRUCTION_OUTPUT_PIXELS,
    reconstruction_grid_for_size,
    resolve_reconstruction_source_profile,
)
from ..shared.option_rendering import image_detail_score, sample_visual_label_font_trace
from .shared.annotations import park_scene_entities, serialize_park_scene
from .shared.defaults import CountDefaults
from .shared.prompts import build_park_prompt_artifacts
from .shared.rendering import (
    PARK_EQUIPMENT_TYPES,
    PARK_PERSON_ACTIVITIES,
    ParkEquipmentSpec,
    ParkPersonSpec,
    render_park_playground_scene,
)
from .shared.sampling import activity_support, bounds, equipment_support, render_params, sample_count, sample_option_answer_index, setting_weights, spawned_task_rng, style_weights


TASK_ID = "task_illustrations__park_playground__jigsaw_arrangement_label"
SCENE_ID = "park_playground"
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "jigsaw_arrangement_label"
GRID_ROWS = 2
GRID_COLS = 2
OPTION_LABELS: Tuple[str, ...] = DEFAULT_OPTION_LABELS[:4]


@dataclass(frozen=True)
class _Defaults:
    person_count_min: int = 8
    person_count_max: int = 13
    equipment_count_min: int = 4
    equipment_count_max: int = 7
    source_width: int = 520
    source_height: int = 366
    canvas_width: int = 1280
    canvas_height: int = 900
    render_scale: int = 2
    min_tile_detail_score: float = 600.0


@dataclass(frozen=True)
class _SampleSpec:
    person_count: int
    equipment_count: int
    person_specs: Tuple[ParkPersonSpec, ...]
    equipment_specs: Tuple[ParkEquipmentSpec, ...]
    source_size: Tuple[int, int]
    source_profile_trace: Dict[str, Any]
    correct_index: int
    person_count_probabilities: Dict[str, float]
    equipment_count_probabilities: Dict[str, float]
    correct_index_probabilities: Dict[str, float]


_DEFAULTS = _Defaults()
_COUNT_DEFAULTS = CountDefaults(
    person_count_min=_DEFAULTS.person_count_min,
    person_count_max=_DEFAULTS.person_count_max,
    equipment_count_min=_DEFAULTS.equipment_count_min,
    equipment_count_max=_DEFAULTS.equipment_count_max,
    canvas_width=_DEFAULTS.canvas_width,
    canvas_height=_DEFAULTS.canvas_height,
    render_scale=_DEFAULTS.render_scale,
)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "illustrations",
    SCENE_ID,
    task_id=TASK_ID,
)


def _int_value(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(str(key), group_default(defaults, str(key), int(fallback))))


def _float_value(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: float) -> float:
    return float(params.get(str(key), group_default(defaults, str(key), float(fallback))))


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any], attempt_index: int) -> _SampleSpec:
    """Sample a dense source park scene and one correct MCQ option position."""

    rng = spawned_task_rng(int(instance_seed), TASK_ID, int(attempt_index))
    person_min, person_max = bounds(
        params,
        _GEN_DEFAULTS,
        "person_count_min",
        "person_count_max",
        _DEFAULTS.person_count_min,
        _DEFAULTS.person_count_max,
    )
    person_count, person_count_probabilities = sample_count(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:person_count",
        low=int(person_min),
        high=int(person_max),
        explicit_key="person_count",
    )
    equipment_min, equipment_max = bounds(
        params,
        _GEN_DEFAULTS,
        "equipment_count_min",
        "equipment_count_max",
        _DEFAULTS.equipment_count_min,
        _DEFAULTS.equipment_count_max,
    )
    equipment_count, equipment_count_probabilities = sample_count(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:equipment_count",
        low=int(equipment_min),
        high=int(equipment_max),
        explicit_key="equipment_count",
    )
    activities = activity_support(params, _GEN_DEFAULTS, fallback=PARK_PERSON_ACTIVITIES)
    equipment_values = equipment_support(params, _GEN_DEFAULTS, fallback=PARK_EQUIPMENT_TYPES)
    person_specs = tuple(
        ParkPersonSpec(activity=str(rng.choice(activities)), role="source")
        for _ in range(int(person_count))
    )
    equipment_specs = tuple(
        ParkEquipmentSpec(equipment_type=str(rng.choice(equipment_values)), role="source")
        for _ in range(int(equipment_count))
    )
    source_profile = resolve_reconstruction_source_profile(
        params=params,
        defaults=_GEN_DEFAULTS,
        fallback_source_width=_DEFAULTS.source_width,
        fallback_source_height=_DEFAULTS.source_height,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:source_profile",
    )
    grid_rows, grid_cols = reconstruction_grid_for_size(int(source_profile.width), int(source_profile.height))
    correct_index, correct_index_probabilities = sample_option_answer_index(
        params=params,
        instance_seed=int(instance_seed),
        seed_scope=TASK_ID,
        option_labels=OPTION_LABELS,
    )
    return _SampleSpec(
        person_count=int(person_count),
        equipment_count=int(equipment_count),
        person_specs=person_specs,
        equipment_specs=equipment_specs,
        source_size=tuple(int(value) for value in source_profile.size),
        source_profile_trace=dict(source_profile.trace()),
        correct_index=int(correct_index),
        person_count_probabilities=dict(person_count_probabilities),
        equipment_count_probabilities=dict(equipment_count_probabilities),
        correct_index_probabilities=dict(correct_index_probabilities),
    )


def _tile_detail_scores(source_image: Image.Image, *, rows: int, cols: int) -> Tuple[float, ...]:
    pieces = piece_crops(source_image.convert("RGB"), rows=int(rows), cols=int(cols))
    return tuple(float(image_detail_score(piece)) for piece, _box in pieces)


@register_task
class IllustrationsParkPlaygroundJigsawArrangementLabelTask:
    """Select the option that correctly arranges profile-aware park-scene tiles."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    domain = "illustrations"
    supported_query_ids = (QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Render a source park scene, build jigsaw MCQ options, and bind the selected option witness."""

        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        scene = None
        source_panel = None
        artifacts = None
        board_style = None
        label_font_trace: Dict[str, Any] | None = None
        tile_detail_scores: Tuple[float, ...] = tuple()
        min_tile_detail_score = _float_value(
            params,
            _GEN_DEFAULTS,
            "min_tile_detail_score",
            _DEFAULTS.min_tile_detail_score,
        )

        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params, attempt_index=int(attempt))
                scene_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:scene", int(attempt))
                source_width, source_height = int(sample.source_size[0]), int(sample.source_size[1])
                grid_rows, grid_cols = reconstruction_grid_for_size(source_width, source_height)
                option_labels = OPTION_LABELS
                rp = render_params(
                    {
                        **dict(params),
                        "canvas_width": source_width,
                        "canvas_height": source_height,
                    },
                    _RENDER_DEFAULTS,
                    fallback_width=_COUNT_DEFAULTS.canvas_width,
                    fallback_height=_COUNT_DEFAULTS.canvas_height,
                    fallback_scale=_COUNT_DEFAULTS.render_scale,
                    instance_seed=int(instance_seed),
                    namespace=f"{TASK_ID}:source_profile",
                )
                scene = render_park_playground_scene(
                    rng=scene_rng,
                    person_specs=sample.person_specs,
                    equipment_specs=sample.equipment_specs,
                    canvas_width=int(rp["canvas_width"]),
                    canvas_height=int(rp["canvas_height"]),
                    render_scale=int(rp["render_scale"]),
                    setting_weights=setting_weights(params, _RENDER_DEFAULTS),
                    style_weights=style_weights(params, _RENDER_DEFAULTS),
                )
                source_panel = scene.image.convert("RGB")
                tile_detail_scores = _tile_detail_scores(source_panel, rows=grid_rows, cols=grid_cols)
                if min(tile_detail_scores) < float(min_tile_detail_score):
                    raise ValueError("source scene has a weak jigsaw tile")
                option_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:jigsaw_options", int(attempt))
                board_style = sample_style(option_rng, JIGSAW_BOARD_STYLES)
                label_font_trace = sample_visual_label_font_trace(
                    namespace_prefix=TASK_ID,
                    instance_seed=int(instance_seed),
                    params={**dict(_RENDER_DEFAULTS), **dict(params)},
                    namespace_suffix="jigsaw_option_labels",
                    explicit_key="jigsaw_label_font_family",
                    weights_key="jigsaw_label_font_weights",
                )
                artifacts = compose_jigsaw_arrangement_options(
                    source_image=source_panel,
                    rows=grid_rows,
                    cols=grid_cols,
                    correct_index=int(sample.correct_index),
                    rng=option_rng,
                    board_style=board_style,
                    label_font_family=str(label_font_trace["font_family"]),
                    labels=option_labels,
                )
                artifacts = downscale_jigsaw_arrangement_artifacts(
                    artifacts,
                    max_pixels=MAX_RECONSTRUCTION_OUTPUT_PIXELS,
                )
                break
            except Exception as exc:  # pragma: no cover - retry surface is seed/layout dependent.
                last_error = exc
                sample = None
                scene = None
                source_panel = None
                artifacts = None
                board_style = None
                label_font_trace = None
                tile_detail_scores = tuple()

        if (
            sample is None
            or scene is None
            or source_panel is None
            or artifacts is None
            or board_style is None
            or label_font_trace is None
        ):
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        serialized_scene, person_bboxes = serialize_park_scene(scene)
        answer_label = str(artifacts.selected_label)
        grid_rows, grid_cols = int(artifacts.grid_shape[0]), int(artifacts.grid_shape[1])
        option_labels = OPTION_LABELS
        annotation_artifacts = bbox_annotation_artifacts(artifacts.selected_option_bbox)
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            [
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "answer_hint_jigsaw_arrangement",
                "annotation_hint_jigsaw_arrangement",
                "json_example_jigsaw_arrangement",
                "json_example_answer_only_jigsaw_arrangement",
            ],
            context=f"prompt defaults for {TASK_ID}",
        )
        slots = {
            "person_count": int(sample.person_count),
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "answer_hint": str(prompt_defaults["answer_hint_jigsaw_arrangement"]),
            "annotation_hint": str(prompt_defaults["annotation_hint_jigsaw_arrangement"]),
            "json_example": str(prompt_defaults["json_example_jigsaw_arrangement"]),
            "json_example_answer_only": str(prompt_defaults["json_example_answer_only_jigsaw_arrangement"]),
        }
        prompt_artifacts = build_park_prompt_artifacts(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=prompt_defaults,
            prompt_query_key=PROMPT_QUERY_KEY,
            slots=slots,
            instance_seed=int(instance_seed),
        )
        option_permutations_by_label = {
            str(label): [int(value) for value in artifacts.option_permutations[index]]
            for index, label in enumerate(option_labels)
        }
        trace_payload = {
            "scene_ir": {
                "domain": self.domain,
                "scene_id": SCENE_ID,
                "entities": park_scene_entities(scene),
                "relations": {
                    "query_id": QUERY_ID,
                    "prompt_query_key": PROMPT_QUERY_KEY,
                    "answer_label": answer_label,
                    "grid_shape": [grid_rows, grid_cols],
                },
            },
            "query_spec": {
                "task_id": self.task_id,
                "scene_id": SCENE_ID,
                "query_id": QUERY_ID,
                "prompt_query_key": PROMPT_QUERY_KEY,
                "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "query_id": QUERY_ID,
                    "prompt_query_key": PROMPT_QUERY_KEY,
                    "person_count": int(sample.person_count),
                    "equipment_count": int(sample.equipment_count),
                    "person_count_probabilities": dict(sample.person_count_probabilities),
                    "equipment_count_probabilities": dict(sample.equipment_count_probabilities),
                    "correct_index": int(sample.correct_index),
                    "correct_index_probabilities": dict(sample.correct_index_probabilities),
                    "answer_label": answer_label,
                    "option_labels": list(option_labels),
                    "grid_shape": [grid_rows, grid_cols],
                    "source_size": [int(sample.source_size[0]), int(sample.source_size[1])],
                    **dict(sample.source_profile_trace),
                    "min_tile_detail_score": float(min_tile_detail_score),
                    "tile_detail_scores": [round(float(value), 3) for value in tile_detail_scores],
                },
            },
            "render_spec": {
                "canvas_size": [int(artifacts.image.width), int(artifacts.image.height)],
                "coord_space": "pixel",
                "scene_id": SCENE_ID,
                "source_scene_canvas_size": [int(scene.canvas_width), int(scene.canvas_height)],
                "source_profile": dict(sample.source_profile_trace),
                "style": {
                    "source_setting_id": str(scene.setting_id),
                    "source_style_id": str(scene.style_id),
                    "source_render_scale": int(scene.render_scale),
                    "source_layout": dict(scene.layout),
                    "jigsaw_style": style_trace(board_style),
                    "jigsaw_label_font": dict(label_font_trace),
                },
            },
            "render_map": {
                "image_id": "img0",
                "option_bboxes_px_by_label": {str(key): list(value) for key, value in artifacts.option_bboxes.items()},
                "selected_option_bbox_px": list(artifacts.selected_option_bbox),
                "source_person_bboxes_px": person_bboxes,
                "source_scene_canvas_size": [int(scene.canvas_width), int(scene.canvas_height)],
                "source_size": [int(sample.source_size[0]), int(sample.source_size[1])],
                "tile_source_boxes_px": [[int(coord) for coord in box] for box in artifacts.tile_source_boxes],
                "option_permutations_by_label": option_permutations_by_label,
                "correct_permutation": [int(value) for value in artifacts.correct_permutation],
                "grid_shape": [grid_rows, grid_cols],
                "option_layout_shape": [int(artifacts.option_layout_shape[0]), int(artifacts.option_layout_shape[1])],
                "pre_downscale_canvas_size": [int(value) for value in artifacts.pre_downscale_canvas_size],
                "output_scale_xy": [float(value) for value in artifacts.output_scale_xy],
            },
            "execution_trace": {
                "query_id": QUERY_ID,
                "prompt_query_key": PROMPT_QUERY_KEY,
                "scene_id": SCENE_ID,
                "answer": answer_label,
                "answer_label": answer_label,
                "correct_index": int(sample.correct_index),
                "option_labels": list(option_labels),
                "option_permutations_by_label": option_permutations_by_label,
                "correct_permutation": [int(value) for value in artifacts.correct_permutation],
                "grid_shape": [grid_rows, grid_cols],
                "person_count": int(sample.person_count),
                "equipment_count": int(sample.equipment_count),
                "source_scene": serialized_scene[0],
            },
            "witness_symbolic": {
                "selected_option": list(artifacts.selected_option_bbox),
                "answer_label": answer_label,
                "correct_index": int(sample.correct_index),
            },
            "projected_annotation": {
                **dict(annotation_artifacts.projected_annotation),
            },
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
            answer_gt=TypedValue(type="option_letter", value=answer_label),
            annotation_gt=annotation_artifacts.annotation_gt,
            image=artifacts.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=QUERY_ID,
        )


__all__ = ["IllustrationsParkPlaygroundJigsawArrangementLabelTask", "TASK_ID", "_sample_spec"]
