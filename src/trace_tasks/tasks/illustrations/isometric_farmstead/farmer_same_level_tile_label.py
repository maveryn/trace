"""Select the lettered terrain tile at the farmer's elevation level."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from .shared.output import (
    isometric_farmstead_elevation_render_map,
    make_tile_label_trace_payload,
    rounded_bbox,
)
from .shared.prompts import build_isometric_farmstead_task_prompt_with_default_slots
from .shared.rendering import (
    DEFAULT_CANDIDATE_LABELS,
    SCENE_ID,
    render_isometric_farmstead_labeled_scene_with_retry,
)
from .shared.sampling import sample_label_task_spec
from .shared.state import IsoFarmsteadScene, IsoFarmsteadTile
from .shared.spatial_primitives import (
    eligible_label_tiles_by_level,
    farmer_reference_bbox,
    label_clear_of_candidate_tiles,
)


TASK_ID = "task_illustrations__isometric_farmstead__farmer_same_level_tile_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("farmer_same_level_tile",)


_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _select_farmer_and_candidate_tiles(
    *,
    scene: IsoFarmsteadScene,
    candidate_count: int,
    instance_seed: int,
) -> tuple[str, dict[str, str], str]:
    """Select a safe farmer tile plus one same-level candidate and distractors."""

    base_by_level = eligible_label_tiles_by_level(scene, exclude_unsafe_low_adjacent_higher=True)
    active_levels = tuple(level for level in sorted(base_by_level) if base_by_level[level])
    if len(active_levels) < 2:
        raise ValueError("not enough active levels for farmer same-level task")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:farmer_same_level_candidates")
    level_order = list(active_levels)
    rng.shuffle(level_order)
    labels = list(DEFAULT_CANDIDATE_LABELS[: int(candidate_count)])

    for farmer_level in level_order:
        farmer_pool = list(base_by_level[int(farmer_level)])
        rng.shuffle(farmer_pool)
        for farmer_tile in farmer_pool:
            farmer_box = farmer_reference_bbox(farmer_tile)
            by_level = eligible_label_tiles_by_level(
                scene,
                blocking_boxes=(farmer_box,),
                exclude_tile_ids=(str(farmer_tile.tile_id),),
                exclude_unsafe_low_adjacent_higher=True,
            )
            answer_pool = list(by_level.get(int(farmer_level), []))
            distractor_pool = [tile for level in active_levels if int(level) != int(farmer_level) for tile in by_level.get(int(level), [])]
            if not answer_pool or len(distractor_pool) < int(candidate_count) - 1:
                continue
            rng.shuffle(answer_pool)
            rng.shuffle(distractor_pool)
            for answer_tile in answer_pool:
                candidate_tiles = [answer_tile]
                for tile in distractor_pool:
                    if not label_clear_of_candidate_tiles(candidate_tiles, tile):
                        continue
                    candidate_tiles.append(tile)
                    if len(candidate_tiles) >= int(candidate_count):
                        break
                if len(candidate_tiles) < int(candidate_count):
                    continue
                selected_label = str(labels[int(instance_seed) % int(candidate_count)])
                distractor_labels = [str(label) for label in labels if str(label) != selected_label]
                candidate_tile_ids_by_label = {selected_label: str(answer_tile.tile_id)}
                for label, tile in zip(distractor_labels, candidate_tiles[1:]):
                    candidate_tile_ids_by_label[str(label)] = str(tile.tile_id)
                return (
                    str(farmer_tile.tile_id),
                    {str(label): str(candidate_tile_ids_by_label[str(label)]) for label in labels},
                    selected_label,
                )
    raise ValueError("could not select farmer and label-separated same-level candidate tiles")


@register_task
class IllustrationsIsometricFarmsteadFarmerSameLevelTileLabelTask:
    """Choose the lettered terrain tile matching the farmer's elevation."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations',)
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one farmer-reference tile-label instance from one scene trace."""

        sample = sample_label_task_spec(
            instance_seed=int(instance_seed),
            params=params,
            identity_label=TASK_ID,
            query_options=SUPPORTED_QUERY_IDS,
            default_query_id="farmer_same_level_tile",
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            max_candidate_count=len(DEFAULT_CANDIDATE_LABELS),
        )
        _prompt_defaults, prompt_artifacts = build_isometric_farmstead_task_prompt_with_default_slots(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults_source=_PROMPT_DEFAULTS,
            prompt_query_key=sample.prompt_key,
            answer_hint_key="answer_hint_farmer_same_level_tile_label",
            annotation_hint_key="annotation_hint_farmer_same_level_tile_label",
            json_example_key="json_example_farmer_same_level_tile_label",
            json_example_answer_only_key="json_example_answer_only_farmer_same_level_tile_label",
            context_label=TASK_ID,
            instance_seed=int(instance_seed),
        )
        def _select_from_scene(base_scene: IsoFarmsteadScene, scene_seed: int) -> Mapping[str, Any]:
            farmer_tile, candidates, answer_label = _select_farmer_and_candidate_tiles(
                scene=base_scene,
                candidate_count=sample.candidate_count,
                instance_seed=scene_seed,
            )
            return {
                "candidate_tile_ids_by_label": candidates,
                "selected_label": answer_label,
                "render_kwargs": {"reference_farmer_tile_id": str(farmer_tile)},
            }

        rendered = render_isometric_farmstead_labeled_scene_with_retry(
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            sample=sample,
            params=params,
            rendering_defaults=_RENDER_DEFAULTS,
            font_seed_label=TASK_ID,
            label_namespace_suffix="farmer_same_level_tile_labels",
            selection_fn=_select_from_scene,
        )
        scene = rendered["scene"]
        candidates_by_label = dict(rendered["candidate_tile_ids_by_label"])
        selected_label = str(rendered["selected_label"])
        label_font_trace = dict(rendered["label_font_trace"])

        tiles_by_id = {str(tile.tile_id): tile for tile in scene.tiles}
        entities_by_id = {str(entity.entity_id): entity for entity in scene.entities}
        farmer_entity = entities_by_id.get("farmer_00")
        if farmer_entity is None:
            raise RuntimeError("rendered farmer reference entity is missing")
        farmer_tile_id = str(farmer_entity.tile_ids[0])
        farmer_tile = tiles_by_id[farmer_tile_id]
        selected_tile_id = str(candidates_by_label[str(selected_label)])
        selected_tile = tiles_by_id[selected_tile_id]
        if int(selected_tile.level) != int(farmer_tile.level):
            raise RuntimeError("selected tile is not at the farmer's elevation level")

        annotation_value = rounded_bbox(selected_tile.bbox_xyxy)
        render_map = isometric_farmstead_elevation_render_map(
            scene=scene,
            candidate_tile_ids_by_label=candidates_by_label,
            selected_label=str(selected_label),
        )
        render_map = {
            **render_map,
            "reference_farmer_entity_id": "farmer_00",
            "reference_farmer_tile_id": str(farmer_tile_id),
            "reference_farmer_level": int(farmer_tile.level),
            "reference_farmer_bbox_px": rounded_bbox(farmer_entity.bbox_xyxy),
        }
        farmer_trace_fields = {
            "reference_farmer_entity_id": "farmer_00",
            "reference_farmer_tile_id": str(farmer_tile_id),
            "reference_farmer_level": int(farmer_tile.level),
        }
        trace_payload = make_tile_label_trace_payload(
            domain=self.domain,
            scene_id=SCENE_ID,
            task_identity=TASK_ID,
            scene=scene,
            sample=sample,
            prompt_artifacts=prompt_artifacts,
            label_font_trace=label_font_trace,
            render_map=render_map,
            candidate_tile_ids_by_label=candidates_by_label,
            candidate_labels=DEFAULT_CANDIDATE_LABELS[: int(sample.candidate_count)],
            selected_label=str(selected_label),
            selected_tile=selected_tile,
            annotation_value=annotation_value,
            relations={
                "operation": "select_tile_same_elevation_as_farmer",
                **farmer_trace_fields,
                "candidate_tile_ids_by_label": dict(candidates_by_label),
                "candidate_levels_by_label": dict(render_map["candidate_levels_by_label"]),
                "selected_label": str(selected_label),
                "selected_tile_id": str(selected_tile_id),
                "selected_tile_level": int(selected_tile.level),
            },
            selection_params_extra={
                "reference_farmer_tile_id": str(farmer_tile_id),
                "reference_farmer_level": int(farmer_tile.level),
            },
            execution_extra=farmer_trace_fields,
            witness_extra={
                "reference_farmer_tile_id": str(farmer_tile_id),
                "reference_farmer_level": int(farmer_tile.level),
                "reference_farmer_bbox": rounded_bbox(farmer_entity.bbox_xyxy),
            },
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
            answer_gt=TypedValue(type="option_letter", value=str(selected_label)),
            annotation_gt=TypedValue(type="bbox", value=list(annotation_value)),
            image=scene.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(sample.selected_key),
        )


__all__ = [
    "IllustrationsIsometricFarmsteadFarmerSameLevelTileLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
