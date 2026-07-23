"""Select the lettered terrain tile at the elevation extremum."""

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
    isometric_quarry_elevation_render_map,
    make_tile_label_trace_payload,
    rounded_bbox,
)
from .shared.prompts import build_isometric_quarry_task_prompt_with_default_slots
from .shared.rendering import (
    DEFAULT_CANDIDATE_LABELS,
    SCENE_ID,
    render_isometric_quarry_labeled_scene_with_retry,
)
from .shared.sampling import sample_label_task_spec
from .shared.state import IsoQuarryScene, IsoQuarryTile
from .shared.spatial_primitives import eligible_label_tiles_by_level, label_clear_of_candidate_tiles


TASK_ID = "task_illustrations__isometric_quarry__terrain_elevation_extremum_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("highest_terrain_tile", "lowest_terrain_tile")


_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any]) -> Any:
    return sample_label_task_spec(
        instance_seed=int(instance_seed),
        params=params,
        identity_label=TASK_ID,
        query_options=SUPPORTED_QUERY_IDS,
        default_query_id="highest_terrain_tile",
        generation_defaults=_GEN_DEFAULTS,
        rendering_defaults=_RENDER_DEFAULTS,
        max_candidate_count=len(DEFAULT_CANDIDATE_LABELS),
    )


def _select_candidate_tiles(
    *,
    scene: IsoQuarryScene,
    selected_query: str,
    candidate_count: int,
    instance_seed: int,
) -> tuple[dict[str, str], str]:
    """Select one unique elevation-extremum tile plus distractor tiles."""

    by_level = eligible_label_tiles_by_level(scene)
    active_levels = tuple(level for level in sorted(by_level) if by_level[level])
    if len(active_levels) < 2:
        raise ValueError("not enough active elevation levels with eligible candidate tiles")
    if str(selected_query) == "highest_terrain_tile":
        target_level = max(active_levels)
        distractor_levels = [level for level in active_levels if int(level) < int(target_level)]
    elif str(selected_query) == "lowest_terrain_tile":
        target_level = min(active_levels)
        distractor_levels = [level for level in active_levels if int(level) > int(target_level)]
    else:
        raise ValueError(f"unsupported elevation query: {selected_query}")

    answer_pool = list(by_level[int(target_level)])
    distractor_pool = [tile for level in distractor_levels for tile in by_level[int(level)]]
    if not answer_pool:
        raise ValueError(f"no eligible target tiles at level {target_level}")
    if len(distractor_pool) < int(candidate_count) - 1:
        raise ValueError("not enough eligible elevation distractor tiles")

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:candidate_tiles:{selected_query}")
    rng.shuffle(answer_pool)
    rng.shuffle(distractor_pool)
    selected_tiles: list[IsoQuarryTile] | None = None
    for answer_tile in answer_pool:
        candidate_tiles = [answer_tile]
        for tile in distractor_pool:
            if str(tile.tile_id) == str(answer_tile.tile_id):
                continue
            if not label_clear_of_candidate_tiles(candidate_tiles, tile):
                continue
            candidate_tiles.append(tile)
            if len(candidate_tiles) >= int(candidate_count):
                break
        if len(candidate_tiles) >= int(candidate_count):
            selected_tiles = candidate_tiles
            break
    if selected_tiles is None:
        raise ValueError("not enough label-separated candidate tiles")
    labels = list(DEFAULT_CANDIDATE_LABELS[: int(candidate_count)])
    selected_label = str(labels[int(instance_seed) % int(candidate_count)])
    distractor_labels = [str(label) for label in labels if str(label) != selected_label]
    candidate_tile_ids_by_label = {selected_label: str(selected_tiles[0].tile_id)}
    for label, tile in zip(distractor_labels, selected_tiles[1:]):
        candidate_tile_ids_by_label[str(label)] = str(tile.tile_id)
    candidate_tile_ids_by_label = {str(label): str(candidate_tile_ids_by_label[str(label)]) for label in labels}
    return candidate_tile_ids_by_label, selected_label


@register_task
class IllustrationsIsometricQuarryTerrainElevationExtremumLabelTask:
    """Choose the lettered terrain tile that is highest or lowest."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one option-selection instance with task-owned candidate and answer binding."""

        sample = _sample_spec(instance_seed=int(instance_seed), params=params)
        _prompt_defaults, prompt_artifacts = build_isometric_quarry_task_prompt_with_default_slots(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults_source=_PROMPT_DEFAULTS,
            prompt_query_key=sample.prompt_key,
            answer_hint_key="answer_hint_terrain_elevation_extremum_label",
            annotation_hint_key="annotation_hint_terrain_elevation_extremum_label",
            json_example_key="json_example_terrain_elevation_extremum_label",
            json_example_answer_only_key="json_example_answer_only_terrain_elevation_extremum_label",
            context_label=TASK_ID,
            instance_seed=int(instance_seed),
        )
        def _select_from_scene(base_scene: IsoQuarryScene, scene_seed: int) -> Mapping[str, Any]:
            candidates, answer_label = _select_candidate_tiles(
                scene=base_scene,
                selected_query=sample.selected_key,
                candidate_count=sample.candidate_count,
                instance_seed=scene_seed,
            )
            return {"candidate_tile_ids_by_label": candidates, "selected_label": answer_label}

        rendered = render_isometric_quarry_labeled_scene_with_retry(
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            sample=sample,
            params=params,
            rendering_defaults=_RENDER_DEFAULTS,
            font_seed_label=TASK_ID,
            label_namespace_suffix="terrain_tile_labels",
            selection_fn=_select_from_scene,
        )
        scene = rendered["scene"]
        candidates_by_label = dict(rendered["candidate_tile_ids_by_label"])
        selected_label = str(rendered["selected_label"])
        label_font_trace = dict(rendered["label_font_trace"])

        tiles_by_id = {str(tile.tile_id): tile for tile in scene.tiles}
        selected_tile_id = str(candidates_by_label[str(selected_label)])
        selected_tile = tiles_by_id[selected_tile_id]
        annotation_value = rounded_bbox(selected_tile.bbox_xyxy)
        render_map = isometric_quarry_elevation_render_map(
            scene=scene,
            candidate_tile_ids_by_label=candidates_by_label,
            selected_label=str(selected_label),
        )
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
                "operation": "select_elevation_extremum",
                "extremum": "highest" if sample.selected_key == "highest_terrain_tile" else "lowest",
                "candidate_tile_ids_by_label": dict(candidates_by_label),
                "selected_label": str(selected_label),
                "selected_tile_id": str(selected_tile_id),
                "selected_tile_level": int(selected_tile.level),
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
    "IllustrationsIsometricQuarryTerrainElevationExtremumLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
