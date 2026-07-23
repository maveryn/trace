"""Select the wallpaper panel matching a Reference panel arrangement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.taxonomy import resolve_task_taxonomy
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import build_prompt_query_spec
from ..shared.annotation import bbox_map_annotation

from .shared.defaults import DOMAIN, SCENE_ID, REFERENCE_LABEL, WallpaperPanelDefaults
from .shared.output import panel_bboxes_by_label, wallpaper_trace_payload
from .shared.prompts import render_wallpaper_prompt_artifacts
from .shared.rendering import render_reference_wallpaper_scene
from .shared.sampling import (
    choose_option_count,
    choose_panel_label,
    choose_wallpaper_group,
    normalize_single_branch,
    option_count_support,
    shuffled_remaining_groups,
    wallpaper_group_support,
)
from .shared.styles import resolve_wallpaper_render_params


TASK_ID = "task_icons__wallpaper_panels__same_pattern_as_reference_label"
QUERY_ID = SINGLE_QUERY_ID
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)

_DEFAULTS = WallpaperPanelDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class IconsWallpaperPanelsSamePatternAsReferenceLabelTask:
    """Select the candidate panel whose wallpaper pattern matches the Reference panel."""

    task_id = TASK_ID
    reasoning_operations = ('matching',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic reference-match selection instance."""

        query_probabilities, task_params = _normalize_public_query(params)
        spec = _resolve_wallpaper_spec(instance_seed=int(instance_seed), params=task_params)
        render_params = resolve_wallpaper_render_params(
            params=task_params,
            render_defaults=_RENDER_DEFAULTS,
            fallback_defaults=_DEFAULTS,
            instance_seed=int(instance_seed),
            include_reference_panel=True,
        )
        if int(render_params["lattice_rows"]) != int(_DEFAULTS.lattice_rows) or int(render_params["lattice_cols"]) != int(
            _DEFAULTS.lattice_cols
        ):
            raise ValueError("wallpaper reference tasks currently use a fixed 4 x 4 motif lattice")

        pool_manifest = str(task_params.get("pool_manifest", group_default(_GEN_DEFAULTS, "pool_manifest", _DEFAULTS.pool_manifest)))
        scene_rng = spawn_rng(int(instance_seed), "scene")
        scene, image = render_reference_wallpaper_scene(
            rng=scene_rng,
            instance_seed=int(instance_seed),
            option_labels=spec.option_labels,
            reference_wallpaper_group_id=spec.reference_wallpaper_group_id,
            wallpaper_group_ids_by_label=spec.wallpaper_group_ids_by_label,
            answer_labels=(spec.answer_label,),
            pool_manifest=pool_manifest,
            render_params=render_params,
            noise_namespace=TASK_ID,
        )
        panel_bboxes = panel_bboxes_by_label(scene.scene_panels)
        reference_bbox = panel_bboxes[REFERENCE_LABEL]
        answer_bbox = panel_bboxes[str(spec.answer_label)]
        annotation = bbox_map_annotation(
            {
                "reference_panel": reference_bbox,
                "selected_panel": answer_bbox,
            }
        )
        _prompt_defaults, prompt_artifacts = render_wallpaper_prompt_artifacts(
            instance_seed=int(instance_seed),
            prompt_defaults=_PROMPT_DEFAULTS,
        )
        taxonomy = resolve_task_taxonomy(str(self.task_id))
        common_ids = {
            "domain": taxonomy.domain,
            "scene_id": taxonomy.scene_id,
            "task_id": str(self.task_id),
            "query_id": QUERY_ID,
        }
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=QUERY_ID,
            params={
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id_probabilities": dict(query_probabilities),
                "option_count": int(spec.option_count),
                "option_count_probabilities": dict(spec.option_count_probabilities),
                "option_labels": list(spec.option_labels),
                "answer_label": str(spec.answer_label),
                "answer_label_probabilities": dict(spec.answer_label_probabilities),
                "reference_wallpaper_group_id": str(spec.reference_wallpaper_group_id),
                "reference_wallpaper_group_probabilities": dict(spec.reference_wallpaper_group_probabilities),
                "wallpaper_group_ids_by_label": dict(spec.wallpaper_group_ids_by_label),
            },
        )
        trace_payload = wallpaper_trace_payload(
            identity=common_ids,
            taxonomy=taxonomy,
            scene_payload=scene,
            render_params=render_params,
            query_spec=query_spec,
            scene_kind="icons_wallpaper_panels_reference_match",
            relations={
                "motif_policy": "exactly_one_candidate_shares_the_reference_wallpaper_group",
                "reference_panel_label": REFERENCE_LABEL,
                "option_labels": list(spec.option_labels),
                "answer_label": str(spec.answer_label),
                "reference_wallpaper_group_id": str(spec.reference_wallpaper_group_id),
                "wallpaper_group_ids_by_label": dict(spec.wallpaper_group_ids_by_label),
                "visible_internal_grid": False,
            },
            render_map={
                "panel_bboxes_px": {str(label): list(bbox) for label, bbox in panel_bboxes.items()},
                "reference_panel_bbox_px": list(reference_bbox),
                "answer_panel_label": str(spec.answer_label),
                "answer_panel_bbox_px": list(answer_bbox),
            },
            execution_trace={
                "scene_variant": "reference_plus_option_wallpaper_panels",
                "question_format": "select_candidate_panel_matching_reference_wallpaper_pattern",
                "option_count": int(spec.option_count),
                "option_labels": list(spec.option_labels),
                "answer_label": str(spec.answer_label),
                "answer_index": int(spec.answer_index),
                "reference_wallpaper_group_id": str(spec.reference_wallpaper_group_id),
                "wallpaper_group_ids_by_label": dict(spec.wallpaper_group_ids_by_label),
                "icon_ids_by_label": dict(scene.icon_ids_by_label),
                "query_id_probabilities": dict(query_probabilities),
                "visible_internal_grid": False,
                "lattice_rows": int(render_params["lattice_rows"]),
                "lattice_cols": int(render_params["lattice_cols"]),
            },
            witness_symbolic={
                "reference_panel_label": REFERENCE_LABEL,
                "answer_label": str(spec.answer_label),
                "reference_panel_bbox_xyxy": list(reference_bbox),
                "answer_panel_bbox_xyxy": list(answer_bbox),
                "reference_wallpaper_group_id": str(spec.reference_wallpaper_group_id),
            },
            projected_annotation=dict(annotation["projected_annotation"]),
            include_reference_panel_width=False,
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=TypedValue(type="option_letter", value=str(spec.answer_label)),
            annotation_gt=TypedValue(
                type=str(annotation["annotation_type"]),
                value={str(key): list(value) for key, value in annotation["annotation_value"].items()},
            ),
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=QUERY_ID,
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
        )


@dataclass(frozen=True)
class _WallpaperSpec:
    """Resolved reference group, answer panel, and distractor groups."""

    option_count: int
    option_labels: Tuple[str, ...]
    answer_label: str
    answer_index: int
    reference_wallpaper_group_id: str
    wallpaper_group_ids_by_label: Dict[str, str]
    option_count_probabilities: Dict[int, float]
    answer_label_probabilities: Dict[str, float]
    reference_wallpaper_group_probabilities: Dict[str, float]


def _normalize_public_query(params: Mapping[str, Any]) -> tuple[Dict[str, float], Dict[str, Any]]:
    """Validate the single public query selector and remove query aliases."""

    return normalize_single_branch(params, accepted=SUPPORTED_QUERY_IDS, selected=QUERY_ID, owner=TASK_ID)


def _resolve_wallpaper_spec(*, instance_seed: int, params: Mapping[str, Any]) -> _WallpaperSpec:
    """Sample one Reference-matching candidate and distinct nonmatching candidates."""

    sample_rng = spawn_rng(int(instance_seed), f"{TASK_ID}.sample")
    option_count_choices = option_count_support(
        params,
        _GEN_DEFAULTS,
        fallback_defaults=_DEFAULTS,
        context="wallpaper reference match",
    )
    group_support = wallpaper_group_support(params, _GEN_DEFAULTS, fallback_defaults=_DEFAULTS)
    option_count, option_labels, option_count_probabilities = choose_option_count(
        sample_rng,
        params,
        option_count_choices,
    )
    if len(group_support) < int(option_count):
        raise ValueError("reference wallpaper matching requires at least one wallpaper group per option panel")
    answer_label, answer_index, answer_label_probabilities = choose_panel_label(
        sample_rng,
        params,
        option_labels,
    )
    reference_wallpaper_group_id, reference_probabilities = choose_wallpaper_group(
        sample_rng,
        params,
        group_support,
        explicit_keys=("reference_wallpaper_group_id", "wallpaper_group_id"),
        context="reference_wallpaper_group_id",
    )
    distractor_groups = shuffled_remaining_groups(
        instance_seed=int(instance_seed),
        owner=TASK_ID,
        support=group_support,
        excluded=str(reference_wallpaper_group_id),
    )
    wallpaper_group_ids_by_label: Dict[str, str] = {}
    distractor_index = 0
    for label in option_labels:
        if str(label) == str(answer_label):
            wallpaper_group_ids_by_label[str(label)] = str(reference_wallpaper_group_id)
        else:
            wallpaper_group_ids_by_label[str(label)] = str(distractor_groups[int(distractor_index)])
            distractor_index += 1

    return _WallpaperSpec(
        option_count=int(option_count),
        option_labels=tuple(str(label) for label in option_labels),
        answer_label=str(answer_label),
        answer_index=int(answer_index),
        reference_wallpaper_group_id=str(reference_wallpaper_group_id),
        wallpaper_group_ids_by_label=dict(wallpaper_group_ids_by_label),
        option_count_probabilities=dict(option_count_probabilities),
        answer_label_probabilities=dict(answer_label_probabilities),
        reference_wallpaper_group_probabilities=dict(reference_probabilities),
    )


__all__ = ["IconsWallpaperPanelsSamePatternAsReferenceLabelTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
