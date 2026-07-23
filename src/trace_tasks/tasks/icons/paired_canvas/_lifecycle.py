"""Count right-panel icons whose attribute changed from the left panel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.seed import hash64, spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...shared.config_defaults import (
    group_default,
    load_scene_generation_rendering_prompt_defaults,
)
from ...shared.fixed_query import (
    SINGLE_QUERY_ID,
    explicit_query_id_param,
    rewrite_public_query_output,
    select_task_query_id,
    strip_query_id_params,
)
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import build_prompt_query_spec
from ..shared.annotation import icon_bbox_set_annotation
from ..shared.icon_task_rendering import resolve_icon_render_params

from .shared.annotations import bboxes_from_icon_indices
from .shared.defaults import SCENE_ID, PairedCanvasDefaults
from .shared.output import build_paired_canvas_trace_payload
from .shared.prompts import build_paired_prompt, required_paired_prompt_defaults
from .shared.rendering import render_paired_canvas
from .shared.sampling import (
    load_icon_pool_from_params,
    make_icon_spec,
    resolve_paired_counts,
    sample_base_attributes,
    sample_palette,
    sample_positions,
)


DOMAIN = "icons"
QUERY_IDS: Tuple[str, ...] = ("color_changed_count", "rotation_changed_count")
_QUERY_TO_ATTRIBUTE = {
    "color_changed_count": "color",
    "rotation_changed_count": "rotation",
}

_DEFAULTS = PairedCanvasDefaults()


@dataclass(frozen=True)
class _AttributeChangeScene:
    """Task-owned symbolic payload for one attribute-change count instance."""

    image: Any
    panel_geometry: Dict[str, Any]
    left_icons: Tuple[Dict[str, Any], ...]
    right_icons: Tuple[Dict[str, Any], ...]
    matching_right_indices: Tuple[int, ...]
    matching_left_indices: Tuple[int, ...]
    target_count: int
    object_count: int
    distractor_count: int
    query_id: str
    query_probabilities: Dict[str, float]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    object_count_probabilities: Dict[str, float]
    target_count_probabilities: Dict[str, float]
    distractor_count_probabilities: Dict[str, float]
    question_format: str
    trace_relation: Dict[str, Any]


def _select_query(
    instance_seed: int,
    params: Mapping[str, Any],
    *,
    task_id: str,
    query_ids: Tuple[str, ...],
) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
    """Select and validate one semantic attribute-change query branch."""

    return select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(query_ids),
        default_query_id=str(query_ids[0]),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )


def _rotation_candidates(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> Tuple[int, ...]:
    raw = params.get(
        "rotation_candidates_degrees",
        group_default(gen_defaults, "rotation_candidates_degrees", _DEFAULTS.rotation_candidates_degrees),
    )
    return tuple(int(value) for value in raw)


def _changed_attr(
    rng,
    *,
    attr: Mapping[str, Any],
    changed_attribute: str,
    palette: Sequence[Tuple[int, int, int]],
    render_params: Mapping[str, Any],
    size_scale_small: float,
    size_scale_large: float,
    rotation_candidates: Sequence[int],
) -> Dict[str, Any]:
    """Return a copied attribute dict with exactly one requested visual axis changed."""

    next_attr = dict(attr)
    if changed_attribute == "color":
        choices = [tuple(color) for color in palette if tuple(color) != tuple(attr["tint_rgb"])]
        next_attr["tint_rgb"] = tuple(rng.choice(choices))
    elif changed_attribute == "size":
        base = int(attr["size_px"])
        direction = rng.choice(("grow", "shrink"))
        if str(direction) == "grow":
            next_attr["size_px"] = int(
                min(
                    int(render_params["scene_icon_size_max_px"]),
                    max(base + 12, round(base * float(size_scale_large))),
                )
            )
        else:
            next_attr["size_px"] = int(
                max(
                    int(render_params["scene_icon_size_min_px"]),
                    min(base - 12, round(base * float(size_scale_small))),
                )
            )
        if int(next_attr["size_px"]) == int(base):
            next_attr["size_px"] = int(
                max(
                    int(render_params["scene_icon_size_min_px"]),
                    min(int(render_params["scene_icon_size_max_px"]), base + 14),
                )
            )
    elif changed_attribute == "rotation":
        choices = [
            int(value)
            for value in rotation_candidates
            if int(value) % 360 != int(attr["rotation_degrees"]) % 360
        ]
        next_attr["rotation_degrees"] = int(rng.choice(choices))
    else:
        raise ValueError(f"unsupported changed attribute: {changed_attribute}")
    return next_attr


def _make_scene(
    *,
    instance_seed: int,
    task_id: str,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    render_params: Mapping[str, Any],
    query_id: str,
    query_probabilities: Mapping[str, float],
) -> _AttributeChangeScene:
    """Render a paired-panel scene with controlled queried attribute changes."""

    rng = spawn_rng(int(instance_seed), "scene")
    active_attribute = _QUERY_TO_ATTRIBUTE[str(query_id)]
    (
        object_count,
        object_count_probabilities,
        target_count,
        target_count_probabilities,
        distractor_count,
        distractor_count_probabilities,
    ) = resolve_paired_counts(
        rng,
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        defaults=_DEFAULTS,
    )
    pool = list(load_icon_pool_from_params(params=params, gen_defaults=gen_defaults, defaults=_DEFAULTS))
    rng.shuffle(pool)
    palette = sample_palette(rng, render_params=render_params)
    rotation_candidates = _rotation_candidates(params, gen_defaults)
    attrs = sample_base_attributes(
        rng,
        pool=pool,
        palette=palette,
        count=int(object_count),
        render_params=render_params,
        rotation_candidates=rotation_candidates,
    )
    match_indices = set(rng.sample(list(range(int(object_count))), int(target_count)))
    other_attributes = [value for value in ("color", "size", "rotation") if value != active_attribute]
    size_scale_small = float(
        params.get("size_scale_small", group_default(gen_defaults, "size_scale_small", _DEFAULTS.size_scale_small))
    )
    size_scale_large = float(
        params.get("size_scale_large", group_default(gen_defaults, "size_scale_large", _DEFAULTS.size_scale_large))
    )
    gap = float(
        params.get(
            "min_center_gap_frac",
            group_default(render_defaults, "min_center_gap_frac", _DEFAULTS.min_center_gap_frac),
        )
    )
    positions = sample_positions(rng, count=int(object_count), min_gap_frac=gap)
    left_specs = []
    right_specs = []
    changed_attributes_by_index = []
    for index, (attr, pos) in enumerate(zip(attrs, positions)):
        right_attr = dict(attr)
        changed_attributes = []
        if int(index) in match_indices:
            right_attr = _changed_attr(
                rng,
                attr=right_attr,
                changed_attribute=active_attribute,
                palette=palette,
                render_params=render_params,
                size_scale_small=float(size_scale_small),
                size_scale_large=float(size_scale_large),
                rotation_candidates=rotation_candidates,
            )
            changed_attributes.append(str(active_attribute))
        elif rng.random() < 0.55:
            distractor_attribute = str(rng.choice(tuple(other_attributes)))
            right_attr = _changed_attr(
                rng,
                attr=right_attr,
                changed_attribute=distractor_attribute,
                palette=palette,
                render_params=render_params,
                size_scale_small=float(size_scale_small),
                size_scale_large=float(size_scale_large),
                rotation_candidates=rotation_candidates,
            )
            changed_attributes.append(str(distractor_attribute))
        changed_attributes_by_index.append(tuple(changed_attributes))
        left_specs.append(
            make_icon_spec(
                instance_seed=int(instance_seed),
                namespace=f"{task_id}:left:{index}",
                render_params=render_params,
                instance_id=f"left_{index}",
                identity_id=str(attr["identity_id"]),
                icon_id=str(attr["icon_id"]),
                panel="left",
                position=pos,
                tint_rgb=tuple(attr["tint_rgb"]),
                size_px=int(attr["size_px"]),
                rotation_degrees=int(attr["rotation_degrees"]),
            )
        )
        right_specs.append(
            make_icon_spec(
                instance_seed=int(instance_seed),
                namespace=f"{task_id}:right:{index}",
                render_params=render_params,
                instance_id=f"right_{index}",
                identity_id=str(attr["identity_id"]),
                icon_id=str(right_attr["icon_id"]),
                panel="right",
                position=pos,
                tint_rgb=tuple(right_attr["tint_rgb"]),
                size_px=int(right_attr["size_px"]),
                rotation_degrees=int(right_attr["rotation_degrees"]),
            )
        )
    rendered = render_paired_canvas(left_icons=left_specs, right_icons=right_specs, render_params=render_params)
    left_icons = []
    right_icons = []
    for index, icon in enumerate(rendered.left_icons):
        item = dict(icon)
        item["pair_index"] = int(index)
        left_icons.append(item)
    for index, icon in enumerate(rendered.right_icons):
        item = dict(icon)
        item["pair_index"] = int(index)
        item["changed_attributes"] = [str(value) for value in changed_attributes_by_index[int(index)]]
        item["is_match"] = bool(int(index) in match_indices)
        right_icons.append(item)

    return _AttributeChangeScene(
        image=rendered.image,
        panel_geometry=dict(rendered.panel_geometry),
        left_icons=tuple(left_icons),
        right_icons=tuple(right_icons),
        matching_right_indices=tuple(sorted(int(index) for index in match_indices)),
        matching_left_indices=tuple(sorted(int(index) for index in match_indices)),
        target_count=int(target_count),
        object_count=int(object_count),
        distractor_count=int(distractor_count),
        query_id=str(query_id),
        query_probabilities=dict(query_probabilities),
        sampled_palette_rgb=tuple(palette),
        object_count_probabilities=dict(object_count_probabilities),
        target_count_probabilities=dict(target_count_probabilities),
        distractor_count_probabilities=dict(distractor_count_probabilities),
        question_format="count_right_icons_with_queried_attribute_changed_from_left_counterpart",
        trace_relation={
            "counting_target": str(query_id),
            "active_attribute": str(active_attribute),
            "changed_attributes_by_pair": [list(value) for value in changed_attributes_by_index],
        },
    )


def _generate_attribute_scene_with_retries(
    *,
    instance_seed: int,
    task_id: str,
    task_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    render_params: Mapping[str, Any],
    query_id: str,
    query_probabilities: Mapping[str, float],
    max_attempts: int,
) -> _AttributeChangeScene:
    """Retry only the attribute-change scene sampler; do not relax constraints."""

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            attempt_seed = int(hash64(int(instance_seed), str(task_id), int(attempt_index)))
            return _make_scene(
                instance_seed=attempt_seed,
                task_id=str(task_id),
                params=task_params,
                gen_defaults=gen_defaults,
                render_defaults=render_defaults,
                render_params=render_params,
                query_id=str(query_id),
                query_probabilities=query_probabilities,
            )
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"failed to generate {task_id} instance") from last_error


def _attribute_change_task_output(
    *,
    domain: str,
    task_id: str,
    instance_seed: int,
    scene: _AttributeChangeScene,
    prompt_defaults_source: Mapping[str, Any],
    render_params: Mapping[str, Any],
) -> TaskOutput:
    """Build the final TaskOutput for an attribute-change count scene."""

    prompt_defaults = required_paired_prompt_defaults(
        prompt_defaults_source,
        run_namespace=str(task_id),
        extra_required_keys=(f"question_text_{scene.query_id}",),
    )
    annotation_bboxes = bboxes_from_icon_indices(
        panel_icons=scene.right_icons,
        indices=scene.matching_right_indices,
    )
    query_params = {
        "query_id": str(scene.query_id),
        "query_id_probabilities": dict(scene.query_probabilities),
        "object_count": int(scene.object_count),
        "object_count_probabilities": dict(scene.object_count_probabilities),
        "target_count": int(scene.target_count),
        "target_count_probabilities": dict(scene.target_count_probabilities),
        "distractor_count": int(scene.distractor_count),
        "distractor_count_probabilities": dict(scene.distractor_count_probabilities),
        "annotation_panel": "right",
    }
    execution_trace = {
        "scene_variant": SCENE_ID,
        "query_id": str(scene.query_id),
        "query_id_probabilities": dict(scene.query_probabilities),
        "question_format": str(scene.question_format),
        "object_count": int(scene.object_count),
        "object_count_probabilities": dict(scene.object_count_probabilities),
        "target_count": int(scene.target_count),
        "target_count_probabilities": dict(scene.target_count_probabilities),
        "distractor_count": int(scene.distractor_count),
        "distractor_count_probabilities": dict(scene.distractor_count_probabilities),
        "matching_right_indices": list(scene.matching_right_indices),
        "matching_left_indices": list(scene.matching_left_indices),
        "annotation_panel": "right",
        **dict(scene.trace_relation),
    }
    prompt_artifacts = build_paired_prompt(
        domain=domain,
        prompt_defaults=prompt_defaults,
        question_text=str(prompt_defaults[f"question_text_{scene.query_id}"]),
        instance_seed=int(instance_seed),
    )
    annotation_artifacts = icon_bbox_set_annotation(annotation_bboxes)
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(scene.query_id),
        params=query_params,
    )
    trace_payload = build_paired_canvas_trace_payload(
        scene_kind=f"icons_{SCENE_ID}",
        panel_geometry=scene.panel_geometry,
        left_icons=scene.left_icons,
        right_icons=scene.right_icons,
        relations=scene.trace_relation,
        query_spec=query_spec,
        render_params=render_params,
        sampled_palette_rgb=scene.sampled_palette_rgb,
        render_map_extra=None,
        execution_trace=execution_trace,
        witness_symbolic={
            "query_id": str(scene.query_id),
            "matching_right_indices": list(scene.matching_right_indices),
            "matching_left_indices": list(scene.matching_left_indices),
            "annotation_panel": "right",
        },
        annotation_payload=annotation_artifacts,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=TypedValue(type="integer", value=int(scene.target_count)),
        annotation_gt=TypedValue(
            type=str(annotation_artifacts["annotation_type"]),
            value=[list(bbox) for bbox in annotation_artifacts["annotation_value"]],
        ),
        image=scene.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(scene.query_id),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


class IconsPanelAttributeChangeCountTaskBase:
    """Count Right icons whose queried visual attribute changed."""

    domain = DOMAIN
    supported_query_ids = QUERY_IDS
    query_ids = QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic paired-panel attribute-change count instance."""

        if not str(self.task_id):
            raise RuntimeError("task_id must be set on icon attribute-change task subclasses")
        fixed_public_query = tuple(self.supported_query_ids) == (SINGLE_QUERY_ID,) and len(tuple(self.query_ids)) == 1
        task_params_source = dict(params)
        if fixed_public_query:
            explicit_query_id = explicit_query_id_param(task_params_source, allow_default=True)
            if explicit_query_id is not None and str(explicit_query_id) != SINGLE_QUERY_ID:
                raise ValueError(
                    f"unsupported public query_id for {self.task_id}: {explicit_query_id}; expected {SINGLE_QUERY_ID}"
                )
            task_params_source = strip_query_id_params(task_params_source)
        gen_defaults, render_defaults, prompt_defaults = load_scene_generation_rendering_prompt_defaults(
            DOMAIN,
            SCENE_ID,
            task_id=str(self.task_id),
        )
        query_id, query_probabilities, task_params = _select_query(
            int(instance_seed),
            task_params_source,
            task_id=str(self.task_id),
            query_ids=tuple(self.query_ids),
        )
        render_params = resolve_icon_render_params(
            params=task_params,
            render_defaults=render_defaults,
            fallback_defaults=_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        scene = _generate_attribute_scene_with_retries(
            instance_seed=int(instance_seed),
            task_id=str(self.task_id),
            task_params=task_params,
            gen_defaults=gen_defaults,
            render_defaults=render_defaults,
            render_params=render_params,
            query_id=str(query_id),
            query_probabilities=query_probabilities,
            max_attempts=int(max_attempts),
        )
        output = _attribute_change_task_output(
            domain=self.domain,
            task_id=str(self.task_id),
            instance_seed=int(instance_seed),
            scene=scene,
            prompt_defaults_source=prompt_defaults,
            render_params=render_params,
        )
        if fixed_public_query:
            return rewrite_public_query_output(
                output,
                query_id=SINGLE_QUERY_ID,
                scene_id=SCENE_ID,
                task_id=str(self.task_id),
                preserve_internal_query_id_as="internal_query_id",
                query_id_probabilities={SINGLE_QUERY_ID: 1.0},
                params_query_id_probabilities={SINGLE_QUERY_ID: 1.0},
            )
        return output


def run_panel_attribute_change_task(
    *,
    task: IconsPanelAttributeChangeCountTaskBase,
    instance_seed: int,
    params: Dict[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run the scene-private attribute-change lifecycle for one public objective."""

    return IconsPanelAttributeChangeCountTaskBase.generate(
        task,
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
    )


__all__ = ["IconsPanelAttributeChangeCountTaskBase", "QUERY_IDS", "run_panel_attribute_change_task"]
