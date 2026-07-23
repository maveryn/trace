"""Compute a missing square area in a Pythagorean attached-square tree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.annotation_artifacts import bbox_annotation_artifacts
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.defaults import DOMAIN, POST_IMAGE_NOISE_DEFAULTS, SCENE_DEFAULTS, SCENE_ID
from .shared.output import build_pythagorean_tree_trace_sections
from .shared.prompts import pythagorean_tree_prompt_artifacts
from .shared.rendering import make_render_context, render_pythagorean_tree_scene
from .shared.sampling import LEG_TARGET_ROLES, TREE_TRIPLES, select_leg_target_role, select_tree_triple
from .shared.state import PythagoreanTreePlan, PythagoreanTreeTriple, RenderedPythagoreanTreeScene

TASK_ID = "task_geometry__pythagorean_tree__missing_square_area_value"
SUPPORTED_QUERY_IDS: tuple[str, ...] = ("hypotenuse_square_area", "leg_square_area")
PROMPT_TASK_KEY = "missing_square_area_value"
_TRIPLES: tuple[tuple[int, int, int], ...] = tuple(
    (int(triple.leg_a), int(triple.leg_b), int(triple.hypotenuse)) for triple in TREE_TRIPLES
)


@dataclass(frozen=True)
class _MissingSquareAreaRequest:
    """Task-owned public query and answer binding before pixel rendering."""

    selected_query: str
    query_probabilities: dict[str, float]
    params: dict[str, Any]
    plan: PythagoreanTreePlan
    triple_probabilities: dict[str, float]
    target_role_probabilities: dict[str, float]


def _hypotenuse_square_plan(triple: PythagoreanTreeTriple) -> PythagoreanTreePlan:
    """Bind a hypotenuse-square query to the visible labels and answer."""

    witness = {
        "formula_family": "pythagorean_attached_square_area",
        "target_role": "hypotenuse_square",
        "leg_a": int(triple.leg_a),
        "leg_b": int(triple.leg_b),
        "hypotenuse": int(triple.hypotenuse),
        "leg_square_1_area": int(triple.leg_square_1_area),
        "leg_square_2_area": int(triple.leg_square_2_area),
        "hypotenuse_square_area": int(triple.hypotenuse_square_area),
        "equation": "hypotenuse_square_area = leg_square_1_area + leg_square_2_area",
        "answer_value": int(triple.hypotenuse_square_area),
    }
    return PythagoreanTreePlan(
        triple=triple,
        target_role="hypotenuse_square",
        answer=int(triple.hypotenuse_square_area),
        known_area_labels={
            "leg_square_1": f"Area={int(triple.leg_square_1_area)}",
            "leg_square_2": f"Area={int(triple.leg_square_2_area)}",
            "hypotenuse_square": "Area=?",
        },
        witness=dict(witness),
    )


def _leg_square_plan(*, triple: PythagoreanTreeTriple, target_role: str) -> PythagoreanTreePlan:
    """Bind a leg-square query to the visible labels and answer."""

    if str(target_role) == "leg_square_1":
        answer = int(triple.leg_square_1_area)
        equation = "leg_square_1_area = hypotenuse_square_area - leg_square_2_area"
        labels = {
            "leg_square_1": "Area=?",
            "leg_square_2": f"Area={int(triple.leg_square_2_area)}",
            "hypotenuse_square": f"Area={int(triple.hypotenuse_square_area)}",
        }
    elif str(target_role) == "leg_square_2":
        answer = int(triple.leg_square_2_area)
        equation = "leg_square_2_area = hypotenuse_square_area - leg_square_1_area"
        labels = {
            "leg_square_1": f"Area={int(triple.leg_square_1_area)}",
            "leg_square_2": "Area=?",
            "hypotenuse_square": f"Area={int(triple.hypotenuse_square_area)}",
        }
    else:
        raise ValueError(f"unsupported pythagorean tree target_role: {target_role}")
    witness = {
        "formula_family": "pythagorean_attached_square_area",
        "target_role": str(target_role),
        "leg_a": int(triple.leg_a),
        "leg_b": int(triple.leg_b),
        "hypotenuse": int(triple.hypotenuse),
        "leg_square_1_area": int(triple.leg_square_1_area),
        "leg_square_2_area": int(triple.leg_square_2_area),
        "hypotenuse_square_area": int(triple.hypotenuse_square_area),
        "equation": str(equation),
        "answer_value": int(answer),
    }
    return PythagoreanTreePlan(
        triple=triple,
        target_role=str(target_role),
        answer=int(answer),
        known_area_labels=dict(labels),
        witness=dict(witness),
    )


def _resolve_missing_square_area_request(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
) -> _MissingSquareAreaRequest:
    """Resolve the public query branch, then bind one right-triangle case."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id="hypotenuse_square_area",
        task_id=TASK_ID,
        namespace=f"{TASK_ID}.query",
    )
    triple, triple_probabilities = select_tree_triple(
        params=task_params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.triple",
    )
    if str(selected_query) == "hypotenuse_square_area":
        plan = _hypotenuse_square_plan(triple)
        target_role_probabilities = {"hypotenuse_square": 1.0}
    elif str(selected_query) == "leg_square_area":
        target_role, target_role_probabilities = select_leg_target_role(
            params=task_params,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.leg_target_role",
        )
        plan = _leg_square_plan(triple=triple, target_role=str(target_role))
    else:
        raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_query}")
    return _MissingSquareAreaRequest(
        selected_query=str(selected_query),
        query_probabilities=dict(query_probabilities),
        params=dict(task_params),
        plan=plan,
        triple_probabilities=dict(triple_probabilities),
        target_role_probabilities=dict(target_role_probabilities),
    )


def _render_missing_square_area(
    *,
    request: _MissingSquareAreaRequest,
    instance_seed: int,
    max_attempts: int,
    rendering_defaults: Mapping[str, Any],
) -> tuple[RenderedPythagoreanTreeScene, dict[str, Any]]:
    """Render the task-bound tree, retrying only layout/style failures."""

    rendered: RenderedPythagoreanTreeScene | None = None
    render_meta: dict[str, Any] | None = None
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + int(attempt)
        attempt_params = dict(request.params)
        attempt_params["_render_attempt"] = int(attempt)
        try:
            ctx, attempt_meta = make_render_context(
                instance_seed=attempt_seed,
                params=attempt_params,
                rendering_defaults=rendering_defaults,
            )
            rendered = render_pythagorean_tree_scene(
                ctx,
                request.plan,
                instance_seed=attempt_seed,
            )
            render_meta = dict(attempt_meta)
            render_meta["single_object_scene_rotation"] = ctx.scene_transform.metadata()
            break
        except Exception as exc:
            last_error = exc
            continue
    if rendered is None or render_meta is None:
        raise RuntimeError(f"failed to render {TASK_ID}") from last_error
    return rendered, render_meta


def _target_square_bbox(
    *,
    request: _MissingSquareAreaRequest,
    rendered: RenderedPythagoreanTreeScene,
) -> tuple[float, float, float, float]:
    """Return the scalar bbox around the target square marked Area=?."""

    target_role = str(request.plan.target_role)
    if target_role not in rendered.square_bboxes:
        raise ValueError(f"missing target square bbox: {target_role}")
    return rendered.square_bboxes[str(target_role)]


def _prompt_annotation_hint_and_example(
    *,
    query_id: str,
) -> tuple[str, list[int]]:
    """Return public annotation guidance for the scalar target-square box."""

    if str(query_id) not in SUPPORTED_QUERY_IDS:
        raise ValueError(f"unsupported prompt query_id: {query_id}")
    return (
        'set "annotation" to the pixel bounding box [x0,y0,x1,y1] around the square marked Area=?',
        [220, 150, 380, 310],
    )


@register_task
class GeometryPythagoreanTreeMissingSquareAreaValueTask:
    """Compute a missing attached-square area from the Pythagorean relation."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Own query selection, answer binding, annotation binding, and final output."""

        _generation_defaults, rendering_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
            SCENE_DEFAULTS,
            task_id=TASK_ID,
        )
        request = _resolve_missing_square_area_request(
            instance_seed=int(instance_seed),
            params=params,
        )
        rendered, render_meta = _render_missing_square_area(
            request=request,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            rendering_defaults=rendering_defaults,
        )
        image, noise_meta = apply_post_image_noise(
            rendered.image,
            instance_seed=int(instance_seed),
            params=request.params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )
        annotation_artifacts = bbox_annotation_artifacts(
            _target_square_bbox(request=request, rendered=rendered)
        )
        annotation_hint, annotation_example = _prompt_annotation_hint_and_example(
            query_id=request.selected_query,
        )
        _prompt_defaults, prompt_artifacts = pythagorean_tree_prompt_artifacts(
            prompt_defaults=prompt_defaults,
            prompt_task_key=PROMPT_TASK_KEY,
            prompt_query_key=request.selected_query,
            annotation_hint=annotation_hint,
            json_example_annotation=annotation_example,
            answer=int(request.plan.answer),
            instance_seed=int(instance_seed),
        )
        sections = build_pythagorean_tree_trace_sections(
            plan=request.plan,
            rendered=rendered,
            annotation_artifacts=annotation_artifacts,
            annotation_roles=("target_square",),
            triple_probabilities=request.triple_probabilities,
            target_role_probabilities=request.target_role_probabilities,
            render_meta=render_meta,
            noise_meta=noise_meta,
            image_size=(int(image.size[0]), int(image.size[1])),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=request.selected_query,
            params={
                "scene_id": SCENE_ID,
                "query_id": request.selected_query,
                "query_id_probabilities": dict(request.query_probabilities),
                **sections.query_params_base,
            },
        )
        query_spec["scene_id"] = SCENE_ID
        scene_ir = {
            "scene_kind": sections.scene_kind,
            "scene_id": SCENE_ID,
            "task_id": TASK_ID,
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "query_id": request.selected_query,
                **sections.scene_relations,
            },
        }
        trace_payload = {
            "scene_ir": scene_ir,
            "query_spec": query_spec,
            "render_spec": {
                "task_id": TASK_ID,
                "scene_id": SCENE_ID,
                "query_id": request.selected_query,
                "prompt": {
                    "prompt_bundle_id": str(prompt_artifacts.prompt_variant.get("prompt_bundle_id", "")),
                    "prompt_variant": dict(prompt_artifacts.prompt_variant),
                    "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                    "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                },
                **sections.render_spec_base,
            },
            "render_map": dict(rendered.render_map),
            "execution_trace": {
                "task_id": TASK_ID,
                "scene_id": SCENE_ID,
                "query_id": request.selected_query,
                "query_id_probabilities": dict(request.query_probabilities),
                "objective": "missing_square_area_value",
                "answer": int(request.plan.answer),
                "leg_square_1_area": int(request.plan.triple.leg_square_1_area),
                "leg_square_2_area": int(request.plan.triple.leg_square_2_area),
                "hypotenuse_square_area": int(request.plan.triple.hypotenuse_square_area),
                **sections.execution_common,
            },
            "witness_symbolic": {
                "type": "pythagorean_attached_square_area",
                "task_id": TASK_ID,
                "query_id": request.selected_query,
                **sections.witness_common,
            },
            "projected_annotation": dict(annotation_artifacts.projected_annotation),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(request.plan.answer)),
            annotation_gt=TypedValue(
                type=annotation_artifacts.annotation_type,
                value=annotation_artifacts.value,
            ),
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=request.selected_query,
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


__all__ = [
    "GeometryPythagoreanTreeMissingSquareAreaValueTask",
    "LEG_TARGET_ROLES",
    "SCENE_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "_TRIPLES",
]
