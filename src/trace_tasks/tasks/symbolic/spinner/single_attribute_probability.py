"""Single-spinner probability for one visible sector attribute."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.fixed_query import select_task_query_id

from ._lifecycle import prepare_spinner_task_output_parts, render_spinner_bundle, spinner_task_output_fields
from .shared.annotations import scalar_panel_bbox
from .shared.defaults import load_spinner_defaults
from .shared.rules import (
    build_single_spinner_dataset,
    color_names,
    select_event_candidate,
    shape_names,
    valid_favorable_count,
)
from ...shared.config_defaults import group_default


TASK_ID = "task_symbolic__spinner__single_attribute_probability"
DOMAIN = "symbolic"
SUPPORTED_QUERY_IDS = ("single_color_probability", "single_shape_probability")
_REASONING_LOAD = {
    "single_color_probability": 0.26,
    "single_shape_probability": 0.30,
}


def _configured_count_limits(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    total: int,
) -> tuple[int, int]:
    min_count = int(params.get("single_favorable_count_min", group_default(gen_defaults, "single_favorable_count_min", 2)))
    max_count = int(
        params.get("single_favorable_count_max", group_default(gen_defaults, "single_favorable_count_max", max(2, int(total) - 2)))
    )
    return int(min_count), int(max_count)


def _color_candidates(
    sectors: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    _rng,
) -> List[Dict[str, Any]]:
    total = len(sectors)
    min_count, max_count = _configured_count_limits(params, gen_defaults, total)
    candidates: List[Dict[str, Any]] = []
    for color in color_names(sectors):
        favorable = [
            str(sector["sector_id"])
            for sector in sectors
            if str(sector["color_name"]) == str(color)
        ]
        if valid_favorable_count(len(favorable), total, min_count=min_count, max_count=max_count):
            candidates.append(
                {
                    "event_description": str(color),
                    "target_color": str(color),
                    "favorable_sector_ids": list(favorable),
                }
            )
    return candidates


def _shape_candidates(
    sectors: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    _rng,
) -> List[Dict[str, Any]]:
    total = len(sectors)
    min_count, max_count = _configured_count_limits(params, gen_defaults, total)
    candidates: List[Dict[str, Any]] = []
    for shape in shape_names(sectors):
        favorable = [
            str(sector["sector_id"])
            for sector in sectors
            if str(sector["shape"]) == str(shape)
        ]
        if valid_favorable_count(len(favorable), total, min_count=min_count, max_count=max_count):
            candidates.append(
                {
                    "event_description": f"marked with a {shape}",
                    "target_shape": str(shape),
                    "favorable_sector_ids": list(favorable),
                }
            )
    return candidates


def _build_event(
    *,
    public_query_id: str,
    sectors: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    rng,
) -> Dict[str, Any]:
    builders = {
        "single_color_probability": _color_candidates,
        "single_shape_probability": _shape_candidates,
    }
    if str(public_query_id) not in builders:
        raise ValueError(f"unsupported spinner probability query_id: {public_query_id}")
    candidates = builders[str(public_query_id)](sectors, params, gen_defaults, rng)
    return select_event_candidate(
        candidates,
        rng=rng,
        favorable_key="favorable_sector_ids",
        total_outcome_count=len(sectors),
    )


def _build_dataset(
    *,
    public_query_id: str,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    def _event_builder(**kwargs: Any) -> Mapping[str, Any]:
        return _build_event(public_query_id=str(public_query_id), **kwargs)

    return build_single_spinner_dataset(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        rng_namespace=f"{TASK_ID}.dataset",
        event_builder=_event_builder,
    )


@register_task
class SymbolicSpinnerSingleAttributeProbabilityTask:
    """Compute a probability from one visible color or shape property."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'formula_evaluation')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Own the attribute query/event choice, then bind scalar panel annotation."""

        del max_attempts
        gen_defaults, render_defaults, prompt_defaults = load_spinner_defaults(TASK_ID)
        public_query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SUPPORTED_QUERY_IDS[0],
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.query",
        )
        dataset = _build_dataset(
            public_query_id=str(public_query_id),
            params=task_params,
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
        )
        event = dict(dataset["event"])
        bundle = render_spinner_bundle(
            public_task_id=TASK_ID,
            prompt_query_key=str(public_query_id),
            dataset=dataset,
            params=task_params,
            gen_defaults=gen_defaults,
            render_defaults=render_defaults,
            prompt_defaults=prompt_defaults,
            instance_seed=int(instance_seed),
            event_description=str(event["event_description"]),
        )
        annotation_gt, projected_annotation, witness_symbolic, annotation_source = scalar_panel_bbox(
            bundle.rendered_scene.item_bbox_map,
            item_id="spinner_panel",
        )
        answer_gt = TypedValue(type="option_letter", value=str(bundle.answer_options["correct_label"]))
        output_parts = prepare_spinner_task_output_parts(
            public_query_id=str(public_query_id),
            prompt_query_key=str(public_query_id),
            query_probabilities=query_probabilities,
            dataset=dataset,
            bundle=bundle,
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            projected_annotation=projected_annotation,
            witness_symbolic=witness_symbolic,
            annotation_source=str(annotation_source),
            reasoning_load_base=float(_REASONING_LOAD[str(public_query_id)]),
        )
        return TaskOutput(
            **spinner_task_output_fields(
                output_parts=output_parts,
                answer_gt=answer_gt,
                annotation_gt=annotation_gt,
                query_id=str(public_query_id),
            )
        )


__all__ = [
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "SymbolicSpinnerSingleAttributeProbabilityTask",
]
