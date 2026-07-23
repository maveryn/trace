"""Two-spinner probability for a visible color event."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default
from ...shared.fixed_query import select_task_query_id

from ._lifecycle import prepare_spinner_task_output_parts, render_spinner_bundle, spinner_task_output_fields
from .shared.annotations import pair_panel_bbox_map
from .shared.defaults import load_spinner_defaults
from .shared.rules import build_pair_spinner_dataset, color_names, select_event_candidate, valid_favorable_count


TASK_ID = "task_symbolic__spinner__pair_color_event_probability"
DOMAIN = "symbolic"
SUPPORTED_QUERY_IDS = (
    "pair_both_target_color_probability",
    "pair_at_least_one_target_color_probability",
    "pair_same_color_probability",
)
_REASONING_LOAD = {
    "pair_both_target_color_probability": 0.42,
    "pair_at_least_one_target_color_probability": 0.50,
    "pair_same_color_probability": 0.54,
}


def _configured_count_limits(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    total: int,
) -> tuple[int, int]:
    min_count = int(params.get("pair_favorable_count_min", group_default(gen_defaults, "pair_favorable_count_min", 3)))
    max_count = int(
        params.get("pair_favorable_count_max", group_default(gen_defaults, "pair_favorable_count_max", max(3, int(total) - 3)))
    )
    return int(min_count), int(max_count)


def _both_target_color_candidates(
    sectors_a: Sequence[Mapping[str, Any]],
    sectors_b: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    rng,
) -> List[Dict[str, Any]]:
    del rng
    total = int(len(sectors_a) * len(sectors_b))
    min_count, max_count = _configured_count_limits(params, gen_defaults, total)
    candidates: List[Dict[str, Any]] = []
    shared_colors = sorted(set(color_names(sectors_a)) & set(color_names(sectors_b)))
    for color in shared_colors:
        favorable_pairs = [
            [str(a["sector_id"]), str(b["sector_id"])]
            for a in sectors_a
            for b in sectors_b
            if str(a["color_name"]) == str(color) and str(b["color_name"]) == str(color)
        ]
        if valid_favorable_count(len(favorable_pairs), total, min_count=min_count, max_count=max_count):
            supporting = [
                str(sector["sector_id"])
                for sector in [*sectors_a, *sectors_b]
                if str(sector["color_name"]) == str(color)
            ]
            candidates.append(
                {
                    "event_description": f"Spinner A lands on {color} and Spinner B lands on {color}",
                    "target_color": str(color),
                    "favorable_pairs": list(favorable_pairs),
                    "supporting_sector_ids": list(supporting),
                }
            )
    return candidates


def _at_least_one_target_color_candidates(
    sectors_a: Sequence[Mapping[str, Any]],
    sectors_b: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    rng,
) -> List[Dict[str, Any]]:
    del rng
    total = int(len(sectors_a) * len(sectors_b))
    min_count, max_count = _configured_count_limits(params, gen_defaults, total)
    candidates: List[Dict[str, Any]] = []
    colors = sorted(set(color_names(sectors_a)) | set(color_names(sectors_b)))
    for color in colors:
        favorable_pairs = [
            [str(a["sector_id"]), str(b["sector_id"])]
            for a in sectors_a
            for b in sectors_b
            if str(a["color_name"]) == str(color) or str(b["color_name"]) == str(color)
        ]
        if valid_favorable_count(len(favorable_pairs), total, min_count=min_count, max_count=max_count):
            supporting = [
                str(sector["sector_id"])
                for sector in [*sectors_a, *sectors_b]
                if str(sector["color_name"]) == str(color)
            ]
            candidates.append(
                {
                    "event_description": f"at least one spinner lands on {color}",
                    "target_color": str(color),
                    "favorable_pairs": list(favorable_pairs),
                    "supporting_sector_ids": list(supporting),
                }
            )
    return candidates


def _same_color_candidates(
    sectors_a: Sequence[Mapping[str, Any]],
    sectors_b: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    rng,
) -> List[Dict[str, Any]]:
    del rng
    total = int(len(sectors_a) * len(sectors_b))
    min_count, max_count = _configured_count_limits(params, gen_defaults, total)
    shared_colors = sorted(set(color_names(sectors_a)) & set(color_names(sectors_b)))
    favorable_pairs = [
        [str(a["sector_id"]), str(b["sector_id"])]
        for a in sectors_a
        for b in sectors_b
        if str(a["color_name"]) == str(b["color_name"])
    ]
    if not valid_favorable_count(len(favorable_pairs), total, min_count=min_count, max_count=max_count):
        return []
    supporting = [
        str(sector["sector_id"])
        for sector in [*sectors_a, *sectors_b]
        if str(sector["color_name"]) in set(shared_colors)
    ]
    return [
        {
            "event_description": "both spinners show the same color",
            "favorable_pairs": list(favorable_pairs),
            "supporting_sector_ids": list(supporting),
        }
    ]


def _build_dataset(
    *,
    public_query_id: str,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    builders = {
        "pair_both_target_color_probability": _both_target_color_candidates,
        "pair_at_least_one_target_color_probability": _at_least_one_target_color_candidates,
        "pair_same_color_probability": _same_color_candidates,
    }
    if str(public_query_id) not in builders:
        raise ValueError(f"unsupported spinner probability query_id: {public_query_id}")

    def _event_builder(**kwargs: Any) -> Mapping[str, Any]:
        candidates = builders[str(public_query_id)](**kwargs)
        return select_event_candidate(
            candidates,
            rng=kwargs["rng"],
            favorable_key="favorable_pairs",
            total_outcome_count=len(kwargs["sectors_a"]) * len(kwargs["sectors_b"]),
        )

    return build_pair_spinner_dataset(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        rng_namespace=f"{TASK_ID}.dataset",
        event_builder=_event_builder,
    )


@register_task
class SymbolicSpinnerPairColorEventProbabilityTask:
    """Compute a product-space probability from two independent spinners."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison', 'formula_evaluation')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Own the two-spinner color event query, then bind keyed panel annotation."""

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
        annotation_gt, projected_annotation, witness_symbolic, annotation_source, role_item_ids = pair_panel_bbox_map(
            bundle.rendered_scene.item_bbox_map,
            role_item_ids={"spinner_a": "spinner_a_panel", "spinner_b": "spinner_b_panel"},
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
            annotation_role_item_ids=role_item_ids,
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
    "SymbolicSpinnerPairColorEventProbabilityTask",
]
