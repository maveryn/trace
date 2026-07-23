"""Count descendant leaves in a marked phylogeny clade."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ...registry import register_task
from ...shared.config_defaults import group_default
from ._lifecycle import (
    BoundPhylogenyResult,
    DEFAULTS,
    SCENE_ID,
    SingleTreeCase,
    integer_support,
    resolve_integer_axis,
    run_single_tree_objective,
    scene_default_sections,
)
from .shared.algorithms import descendant_leaf_labels
from .shared.annotations import projected_leaf_point_annotation
from .shared.prompts import PROMPT_BUNDLE_ID as PHYLOGENY_PROMPT_BUNDLE_ID
from .shared.rendering import phylogeny_scene_entities
from .shared.sampling import sample_phylogeny_with_clade_size


TASK_ID = "task_graph__phylogeny_tree__clade_leaf_count"
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "marked_clade_leaf_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
SAMPLING_NAMESPACE = "graph.phylogeny_tree.clade_leaf_count"
OBJECT_DESCRIPTION = "a rooted phylogeny cladogram with labeled taxa and one marked clade"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = scene_default_sections(TASK_ID)
PROMPT_BUNDLE_ID = str(group_default(_PROMPT_DEFAULTS, "bundle_id", PHYLOGENY_PROMPT_BUNDLE_ID))


def _target_clade_count(instance_seed: int, params: Mapping[str, Any]) -> Tuple[int, Dict[str, float]]:
    """Resolve the requested descendant-leaf count for the marked clade."""

    support = integer_support(
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        key="target_clade_leaf_count",
        fallback_min=DEFAULTS.target_clade_leaf_count_min,
        fallback_max=DEFAULTS.target_clade_leaf_count_max,
    )
    return resolve_integer_axis(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.target_clade_leaf_count",
        support=support,
        explicit_key="target_clade_leaf_count",
    )


def _prepare_clade_case(instance_seed: int, task_params: Mapping[str, Any], max_attempts: int) -> SingleTreeCase:
    """Sample a tree with one marked clade at the task-owned answer size."""

    target_count, target_probabilities = _target_clade_count(int(instance_seed), task_params)
    sample, target_node_id = sample_phylogeny_with_clade_size(
        int(instance_seed),
        target_size=int(target_count),
        leaf_count_min=int(group_default(_GEN_DEFAULTS, "leaf_count_min", DEFAULTS.leaf_count_min)),
        leaf_count_max=int(group_default(_GEN_DEFAULTS, "leaf_count_max", DEFAULTS.leaf_count_max)),
        max_attempts=max(40, int(max_attempts)),
    )
    return SingleTreeCase(
        sample=sample,
        marked_node_id=str(target_node_id),
        trace_params={
            "target_clade_leaf_count": int(target_count),
            "target_count_probabilities": dict(target_probabilities),
        },
        semantic_payload={"target_node_id": str(target_node_id)},
    )


def _bind_clade_result(case: SingleTreeCase, rendered: Any, selected_query: str) -> BoundPhylogenyResult:
    """Bind descendant leaf witnesses and the integer clade-size answer."""

    target_node_id = str((case.semantic_payload or {})["target_node_id"])
    target_leaf_labels = descendant_leaf_labels(case.sample, target_node_id)
    annotation_projection = projected_leaf_point_annotation(rendered.rendered_scene, target_leaf_labels)
    annotation_points = [[int(point[0]), int(point[1])] for point in annotation_projection["pixel_point_set"]]
    counted = set(str(label) for label in target_leaf_labels)
    entities = []
    for entity in phylogeny_scene_entities(case.sample, rendered.rendered_scene):
        if entity["entity_kind"] == "phylogeny_leaf":
            entity = dict(entity)
            entity["is_counted"] = str(entity["leaf_label"]) in counted
        entities.append(entity)
    return BoundPhylogenyResult(
        answer_type="integer",
        answer_value=int(len(target_leaf_labels)),
        annotation_type="point_set",
        annotation_value=list(annotation_points),
        prompt_slots={},
        trace_params={},
        scene_relations={
            "marked_clade_node_id": target_node_id,
            "marked_clade_leaf_labels": list(target_leaf_labels),
        },
        execution_trace={
            "answer": int(len(target_leaf_labels)),
            "marked_clade_node_id": target_node_id,
            "target_leaf_labels": list(target_leaf_labels),
            "leaf_count": int(case.sample.leaf_count),
        },
        witness_symbolic={
            "type": "phylogeny_leaf_label_set",
            "labels": list(target_leaf_labels),
            "marked_clade_node_id": target_node_id,
        },
        projected_annotation={
            "type": "point_set",
            "point_set": list(annotation_points),
            "pixel_point_set": list(annotation_points),
            "leaf_label_bbox_map": dict(annotation_projection["leaf_label_bbox_map"]),
        },
        entities=entities,
    )


@register_task
class GraphPhylogenyTreeCladeLeafCountTask:
    """Count all terminal taxa descending from the visually marked clade."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'topology')
    domain = "graph"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int):
        """Sample the marked clade locally, then use common phylogeny rendering/output."""

        case_factory = _prepare_clade_case
        result_binder = _bind_clade_result
        return run_single_tree_objective(
            owner_id=TASK_ID,
            domain=self.domain,
            prompt_key=PROMPT_QUERY_KEY,
            prompt_bundle_id=PROMPT_BUNDLE_ID,
            object_description=OBJECT_DESCRIPTION,
            sampling_namespace=SAMPLING_NAMESPACE,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_case=case_factory,
            bind_rendered=result_binder,
        )


__all__ = ["GraphPhylogenyTreeCladeLeafCountTask", "TASK_ID"]
