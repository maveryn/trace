"""Count leaves under the MRCA of two queried phylogeny leaves."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ...registry import register_task
from ...shared.config_defaults import group_default
from ._lifecycle import (
    BoundPhylogenyResult,
    DEFAULTS,
    SingleTreeCase,
    integer_support,
    leaf_node_id,
    resolve_integer_axis,
    run_single_tree_objective,
    scene_default_sections,
)
from .shared.algorithms import descendant_leaf_labels
from .shared.annotations import projected_leaf_point_annotation
from .shared.prompts import PROMPT_BUNDLE_ID as PHYLOGENY_PROMPT_BUNDLE_ID
from .shared.sampling import sample_phylogeny_with_mrca_size


TASK_ID = "task_graph__phylogeny_tree__mrca_clade_membership_count"
MRCA_TASK_ID = TASK_ID
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "mrca_leaf_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
SAMPLING_NAMESPACE = "graph.phylogeny_tree.mrca_clade_membership_count"
OBJECT_DESCRIPTION = "a rooted phylogeny cladogram with labeled taxa"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = scene_default_sections(TASK_ID)
PROMPT_BUNDLE_ID = str(group_default(_PROMPT_DEFAULTS, "bundle_id", PHYLOGENY_PROMPT_BUNDLE_ID))


@register_task
class GraphPhylogenyTreeMrcaCladeMembershipCountTask:
    """Count terminal taxa under the most recent common ancestor of two leaves."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'topology')
    domain = "graph"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int):
        """Sample the MRCA case locally, then use common phylogeny rendering/output."""

        case_factory = _prepare_mrca_case
        result_binder = _bind_mrca_result
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


def _target_mrca_leaf_count(instance_seed: int, params: Mapping[str, Any]) -> Tuple[int, Dict[str, float]]:
    """Resolve the answer support for the queried MRCA clade size."""

    support = integer_support(
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        key="target_mrca_leaf_count",
        fallback_min=DEFAULTS.target_mrca_leaf_count_min,
        fallback_max=DEFAULTS.target_mrca_leaf_count_max,
    )
    return resolve_integer_axis(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.target_mrca_leaf_count",
        support=support,
        explicit_key="target_mrca_leaf_count",
    )


def _prepare_mrca_case(instance_seed: int, task_params: Mapping[str, Any], max_attempts: int) -> SingleTreeCase:
    """Sample a tree whose queried leaf pair has a controlled MRCA clade size."""

    target_count, target_probabilities = _target_mrca_leaf_count(int(instance_seed), task_params)
    sample, mrca_node_id, leaf_pair = sample_phylogeny_with_mrca_size(
        int(instance_seed),
        target_size=int(target_count),
        leaf_count_min=int(group_default(_GEN_DEFAULTS, "leaf_count_min", DEFAULTS.leaf_count_min)),
        leaf_count_max=int(group_default(_GEN_DEFAULTS, "leaf_count_max", DEFAULTS.leaf_count_max)),
        max_attempts=max(60, int(max_attempts)),
    )
    leaf_a, leaf_b = tuple(str(label) for label in leaf_pair)
    return SingleTreeCase(
        sample=sample,
        marked_node_id=None,
        trace_params={
            "target_mrca_leaf_count": int(target_count),
            "target_count_probabilities": dict(target_probabilities),
        },
        semantic_payload={"mrca_node_id": str(mrca_node_id), "leaf_a": leaf_a, "leaf_b": leaf_b},
    )


def _bind_mrca_result(case: SingleTreeCase, rendered: Any, selected_query: str) -> BoundPhylogenyResult:
    """Bind the two query leaves, their MRCA, and the descendant-leaf count."""

    payload = dict(case.semantic_payload or {})
    leaf_a = str(payload["leaf_a"])
    leaf_b = str(payload["leaf_b"])
    mrca_node_id = str(payload["mrca_node_id"])
    role_to_node = {
        "query_leaf_1": leaf_node_id(case.sample, leaf_a),
        "query_leaf_2": leaf_node_id(case.sample, leaf_b),
        "mrca": mrca_node_id,
    }
    descendant_labels = descendant_leaf_labels(case.sample, mrca_node_id)
    annotation_projection = projected_leaf_point_annotation(rendered.rendered_scene, descendant_labels)
    annotation_points = [[int(point[0]), int(point[1])] for point in annotation_projection["pixel_point_set"]]
    answer_value = int(len(descendant_labels))
    return BoundPhylogenyResult(
        answer_type="integer",
        answer_value=int(answer_value),
        annotation_type="point_set",
        annotation_value=list(annotation_points),
        prompt_slots={"query_label_a": leaf_a, "query_label_b": leaf_b},
        trace_params={},
        scene_relations={
            "mrca_node_id": mrca_node_id,
            "mrca_descendant_leaf_labels": list(descendant_labels),
        },
        execution_trace={
            "query_leaf_labels": [leaf_a, leaf_b],
            "answer": int(answer_value),
            "mrca_node_id": mrca_node_id,
            "mrca_descendant_leaf_labels": list(descendant_labels),
            "annotation_role_to_node_id": dict(role_to_node),
        },
        witness_symbolic={
            "type": "phylogeny_mrca_leaf_count",
            "query_leaf_labels": [leaf_a, leaf_b],
            "mrca_node_id": mrca_node_id,
            "mrca_descendant_leaf_labels": list(descendant_labels),
        },
        projected_annotation={
            "type": "point_set",
            "point_set": list(annotation_points),
            "pixel_point_set": list(annotation_points),
            "leaf_label_bbox_map": dict(annotation_projection["leaf_label_bbox_map"]),
        },
    )


__all__ = ["GraphPhylogenyTreeMrcaCladeMembershipCountTask", "MRCA_TASK_ID", "TASK_ID"]
