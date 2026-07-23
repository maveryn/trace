"""Identify a sister leaf in a rooted phylogeny tree."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ...registry import register_task
from ...shared.config_defaults import group_default
from ._lifecycle import (
    BoundPhylogenyResult,
    DEFAULTS,
    SingleTreeCase,
    leaf_node_id,
    run_single_tree_objective,
    scene_default_sections,
)
from .shared.annotations import projected_keyed_phylogeny_annotation
from .shared.prompts import PROMPT_BUNDLE_ID as PHYLOGENY_PROMPT_BUNDLE_ID
from .shared.sampling import sample_phylogeny_with_cherry


TASK_ID = "task_graph__phylogeny_tree__sister_leaf_label"
SISTER_TASK_ID = TASK_ID
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "sister_leaf_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
SAMPLING_NAMESPACE = "graph.phylogeny_tree.sister_leaf_label"
OBJECT_DESCRIPTION = "a rooted phylogeny cladogram with labeled taxa"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = scene_default_sections(TASK_ID)
PROMPT_BUNDLE_ID = str(group_default(_PROMPT_DEFAULTS, "bundle_id", PHYLOGENY_PROMPT_BUNDLE_ID))


def _prepare_sister_case(instance_seed: int, task_params: Mapping[str, Any], max_attempts: int) -> SingleTreeCase:
    """Sample a tree containing a two-leaf cherry for the queried sister relation."""

    sample, cherry = sample_phylogeny_with_cherry(
        int(instance_seed),
        leaf_count_min=int(group_default(_GEN_DEFAULTS, "leaf_count_min", DEFAULTS.leaf_count_min)),
        leaf_count_max=int(group_default(_GEN_DEFAULTS, "leaf_count_max", DEFAULTS.leaf_count_max)),
        max_attempts=max(40, int(max_attempts)),
    )
    target_leaf, sister_leaf, shared_parent_id = tuple(str(value) for value in cherry)
    return SingleTreeCase(
        sample=sample,
        semantic_payload={
            "target_leaf": target_leaf,
            "sister_leaf": sister_leaf,
            "shared_parent_id": str(shared_parent_id),
        },
    )


def _bind_sister_result(case: SingleTreeCase, rendered: Any, selected_query: str) -> BoundPhylogenyResult:
    """Bind the queried leaf, sister leaf, shared parent, and answer label."""

    payload = dict(case.semantic_payload or {})
    target_leaf = str(payload["target_leaf"])
    sister_leaf = str(payload["sister_leaf"])
    shared_parent_id = str(payload["shared_parent_id"])
    role_to_node = {
        "target_leaf": leaf_node_id(case.sample, target_leaf),
        "sister_leaf": leaf_node_id(case.sample, sister_leaf),
        "shared_parent": shared_parent_id,
    }
    annotation_projection = projected_keyed_phylogeny_annotation(rendered.rendered_scene, role_to_node_id=role_to_node)
    answer_point = [int(value) for value in annotation_projection["point_map"]["sister_leaf"]]
    return BoundPhylogenyResult(
        answer_type="string",
        answer_value=sister_leaf,
        annotation_type="point",
        annotation_value=list(answer_point),
        prompt_slots={"query_label": target_leaf},
        trace_params={},
        scene_relations={},
        execution_trace={
            "query_leaf_label": target_leaf,
            "answer": sister_leaf,
            "sister_leaf_label": sister_leaf,
            "shared_parent_id": shared_parent_id,
            "annotation_role_to_node_id": dict(role_to_node),
        },
        witness_symbolic={
            "type": "phylogeny_sister_leaf",
            "query_leaf_label": target_leaf,
            "sister_leaf_label": sister_leaf,
            "shared_parent_id": shared_parent_id,
        },
        projected_annotation={
            "type": "point",
            "point": list(answer_point),
            "pixel_point": list(answer_point),
            "point_map": dict(annotation_projection["point_map"]),
            "pixel_point_map": dict(annotation_projection["pixel_point_map"]),
        },
    )


@register_task
class GraphPhylogenyTreeSisterLeafLabelTask:
    """Return the leaf label sharing an immediate parent with a queried leaf."""

    task_id = TASK_ID
    reasoning_operations = ('topology',)
    domain = "graph"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int):
        """Sample a cherry pair locally, then use common phylogeny rendering/output."""

        case_factory = _prepare_sister_case
        result_binder = _bind_sister_result
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


__all__ = ["GraphPhylogenyTreeSisterLeafLabelTask", "SISTER_TASK_ID", "TASK_ID"]
