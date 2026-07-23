"""Select the option phylogeny with the outlier rooted topology."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ...registry import register_task
from ...shared.config_defaults import group_default
from ...shared.fixed_query import select_task_query_id
from ._lifecycle import (
    DEFAULTS,
    SCENE_ID,
    finalize_phylogeny_result,
    query_spec,
    render_option_trees,
    render_spec,
    resolve_phylogeny_style,
    scene_default_sections,
)
from .shared.prompts import OPTIONS_SCENE_PROMPT_KEY, PROMPT_BUNDLE_ID as PHYLOGENY_PROMPT_BUNDLE_ID
from .shared.prompts import build_phylogeny_prompt_artifacts
from .shared.sampling import sample_topology_outlier_options


TASK_ID = "task_graph__phylogeny_tree__topology_outlier_label"
TOPOLOGY_TASK_ID = TASK_ID
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "topology_outlier_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
SAMPLING_NAMESPACE = "graph.phylogeny_tree.topology_outlier_label"
OBJECT_DESCRIPTION = "four labeled rooted phylogeny cladograms with the same taxon labels"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = scene_default_sections(TASK_ID)
PROMPT_BUNDLE_ID = str(group_default(_PROMPT_DEFAULTS, "bundle_id", PHYLOGENY_PROMPT_BUNDLE_ID))


def _option_records(option_dataset: Dict[str, Any], option_bboxes: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Serialize visible option panels and their topology roles."""

    records = []
    for spec in option_dataset["option_specs"]:
        label = str(spec["option_label"])
        records.append(
            {
                "option_label": label,
                "role": str(spec["role"]),
                "canonical_signature": [list(item) for item in spec["canonical_signature"]],
                "panel_bbox_xyxy": list(option_bboxes[label]),
            }
        )
    return records


@register_task
class GraphPhylogenyTreeTopologyOutlierLabelTask:
    """Choose the single option whose rooted clade structure differs."""

    task_id = TASK_ID
    reasoning_operations = ('topology', 'matching')
    domain = "graph"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int):
        """Sample four option cladograms, bind the outlier option letter and panel box."""

        selected_query, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{SAMPLING_NAMESPACE}.query",
        )
        style = resolve_phylogeny_style(
            instance_seed=int(instance_seed),
            params=task_params,
            gen_defaults=_GEN_DEFAULTS,
            namespace=SAMPLING_NAMESPACE,
        )
        option_dataset = sample_topology_outlier_options(
            int(instance_seed),
            leaf_count_min=int(group_default(_GEN_DEFAULTS, "option_leaf_count_min", DEFAULTS.option_leaf_count_min)),
            leaf_count_max=int(group_default(_GEN_DEFAULTS, "option_leaf_count_max", DEFAULTS.option_leaf_count_max)),
            option_count=int(group_default(_GEN_DEFAULTS, "option_count", DEFAULTS.option_count)),
            max_attempts=max(100, int(max_attempts)),
        )
        rendered = render_option_trees(
            owner_namespace=SAMPLING_NAMESPACE,
            instance_seed=int(instance_seed),
            params=task_params,
            render_defaults=_RENDER_DEFAULTS,
            style=style,
            option_specs=option_dataset["option_specs"],
        )
        answer_label = str(option_dataset["answer_option_label"])
        selected_bbox = [round(float(value), 3) for value in rendered.rendered_scene.option_panel_bboxes[answer_label]]
        prompt_artifacts = build_phylogeny_prompt_artifacts(
            domain=self.domain,
            bundle_id=PROMPT_BUNDLE_ID,
            scene_key=OPTIONS_SCENE_PROMPT_KEY,
            prompt_key=PROMPT_QUERY_KEY,
            dynamic_slots={"object_description": OBJECT_DESCRIPTION},
            instance_seed=int(instance_seed),
        )
        records = _option_records(option_dataset, rendered.rendered_scene.option_panel_bboxes)
        trace_params = {
            "query_id_probabilities": dict(query_probabilities),
            "objective": PROMPT_QUERY_KEY,
            "scene_variant": str(style.scene_variant),
            "scene_variant_probabilities": dict(style.scene_variant_probabilities),
            "option_count": int(len(records)),
            "answer_option_label": answer_label,
            "leaf_count": int(option_dataset["leaf_count"]),
            "node_color_name": str(style.node_color_name),
            "node_color_name_probabilities": dict(style.node_color_name_probabilities),
        }
        trace_payload = {
            "scene_ir": {
                "task_id": TASK_ID,
                "scene_id": SCENE_ID,
                "scene_kind": "phylogeny_topology_options",
                "entities": [
                    {
                        "entity_id": f"option_{record['option_label']}",
                        "entity_kind": "phylogeny_option_panel",
                        **record,
                    }
                    for record in records
                ],
                "relations": {
                    "option_count": int(len(records)),
                    "leaf_count": int(option_dataset["leaf_count"]),
                    "leaf_labels": list(option_dataset["base_sample"].leaf_labels),
                    "base_canonical_signature": [list(item) for item in option_dataset["base_sample"].canonical_signature],
                    "outlier_canonical_signature": [list(item) for item in option_dataset["outlier_sample"].canonical_signature],
                },
                "frames": {
                    "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                    "panels": dict(rendered.rendered_scene.panel_geometry),
                },
            },
            "query_spec": query_spec(
                owner_id=TASK_ID,
                public_query_id=str(selected_query),
                prompt_artifacts=prompt_artifacts,
                params=trace_params,
                prompt_bundle_id=PROMPT_BUNDLE_ID,
            ),
            "render_spec": render_spec(rendered, style=style),
            "render_map": {"image_id": "img0", "anchors": {}},
            "execution_trace": {
                "task_id": TASK_ID,
                "scene_id": SCENE_ID,
                "query_id": str(selected_query),
                "objective": PROMPT_QUERY_KEY,
                "answer": answer_label,
                "answer_option_label": answer_label,
                "correct_option_index": int(option_dataset["correct_option_index"]),
                "option_records": list(records),
            },
            "witness_symbolic": {
                "type": "phylogeny_topology_outlier_option",
                "answer_option_label": answer_label,
                "rule": "same rooted clades, child order and layout ignored",
            },
            "projected_annotation": {
                "type": "bbox",
                "bbox": list(selected_bbox),
                "pixel_bbox": list(selected_bbox),
            },
        }
        return finalize_phylogeny_result(
            prompt_artifacts=prompt_artifacts,
            answer_type="string",
            answer_value=answer_label,
            annotation_type="bbox",
            annotation_value=list(selected_bbox),
            image=rendered.image,
            trace_payload=trace_payload,
            public_query_id=str(selected_query),
        )


__all__ = ["GraphPhylogenyTreeTopologyOutlierLabelTask", "TOPOLOGY_TASK_ID", "TASK_ID"]
