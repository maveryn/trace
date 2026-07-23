# `task_pages__concept_map__marked_child_count`

## Identity
1. Domain: `pages`
2. Scene id: `concept_map`
3. Source scene: `concept_map`
4. Task id: `task_pages__concept_map__marked_child_count`

## Contract
1. Objective: count visible child-item nodes under a requested branch that have a requested marker.
2. Public task contract: `marked_child_count`
3. Supported `query_id` values: `single`
4. Answer type: `integer`
5. Annotation schema: `bbox_set`
6. Annotation witness: counted marked child-item-node boxes under the requested branch.
7. Query argument axes: target branch label, marker label, context topic, branch count, child-count support, marked-count support, layout variant, style variant, and node shape profile.

## Program Contract
- `concept_map_marked_child_count(branch_label, marker_label); output=integer_value; annotation=bbox_set(marked_child_item_nodes); scene=concept_map; scope=one concept-map diagram`

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## Prompt + Trace
1. Prompt bundle: `pages_concept_map_v1`
2. Scene key: `concept_map_diagram`
3. Task key: `concept_map_lookup_query`
4. Prompt query key: `marked_child_count`
5. Trace records concept branches, child nodes, marker icons, rendered node boxes, sampled context/layout/style metadata, and selected target metadata.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized concept-map render metadata.
