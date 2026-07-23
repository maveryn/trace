# `task_pages__concept_map__ordered_child_label`

## Identity
1. Domain: `pages`
2. Scene id: `concept_map`
3. Source scene: `concept_map`
4. Task id: `task_pages__concept_map__ordered_child_label`

## Contract
1. Objective: read the visible child-item label at a requested ordinal position under a named concept-map branch.
2. Public task contract: `ordered_child_label`
3. Supported `query_id` values: `single`
4. Answer type: `string`
5. Annotation schema: `bbox`
6. Annotation witness: selected child-item-node box.
7. Query argument axes: target branch label, target ordinal rank, reading order, context topic, branch count, child-count support, non-radial layout variant, style variant, and node shape profile.

## Program Contract
- `concept_map_ordered_child_label(branch_label, rank_ordinal, reading_order); output=child_label_string; annotation=bbox(answer_child); scene=concept_map; scope=one concept-map diagram`

## Reasoning Operations

Families: `ranking`, `topology`

## Prompt + Trace
1. Prompt bundle: `pages_concept_map_v1`
2. Scene key: `concept_map_diagram`
3. Task key: `concept_map_lookup_query`
4. Prompt query key: `nth_child_label`
5. Trace records concept branches, child nodes, marker icons, rendered node boxes, sampled context/layout/style metadata, and selected target metadata.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized concept-map render metadata.
7. This task intentionally uses `left_right_map` and `clustered_map` layouts only; dense ordered child lists are not sampled in `radial_mind_map` because radial branch fans make ordinal reading visually cluttered.
