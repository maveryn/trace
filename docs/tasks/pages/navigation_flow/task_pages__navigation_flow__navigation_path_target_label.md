# `task_pages__navigation_flow__navigation_path_target_label`

## Identity
1. Domain: `pages`
2. Scene id: `navigation_flow`
3. Source scene: `navigation_flow`
4. Task id: `task_pages__navigation_flow__navigation_path_target_label`

## Contract
1. Objective: identify the candidate control reached by a visible navigation path.
2. Public task contract: `navigation_path_target_label`
3. Supported `query_id` values: `menu_path_target_label`, `sidebar_tree_target_label`, `ribbon_group_command_label`
4. Answer type: `option_letter`
5. Annotation schema: `bbox`
6. Annotation witness: the selected target command/item/control bounding box.
7. Query argument axes: navigation surface branch, target path, target label, menu/ribbon counts, scene variant, and information-scene treatment.

## Program Contract
- `lookup_candidate_label(surface={menu_path,sidebar_tree,ribbon_group}, path); output=option_letter; annotation=bbox(target_control); scene=navigation_flow; scope=one desktop application navigation screen`

## Reasoning Operations

Families: `topology`

## Prompt + Trace
1. Prompt bundle: `pages_navigation_flow_v1`
2. Scene key: `navigation_flow`
3. Task key: `navigation_path_query`
4. Prompt query keys: `menu_path_target_label`, `sidebar_tree_target_label`, and `ribbon_group_command_label`
5. Runtime `query_id` is the semantic navigation-surface branch; the selected surface still records to trace metadata.
6. Public annotation is the selected target control bbox. Path support boxes for menu roots, sidebar groups, and ribbon groups remain in trace metadata as `path_support_bbox_map`.
7. Trace records all controls, support boxes, candidate labels, target path, final geometry, sampled scene/style metadata, and prompt metadata.
8. Generation is deterministic from `instance_seed`; answer and annotation come from the finalized render metadata.
