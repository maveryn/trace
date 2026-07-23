# `task_pages__web_action__guide_code_target_count`

## Identity
1. Domain: `pages`
2. Scene id: `web_action`
3. Source scene: `web_action`
4. Task id: `task_pages__web_action__guide_code_target_count`

## Contract
1. Objective: count candidate controls in the web-action page that use a requested guide code.
2. Public task contract: `guide_code_target_count`
3. Supported `query_id` values: `click_guide_code_target_count`, `type_field_guide_code_target_count`, `select_option_guide_code_target_count`
4. Answer type: `integer`
5. Annotation schema: `bbox_set`
6. Annotation witness: boxes covering every matching candidate control and its candidate label marker.
7. Query argument axes: counted control family, requested guide code, web page variant, and style variant.
8. Default answer range: `3..6`, controlled by the number of visible item cards, form sections, or option groups for the selected control family.

## Program Contract
- `web_action_guide_code_target_count(control_family={click_button,input_field,select_option}, guide_code); output=integer; annotation=bbox_set(matching_controls_with_candidate_markers); scene=web_action; scope=one browser-style action-target page`

## Reasoning Operations

Families: `counting`

## Prompt + Trace
1. Prompt bundle: `pages_web_action_v1`
2. Scene key: `web_action`
3. Task key: `web_action_query`
4. Prompt query keys: `click_guide_code_target_count`, `type_field_guide_code_target_count`, `select_option_guide_code_target_count`
5. Runtime `query_id` is the semantic count branch; non-semantic values such as page variant, guide-code assignment, candidate labels, and theme are trace metadata.
6. Trace records the requested guide code, matching control ids, matching annotation boxes, control metadata, sampled scene/style metadata, and prompt metadata.
