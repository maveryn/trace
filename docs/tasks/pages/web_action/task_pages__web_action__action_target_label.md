# `task_pages__web_action__action_target_label`

## Identity
1. Domain: `pages`
2. Scene id: `web_action`
3. Source scene: `web_action`
4. Task id: `task_pages__web_action__action_target_label`

## Contract
1. Objective: identify the candidate-labeled web control matching the visible instruction, guide code, and page context.
2. Public task contract: `action_target_label`
3. Supported `query_id` values: `click_target_label`, `type_field_label`, `select_option_label`
4. Answer type: `option_letter`
5. Annotation schema: `bbox`
6. Annotation witness: one box covering the selected target control and its candidate label marker.
7. Query argument axes: target control family, target item/section/group context, guide code, target candidate label, web page variant, and style variant.

## Program Contract
- `web_action_target_control_label(control_family={click_button,input_field,select_option}, instruction_guide_code, page_context); output=option_letter; annotation=bbox(target_control_with_candidate_marker); scene=web_action; scope=one browser-style action-target page`

## Reasoning Operations

Families: `matching`

## Prompt + Trace
1. Prompt bundle: `pages_web_action_v1`
2. Scene key: `web_action`
3. Task key: `web_action_query`
4. Prompt query keys: `click_target_label`, `type_field_label`, `select_option_label`
5. Runtime `query_id` is the semantic control-family branch; non-semantic values such as target label, page variant, guide code, and theme are trace metadata.
6. Trace records target control metadata, guide entries, context support, target annotation bbox, sampled scene/style metadata, and prompt metadata.
