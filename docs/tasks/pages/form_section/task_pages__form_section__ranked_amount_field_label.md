# `task_pages__form_section__ranked_amount_field_label`

## Identity
1. Domain: `pages`
2. Scene id: `form_section`
3. Source scene: `form_section`
4. Task id: `task_pages__form_section__ranked_amount_field_label`

## Contract
1. Objective: identify the field label attached to a ranked currency amount in one named document section.
2. Public task contract: `ranked_amount_field_label`
3. Supported `query_id` values: `second_highest_amount_field_label`, `second_lowest_amount_field_label`
4. Answer type: `string`
5. Annotation schema: `bbox`
6. Annotation witness: the full selected field row box, including its label and value.
7. Query argument axes: scene variant, section label, rank direction, and rank position.

## Program Contract
- `section_ranked_amount_label(section=resolved_section, rank_from=highest|lowest, rank_position=2); output=selected_field_label_string; annotation=bbox(selected_field_row); scene=form_section; scope=one structured document page`

## Reasoning Operations

Families: `ranking`

## Prompt + Trace
1. Prompt bundle: `pages_form_section_v1`
2. Scene key: `structured_document_sections`
3. Task key: `section_expression_query`
4. Prompt query key: `ranked_amount_field_label`
5. Trace records the scene variant, target section, candidate currency field ids/labels/values, selected rank direction and position, selected field id/label/value, rendered field boxes, and sampled visual metadata.
6. Generation is deterministic from `instance_seed`; answer comes from the selected ranked field label and annotation comes from the finalized rendered selected field row box.
