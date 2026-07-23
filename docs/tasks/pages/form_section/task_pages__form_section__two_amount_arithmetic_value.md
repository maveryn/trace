# `task_pages__form_section__two_amount_arithmetic_value`

## Identity
1. Domain: `pages`
2. Scene id: `form_section`
3. Source scene: `form_section`
4. Task id: `task_pages__form_section__two_amount_arithmetic_value`

## Contract
1. Objective: compute either the sum or positive difference of two visible currency amounts in one named document section.
2. Public task contract: `two_amount_arithmetic_value`
3. Supported `query_id` values: `sum_two_amounts_in_section_value`, `difference_two_amounts_in_section_value`
4. Answer type: `string`
5. Annotation schema: `bbox_map`
6. Annotation witness: full operand field boxes, including label and value, keyed as `first_operand` and `second_operand`.
7. Query argument axes: arithmetic operator branch, scene variant, section label, first operand label, and second operand label.

## Program Contract
- `section_arithmetic_value(section=resolved_section, operands=[first_amount, second_amount], operators=[add|subtract]); output=currency_string; annotation=bbox_map(first_operand_field, second_operand_field); scene=form_section; scope=one structured document page`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt + Trace
1. Prompt bundle: `pages_form_section_v1`
2. Scene key: `structured_document_sections`
3. Task key: `section_expression_query`
4. Prompt query keys: `sum_two_amounts_in_section_value`, `difference_two_amounts_in_section_value`
5. Trace records the selected query id, scene variant, target section, operand field ids/labels/values, operator sequence, result cents/value, rendered field boxes and value text boxes, and sampled visual metadata.
6. Generation is deterministic from `instance_seed`; answer comes from the finalized rendered operand values and annotation comes from the finalized rendered operand field boxes.
