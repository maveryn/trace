# `task_pages__paired_forms__shortfall_minus_overage_value`

## Identity
1. Domain: `pages`
2. Scene id: `paired_forms`
3. Source scene: `paired_forms`
4. Task id: `task_pages__paired_forms__shortfall_minus_overage_value`

## Contract
1. Objective: compute signed shortfall value minus overage value across matched purchase-order and receiving-slip rows.
2. Public task contract: `shortfall_minus_overage_value`
3. Supported `query_id` values: `single`
4. Answer type: `integer`
5. Annotation schema: `bbox_set`
6. Annotation witness: full receiving-slip rows whose received quantity differs from the matching purchase-order quantity.
7. Query argument axes: item count, mismatch count, row values, and scene variant.

## Program Contract
- `aggregate_matched_row_delta(match_key=item_code, predicate=ordered_quantity!=received_quantity, term=(ordered_quantity-received_quantity)*unit_value, aggregate=sum); output=integer; annotation=bbox_set(receiving_mismatch_rows); scene=paired_forms; scope=two side-by-side matched business forms`

## Reasoning Operations

Families: `aggregation`, `formula_evaluation`, `matching`

## Prompt + Trace
1. Prompt bundle: `pages_paired_forms_v1`
2. Scene key: `paired_forms_reconciliation`
3. Task key: `paired_forms_reconciliation_value_query`
4. Prompt query key: `shortfall_minus_overage_value`
5. Runtime `query_id` is `single`; semantic branch identity is recorded as `prompt_query_key` / `source_query_id`.
6. Trace records all purchase-order rows, receiving-slip rows, row/cell boxes, mismatch row ids, answer value, sampled render metadata, and prompt metadata.
7. Generation is deterministic from `instance_seed`; answer and annotation come from the finalized render metadata.
