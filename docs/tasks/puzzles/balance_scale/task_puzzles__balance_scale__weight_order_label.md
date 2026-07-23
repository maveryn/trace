# `task_puzzles__balance_scale__weight_order_label`

## Program Contract

Program: `select_option(balance_comparison_grammar, target=lightest_to_heaviest_order, unknowns=A|B|C, panels=3); scene=balance_scale; scope=weight_order_label`

Candidate set: the visible balance-scale panels, object symbols, object counts, side relations, and query markers inside the `weight_order_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `balance_comparison_grammar`, `lightest_to_heaviest_order`, `unknowns`, `A`, `B`, `C`, `panels`, `balance_scale`, `weight_order_label`.
Operation: evaluate `select_option` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; scalar bbox marks the selected order option.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`, `matching`

## 2) Scene + task contract
1. Entities/relations: Three pan-scale comparison panels over three unknown object labels, using direct-offset, shared-object-context, or aggregate comparison expressions, plus a query row with four visual order options.
2. Supported `query_id` values: `single`
3. `answer_gt.type`: `string`
4. Annotation schema: `bbox`
5. Alternate annotation forms: none
6. Annotation witness policy: scalar bbox marks the selected order option.
7. Overlap/touch policy: option bbox covers the selected option card, including its option label and order text.

## 3) Prompt contract
1. `prompt_bundle_id`: `puzzles_balance_scale_v1`
2. `scene_key`: `balance_scale`
3. `task_key`: `balance_scale_query`
4. Optional query-id prompt mapping: public `single` uses prompt query key `weight_order_label`.
5. Required slots:
   - answer output: `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `annotation_hint`, `answer_hint`, `json_example`
6. JSON example validity rule: the example must use one selected-option bbox and one option-label answer.
7. Variant counts: 5 scene templates, 5 query templates, and 5 output templates per output mode.
8. Output modes: `answer_only`, `answer_and_annotation`

## 4) Determinism + constraints
1. Seed namespaces used: task-local object order/comparison namespaces plus shared balance scene/style/font/unit-size namespaces.
2. Unique-answer policy: generated comparison panels must force a unique lightest-to-heaviest order over the configured weight support.
3. Reject/resample conditions: identical pan expressions, redundant same numeric tokens on both sides, any panel count other than three, ambiguous order, non-four-option query row, or missing selected-option bbox raise and retry within `max_attempts`.
4. No-auto-relaxation guarantee: semantic constraints are not relaxed; generation retries rather than accepting ambiguous comparisons.

## 5) Tests
1. Determinism test: `tests/test_puzzles_balance_scale_tasks.py::test_weight_order_task_is_deterministic`
2. Answer/annotation consistency test: `tests/test_puzzles_balance_scale_tasks.py::test_weight_order_task_emits_public_contract`
3. Prompt metadata/placeholder test: covered by source-layout checks and prompt-system tests.
4. Constraint-specific tests: `tests/test_puzzles_balance_scale_tasks.py::test_weight_order_comparisons_imply_one_option`
