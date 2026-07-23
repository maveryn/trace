# `task_puzzles__balance_scale__query_side_relation_label`

## Program Contract

Program: `select_option(balance_query_relation, target=left|right|balanced|not_determined, unknowns=A|B|C, panels=3); scene=balance_scale; scope=query_side_relation_label`

Candidate set: the visible balance-scale panels, object symbols, object counts, side relations, and query markers inside the `query_side_relation_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `balance_query_relation`, `left`, `right`, `balanced`, `not_determined`, `unknowns`, `A`, `B`, `C`, `panels`, `balance_scale`, `query_side_relation_label`.
Operation: evaluate `select_option` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; scalar bbox marks the selected relation option.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`, `matching`

## 2) Scene + task contract
1. Entities/relations: Three balanced pan-scale reference panels over three unknown object labels, plus a query row comparing two symbolic pan expressions and four visual relation options. The internal `not_determined` relation is displayed to users as `Cannot determine`.
2. Supported `query_id` values: `single`
3. `answer_gt.type`: `string`
4. Annotation schema: `bbox`
5. Alternate annotation forms: none
6. Annotation witness policy: scalar bbox marks the selected relation option.
7. Overlap/touch policy: option bbox covers the selected option card, including its option label and relation text.

## 3) Prompt contract
1. `prompt_bundle_id`: `puzzles_balance_scale_v1`
2. `scene_key`: `balance_scale`
3. `task_key`: `balance_scale_query`
4. Optional query-id prompt mapping: public `single` uses prompt query key `query_side_relation_label`.
5. Required slots:
   - answer output: `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `annotation_hint`, `answer_hint`, `json_example`
6. JSON example validity rule: the example must use one selected-option bbox and one option-label answer.
7. Variant counts: 5 scene templates, 5 query templates, and 5 output templates per output mode.
8. Output modes: `answer_only`, `answer_and_annotation`

## 4) Determinism + constraints
1. Seed namespaces used: task-local relation construction/object-order namespaces plus shared balance scene/style/font/unit-size namespaces.
2. Unique-answer policy: left, right, and balanced targets must be uniquely implied by the reference scales over the configured weight support; `not_determined` targets, displayed as `Cannot determine`, must allow more than one relation over that support.
3. Answer-balance policy: target relation is sampled uniformly from left, right, balanced, and `not_determined` using the task RNG; the seed only makes that equal-weight draw reproducible.
4. Reject/resample conditions: identical pan expressions, missing query relation bbox, non-four-option query row, determined targets with ambiguous relation, or `not_determined` targets with only one possible relation raise and retry within `max_attempts`.
5. No-auto-relaxation guarantee: semantic constraints are not relaxed; generation retries rather than accepting ambiguous determined comparisons or determined unanswerable cases.

## 5) Tests
1. Determinism test: `tests/test_puzzles_balance_scale_tasks.py::test_query_side_relation_task_is_deterministic`
2. Answer/annotation consistency test: `tests/test_puzzles_balance_scale_tasks.py::test_query_side_relation_task_emits_public_contract`
3. Prompt metadata/placeholder test: covered by source-layout checks and prompt-system tests.
4. Constraint-specific tests: `tests/test_puzzles_balance_scale_tasks.py::test_query_side_relation_targets_are_valid_and_balanced`
