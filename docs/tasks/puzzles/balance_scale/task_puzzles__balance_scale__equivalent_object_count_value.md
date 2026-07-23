# `task_puzzles__balance_scale__equivalent_object_count_value`

## Program Contract

Program: `solve_count(direct_and_compound_balance_equations, target=source_object_equivalent_repeated_objects, unknowns=A|B|C, panels=3); scene=balance_scale; scope=equivalent_object_count_value`

Candidate set: the visible balance-scale panels, object symbols, object counts, side relations, and query markers inside the `equivalent_object_count_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `direct_and_compound_balance_equations`, `source_object_equivalent_repeated_objects`, `unknowns`, `A`, `B`, `C`, `panels`, `balance_scale`, `equivalent_object_count_value`.
Operation: evaluate `solve_count` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the scalar bbox marks the question-mark count box in the query row.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `formula_evaluation`, `matching`

## 2) Scene + task contract
1. Entities/relations: Three balanced pan-scale panels with three labeled unknown object tokens, numbered weights, direct single-object value panels for the source and repeated objects, at least one compound/context panel, and a query row asking how many repeated objects equal one source object.
2. Supported `query_id` values: `single`
3. `answer_gt.type`: `integer`
4. Annotation schema: `bbox`
5. Alternate annotation forms: none
6. Annotation witness policy: the scalar bbox marks the question-mark count box in the query row.
7. Overlap/touch policy: annotation must be the rendered question-mark count box, not the surrounding query row, object tokens, panel, or scale bbox.

## 3) Prompt contract
1. `prompt_bundle_id`: `puzzles_balance_scale_v1`
2. `scene_key`: `balance_scale`
3. `task_key`: `balance_scale_query`
4. Optional query-id prompt mapping: `equivalent_object_count_value` is stored as `prompt_query_key`; the public `query_id` is `single`.
5. Required slots:
   - answer output: `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `annotation_hint`, `answer_hint`, `json_example`
6. JSON example validity rule: the example must use one bbox and an integer answer valid for the task schema.
7. Variant counts: 5 scene templates, 5 query templates, and 5 output templates per output mode.
8. Output modes: `answer_only`, `answer_and_annotation`

## 4) Determinism + constraints
1. Seed namespaces used: task-local label/object/equation namespaces plus shared balance scene/style/font/unit-size namespaces.
2. Unique-answer policy: the three equations must imply exactly one repeated-object count over the configured count support.
3. Reject/resample conditions: any panel count other than three, missing source/repeated direct-value panels, unbalanced generated panels, equivalent-count ambiguity, missing annotation bbox, or unsupported answer values raise and retry within `max_attempts`.
4. No-auto-relaxation guarantee: semantic constraints are not relaxed; generation retries rather than accepting ambiguous equations.

## 5) Tests
1. Determinism test: `tests/test_puzzles_balance_scale_tasks.py::test_equivalent_object_count_task_is_deterministic`
2. Answer/annotation consistency test: `tests/test_puzzles_balance_scale_tasks.py::test_equivalent_object_count_task_emits_public_contract`
3. Prompt metadata/placeholder test: covered by source-layout checks and prompt-system tests.
4. Constraint-specific tests: `tests/test_puzzles_balance_scale_tasks.py::test_equivalent_object_count_equations_are_balanced_and_unique`
