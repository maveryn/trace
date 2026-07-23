# `task_puzzles__arithmetic_panel__number_wall_value`

## 1) Identity
1. Domain: `puzzles`
2. Scene id: `arithmetic_panel`
3. Task id: `task_puzzles__arithmetic_panel__number_wall_value`
4. Objective contract: `number_wall_value`
5. Supported `query_id` values: `single`
6. Prompt query key: `addition_wall_missing_value`
7. Answer schema: `integer`
8. Annotation schema: `bbox`
9. Program schema: `solve_value(number_wall, rule=addition, target=question_mark_brick); scene=arithmetic_panel; scope=number_wall_value`

## Program Contract

Program: `solve_value(number_wall, rule=addition, target=question_mark_brick); scene=arithmetic_panel; scope=number_wall_value`

Candidate set: the visible arithmetic panels, numeric entries, operators, totals, and marked target cell/node/brick inside the `number_wall_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `number_wall`, `addition`, `question_mark_brick`, `arithmetic_panel`, `number_wall_value`.
Operation: evaluate `solve_value` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the scalar bbox is the pixel box around the single visible question-mark target cell, node, or brick. It is not a one-item set.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## 2) Scene + task contract
1. Entities/relations: A brick-wall layout with visible numeric bricks and one question-mark target brick.
2. Supported `query_id` values: `single`
3. `answer_gt.type`: `integer`
4. Default `annotation_gt.type`: `bbox`
5. Alternate annotation forms: none
6. Annotation witness policy: the scalar bbox is the pixel box around the single visible question-mark target cell, node, or brick. It is not a one-item set.
7. Overlap/touch policy: the target bbox must correspond to the rendered `target` item and must not use the surrounding diagram panel bbox.

## 3) Prompt contract
1. `prompt_bundle_id`: `puzzles_arithmetic_panel_v1`
2. `scene_key`: `arithmetic_panel`
3. `task_key`: `arithmetic_value_query`
4. Optional query-id prompt mapping: public `single` uses prompt query key `addition_wall_missing_value`.
5. Required slots:
   - answer output: `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `annotation_hint`, `answer_hint`, `json_example`
6. JSON example validity rule: the example must use a scalar `[x0, y0, x1, y1]` bbox and an integer answer valid for this task's output schema.
7. Variant counts: 5 scene templates, 5 query templates per prompt query key, and 5 output templates per output mode.
8. Output modes: `answer_only`, `answer_and_annotation`

## 4) Determinism + constraints
1. Seed namespaces used: scene/unit-size/layout/font/panel-style namespaces under `puzzles.arithmetic_panel` plus task-local arithmetic construction namespaces.
2. Unique-answer policy: The visible number-wall pattern determines exactly one target brick value within the branch support.
3. Reject/resample conditions: invalid arithmetic construction, missing render bbox, or a target outside the configured support raises and retries within `max_attempts`.
4. No-auto-relaxation guarantee: semantic constraints are not relaxed; generation retries rather than accepting ambiguous or unsupported targets.

## 5) Tests
1. Determinism test: `tests/test_puzzles_arithmetic_panel_tasks.py::test_arithmetic_panel_task_is_deterministic`
2. Answer/annotation consistency test: `tests/test_puzzles_arithmetic_panel_tasks.py::test_arithmetic_panel_scene_tasks_use_scalar_target_bbox_annotation`
3. Prompt metadata/placeholder test: covered by source-layout checks and prompt-system tests.
4. Constraint-specific tests: `tests/test_puzzles_arithmetic_panel_tasks.py::test_forced_arithmetic_panel_branches_are_valid`
