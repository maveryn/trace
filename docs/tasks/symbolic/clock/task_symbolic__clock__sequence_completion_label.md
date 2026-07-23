# `task_symbolic__clock__sequence_completion_label`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `clock`
3. Scene: `clock`
4. Task id: `task_symbolic__clock__sequence_completion_label`
5. Objective: choose the option clock that completes a four-slot time sequence.

## Program Contract
Program: `clock.sequence_completion_label(scene=clock, scope=four_slot_clock_sequence_with_four_visual_options, output=option_label)`

Candidate set: the four visible option clocks labeled `A..D`.
Operands: the three shown sequence-clock times, the missing slot position, and the constant sequence step.
Operation: infer the hidden sequence time and select the unique option clock showing that time.
Output binding: `answer` is the selected option label.
Annotation witnesses: a `bbox_map` with `sequence_panel` and `correct_option` roles.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`, `matching`

## 2) Scene + Task Contract
1. Public branch metadata: `query_id`
2. Supported public `query_id`: `single`
3. Supported non-semantic visual axes:
   - `scene_variant`: `classic|minimal|outline`
   - `style_variant`: `accented|marker|studio`
   - `accent_color_name`: shared symbolic clock colors
   - `missing_slot_index`: top-row slot index 0 through 3
   - `sequence_step_minutes`: sampled constant step between sequence clocks
4. `answer_gt.type`: `string`
5. Answer schema: one option label from `A` through `D`.
6. `annotation_gt.type`: `bbox_map`
7. Annotation schema: `bbox_map` with roles `sequence_panel` and `correct_option`
8. Scene contract:
   - the top row contains four fixed boxes,
   - three top-row boxes show analog clocks and one box is the missing slot,
   - the bottom row contains exactly four labeled visual option clocks,
   - exactly one option has the time that belongs in the missing sequence slot,
   - distractor option times are unique and separated from the correct time.

## 3) Prompt Contract
1. Bundle: `symbolic_clock_v1`
2. `scene_key`: `clock`
3. `task_key`: `clock_sequence_completion_label_query`
4. Query prompt key: `sequence_completion_label`
5. Required slots:
   - scene: `object_description`
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
6. Modes: `answer_only`, `answer_and_annotation`

## 4) Annotation + Trace Contract
1. Prompt-facing annotation is a role-bound `bbox_map`.
2. `sequence_panel` marks the full top-row sequence region.
3. `correct_option` marks the selected visual option card.
4. Incorrect option cards are not prompt-facing annotation.
5. `projected_annotation` includes `bbox_map` and `pixel_bbox_map`.
6. `execution_trace` records the missing slot index, sequence step, hidden time, option times, correct option label, and resolved visual axes.

## 5) Determinism + Tests
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the finalized rendered sequence and option panel.
3. Behavior/trace/prompt tests: `tests/test_symbolic_clock_elapsed_sequence_tasks.py`
4. Source-layout contract tests: `tests/test_public_source_layout_contracts.py`
