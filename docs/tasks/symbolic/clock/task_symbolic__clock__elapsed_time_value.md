# `task_symbolic__clock__elapsed_time_value`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `clock`
3. Scene: `clock`
4. Task id: `task_symbolic__clock__elapsed_time_value`
5. Objective: compute the forward elapsed minutes from clock A to clock B.

## Program Contract
Program: `clock.elapsed_time_value(scene=clock, scope=two_labeled_analog_clocks_with_minute_options, output=option_letter)`

Candidate set: the six labeled minute answer cards below the two visible analog clock faces labeled `A` and `B`.
Operands: the displayed start time on clock `A` and displayed end time on clock `B`.
Operation: compute the forward elapsed time from `A` to `B` around the 12-hour clock.
Output binding: `answer` is the option label whose card shows the elapsed minutes.
Annotation witnesses: the scalar `bbox` of the selected answer card.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## 2) Scene + Task Contract
1. Public branch metadata: `query_id`
2. Supported public `query_id`: `single`
3. Supported non-semantic visual axes:
   - `scene_variant`: `classic|minimal|outline`
   - `style_variant`: `accented|marker|studio`
   - `accent_color_name`: shared symbolic clock colors
4. `answer_gt.type`: `option_letter`
5. Answer schema: one visible option label from `A` through `F`.
6. `annotation_gt.type`: `bbox`
7. Annotation schema: scalar `bbox` for the selected answer card.
8. Scene contract:
   - two labeled analog clocks are shown side by side,
   - clock A is the starting time and clock B is the ending time,
   - six labeled minute answer cards are shown below the clocks,
   - the prompt explicitly asks for the forward elapsed time around the 12-hour clock,
   - the configured elapsed-minute support excludes zero and full-cycle ambiguities.

## 3) Prompt Contract
1. Bundle: `symbolic_clock_v1`
2. `scene_key`: `clock`
3. `task_key`: `clock_elapsed_time_value_query`
4. Query prompt key: `elapsed_time_value`
5. Required slots:
   - scene: `object_description`
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
6. Modes: `answer_only`, `answer_and_annotation`

## 4) Annotation + Trace Contract
1. Prompt-facing annotation is the scalar `bbox` of the selected answer card.
2. Clock faces, clock labels, and caption text are not prompt-facing annotation.
3. `projected_annotation` includes `bbox` and `pixel_bbox`.
4. `render_map` includes source clock bboxes, option bboxes, and selected option bbox.
5. `execution_trace` records both displayed times, raw elapsed minutes, answer label, option values, and resolved visual axes.

## 5) Determinism + Tests
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the finalized rendered start/end clocks.
3. Behavior/trace/prompt tests: `tests/test_symbolic_clock_elapsed_sequence_tasks.py`
4. Source-layout contract tests: `tests/test_public_source_layout_contracts.py`
