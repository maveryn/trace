# `task_symbolic__clock__equivalent_time_label`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `clock`
3. Scene: `clock`
4. Task id: `task_symbolic__clock__equivalent_time_label`
5. Objective: choose the labeled option display that shows the same time as a reference clock display.

## Program Contract
Program: `clock.equivalent_time_label(scene=clock, scope=reference_clock_display_with_six_options, query=analog_reference_digital_options|digital_reference_analog_options, output=option_label)`

Candidate set: the six visible option displays labeled `A..F`.
Operands: the reference time and the time shown by each option display.
Operation: compare all option times to the reference time and select the unique equivalent option.
Output binding: `answer` is the selected option label.
Annotation witnesses: a `bbox_map` with `reference` and `correct_option` roles.
Query ids: `analog_reference_digital_options`, `digital_reference_analog_options`.

## Reasoning Operations

Families: `formula_evaluation`, `matching`

## 2) Scene + Task Contract
1. Public branch metadata: `query_id`
2. Supported public `query_id`: `analog_reference_digital_options`, `digital_reference_analog_options`
3. Supported non-semantic visual axes:
   - `scene_variant`: `classic|minimal|outline`
   - `style_variant`: `accented|marker|studio`
   - `accent_color_name`: shared symbolic clock colors
   - `digital_display_palette`: display color palette
4. `answer_gt.type`: `string`
5. Answer schema: one option label from `A` through `F`.
6. `annotation_gt.type`: `bbox_map`
7. Annotation schema: `bbox_map` with roles `reference` and `correct_option`
8. Scene contract:
   - the reference appears above six visual options,
   - exactly one option shows the same time as the reference,
   - distractor options use distinct times separated by the configured gap.

## 3) Prompt Contract
1. Bundle: `symbolic_clock_v1`
2. `scene_key`: `clock`
3. `task_key`: `clock_equivalent_time_label_query`
4. Required slots:
   - scene: `object_description`
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
5. Modes: `answer_only`, `answer_and_annotation`

## 4) Annotation + Trace Contract
1. Prompt-facing annotation is a role-bound `bbox_map`.
2. `reference` marks the reference clock/display; `correct_option` marks the matching option display.
3. Distractor options and option labels are not prompt-facing annotation.
4. `projected_annotation` includes `bbox_map` and `pixel_bbox_map`.
5. `render_map` includes reference, option, and correct-option bboxes.

## 5) Determinism + Tests
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the finalized rendered reference and options.
3. Behavior/trace/prompt tests: `tests/test_symbolic_clock_equivalent_time_tasks.py`
4. Source-layout contract tests: `tests/test_public_source_layout_contracts.py`
