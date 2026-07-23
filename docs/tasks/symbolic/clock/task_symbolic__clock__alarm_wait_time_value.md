# `task_symbolic__clock__alarm_wait_time_value`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `clock`
3. Scene: `clock`
4. Task id: `task_symbolic__clock__alarm_wait_time_value`
5. Objective: compute the forward wait time until an analog alarm-clock hour.

## Program Contract
Program: `clock.alarm_wait_time_value(scene=clock, scope=single_alarm_analog_clock_with_minute_options, output=option_letter)`

Candidate set: the six labeled minute answer cards below the single visible analog alarm clock.
Operands: the current time shown by the dark hands and the alarm hour shown by the red hand.
Operation: read the current time, interpret the red alarm hand on the same 1-12 hour scale as the numerals with alarm minute fixed at `:00`, and compute the forward minutes until the next alarm occurrence.
Output binding: `answer` is the option label whose card shows the wait time in minutes.
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
   - `accent_color_name`: shared symbolic clock colors, with red-like accents disabled for this task
4. `answer_gt.type`: `option_letter`
5. Answer schema: one visible option label from `A` through `F`.
6. `annotation_gt.type`: `bbox`
7. Annotation schema: scalar `bbox` for the selected answer card.
8. Scene contract:
   - one analog clock is shown,
   - dark hour and minute hands show the current time,
   - the red alarm hand is a distinct semantic hand and points to an hour numeral on the 1-12 hour scale,
   - six labeled minute answer cards are shown below the clock,
   - the alarm minute is fixed at `:00`,
   - current hands and the red alarm hand are separated by configured angle-gap constraints,
   - annotation marks the selected answer card only.

## 3) Prompt Contract
1. Bundle: `symbolic_clock_v1`
2. `scene_key`: `clock`
3. `task_key`: `clock_alarm_wait_time_value_query`
4. Query prompt key: `alarm_wait_time_value`
5. Required slots:
   - scene: `object_description`
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
6. Modes: `answer_only`, `answer_and_annotation`

## 4) Annotation + Trace Contract
1. Prompt-facing annotation is the scalar `bbox` of the selected answer card.
2. `projected_annotation` includes `bbox` and `pixel_bbox`.
3. `render_map` includes the clock center, current-hand tips, alarm-hand tip, hand bboxes, face bbox, option bboxes, and selected option bbox.
4. `execution_trace` records shown time, alarm hour, alarm time text, raw wait minutes, answer label, option values, hand angle gaps, and resolved visual axes.

## 5) Determinism + Tests
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the finalized rendered clock geometry.
3. Behavior/trace/prompt tests: `tests/test_symbolic_clock_readout_tasks.py`
4. Contract/build tests: `tests/test_symbolic_clock_readout_contracts.py`
