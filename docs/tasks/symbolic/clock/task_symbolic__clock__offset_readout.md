# `task_symbolic__clock__offset_readout`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `clock`
3. Scene: `clock`
4. Task id: `task_symbolic__clock__offset_readout`
5. Objective: read one analog clock and apply a minute offset.

## Program Contract
Program: `clock.offset_readout(scene=clock, scope=single_two_hand_clock_with_time_options, query=minutes_after|minutes_before, output=option_letter)`

Candidate set: the six labeled answer cards below the single visible analog clock.
Operands: the displayed time, sampled minute offset, and offset direction.
Operation: read the displayed time and add or subtract the requested offset in 12-hour time.
Output binding: `answer` is the option label whose card shows the resulting zero-padded `HH:MM` time.
Annotation witnesses: the scalar `bbox` of the selected answer card.
Query ids: `minutes_after`, `minutes_before`.

## Reasoning Operations

Families: `formula_evaluation`

## 2) Scene + Task Contract
1. Public branch metadata: `query_id`
2. Supported public `query_id`: `minutes_after`, `minutes_before`
3. Supported non-semantic visual axes:
   - `scene_variant`: `classic|minimal|outline`
   - `style_variant`: `accented|marker|studio`
   - `accent_color_name`: shared symbolic clock colors
4. `answer_gt.type`: `option_letter`
5. Answer schema: one visible option label from `A` through `F`.
6. `annotation_gt.type`: `bbox`
7. Annotation schema: scalar `bbox` for the selected answer card.
8. Scene contract:
   - one two-hand analog clock is shown,
   - six labeled time answer cards are shown below the clock,
   - the answer is the displayed time plus or minus `delta_minutes`,
   - annotation marks the selected answer card only.

## 3) Prompt Contract
1. Bundle: `symbolic_clock_v1`
2. `scene_key`: `clock`
3. `task_key`: `clock_offset_readout_query`
4. Required slots:
   - scene: `object_description`
   - query: `delta_minutes`
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
5. Modes: `answer_only`, `answer_and_annotation`

## 4) Annotation + Trace Contract
1. Prompt-facing annotation is the scalar `bbox` of the selected answer card.
2. `projected_annotation` includes `bbox` and `pixel_bbox`.
3. `render_map` includes the clock center, hand tips, hand bboxes, face bbox, option bboxes, and selected option bbox.
4. `execution_trace` records shown time, offset direction, offset value, raw answer time, answer label, option values, and resolved visual axes.

## 5) Determinism + Tests
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the finalized rendered clock geometry.
3. Behavior/trace/prompt tests: `tests/test_symbolic_clock_readout_tasks.py`
4. Contract/build tests: `tests/test_symbolic_clock_readout_contracts.py`
