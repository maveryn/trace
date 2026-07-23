# `task_symbolic__clock__full_time_readout`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `clock`
3. Scene: `clock`
4. Task id: `task_symbolic__clock__full_time_readout`
5. Objective: read the exact time from one three-hand analog clock.

## Program Contract
Program: `clock.full_time_readout(scene=clock, scope=single_three_hand_analog_clock_with_time_options, output=option_letter)`

Candidate set: the six labeled `HH:MM:SS` answer cards below the single visible analog clock.
Operands: the finalized clock center and the three hand-tip positions.
Operation: read the displayed hour, minute, and second values from the analog clock.
Output binding: `answer` is the option label whose card shows the displayed zero-padded `HH:MM:SS` time.
Annotation witnesses: the scalar `bbox` of the selected answer card.
Query ids: `single`.

## Reasoning Operations

Families: `direct_retrieval`

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
   - one three-hand analog clock is shown,
   - six labeled `HH:MM:SS` answer cards are shown below the clock,
   - the answer is the exact displayed hour, minute, and second time,
   - sampled times are constrained so the three hands are separated by at least the configured angle gap,
   - annotation marks the selected answer card only.

## 3) Prompt Contract
1. Bundle: `symbolic_clock_v1`
2. `scene_key`: `clock`
3. `task_key`: `clock_full_time_readout_query`
4. Query prompt key: `full_time_readout`
5. Required slots:
   - scene: `object_description`
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
6. Modes: `answer_only`, `answer_and_annotation`

## 4) Annotation + Trace Contract
1. Prompt-facing annotation is the scalar `bbox` of the selected answer card.
2. `projected_annotation` includes `bbox` and `pixel_bbox`.
3. `render_map` includes the clock center, hand tips, hand bboxes, face bbox, option bboxes, and selected option bbox.
4. `execution_trace` records the displayed time, raw answer time, answer label, option values, hand angle gaps, second support, and resolved visual axes.

## 5) Determinism + Tests
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the finalized rendered clock geometry.
3. Behavior/trace/prompt tests: `tests/test_symbolic_clock_readout_tasks.py`
4. Contract/build tests: `tests/test_symbolic_clock_readout_contracts.py`
