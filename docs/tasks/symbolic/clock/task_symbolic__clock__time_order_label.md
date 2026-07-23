# `task_symbolic__clock__time_order_label`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `clock`
3. Scene: `clock`
4. Task id: `task_symbolic__clock__time_order_label`
5. Objective: choose the option card that lists four clock labels from earliest to latest.

## Program Contract
Program: `clock.time_order_label(scene=clock, scope=four_labeled_analog_clocks_with_six_order_options, output=option_label)`

Candidate set: the six numbered answer cards.
Operands: the four labeled analog clock times and the candidate label order shown on each answer card.
Operation: read the four clock times and select the unique card whose order is earliest to latest in the 12-hour cycle, with 12 before 1.
Output binding: `answer` is the selected numbered option label.
Annotation witnesses: one `bbox` marking the correct answer card.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`

## 2) Scene + Task Contract
1. Public branch metadata: `query_id`
2. Supported public `query_id`: `single`
3. Supported non-semantic visual axes:
   - `scene_variant`: `classic|minimal|outline`
   - `style_variant`: `accented|marker|studio`
   - `accent_color_name`: shared symbolic clock colors
4. `answer_gt.type`: `string`
5. Answer schema: one option label from `1` through `6`.
6. `annotation_gt.type`: `bbox`
7. Annotation schema: `bbox`; `[x0, y0, x1, y1]` for the correct option card only.
8. Scene contract:
   - the top row contains exactly four analog clocks labeled `A..D`,
   - the bottom panel contains exactly six numbered answer cards,
   - each answer card lists one candidate ordering of the four clock labels,
   - exactly one card matches the earliest-to-latest order,
   - all visible clock times are distinct and separated by the configured comparison gap.

## 3) Prompt Contract
1. Bundle: `symbolic_clock_v1`
2. `scene_key`: `clock`
3. `task_key`: `clock_time_order_label_query`
4. Query prompt key: `time_order_label`
5. Required slots:
   - scene: `object_description`
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
6. Modes: `answer_only`, `answer_and_annotation`

## 4) Annotation + Trace Contract
1. Prompt-facing annotation is a single bbox for the correct answer card.
2. Clock-face bboxes are trace metadata only and are not prompt-facing annotation.
3. `projected_annotation` includes `bbox` and `pixel_bbox`.
4. `render_map` records clock bboxes, option-card bboxes, the correct label, and the correct option bbox.
5. `execution_trace` records shown times, the true label order, all candidate option orders, the correct option label, and resolved visual axes.

## 5) Determinism + Tests
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the finalized rendered clock and option panel.
3. Behavior/trace/prompt tests: `tests/test_symbolic_clock_compare_tasks.py`
4. Contract/build tests: `tests/test_symbolic_clock_compare_contracts.py`
