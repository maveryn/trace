# `task_symbolic__clock__time_extremum_label`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `clock`
3. Scene: `clock`
4. Task id: `task_symbolic__clock__time_extremum_label`
5. Objective: identify the labeled analog clock with the earliest or latest shown time.

## Program Contract
Program: `clock.time_extremum_label(scene=clock, scope=labeled_analog_grid, query=earliest_time_label|latest_time_label, output=clock_label)`

Candidate set: the six visible labeled analog clocks.
Operands: the displayed time on each clock and the requested extremum direction.
Operation: compare all shown times and select the unique earliest or latest clock.
Output binding: `answer` is the selected visible clock label.
Annotation witnesses: the scalar bbox of the selected clock face.
Query ids: `earliest_time_label`, `latest_time_label`.

## Reasoning Operations

Families: `ranking`

## 2) Scene + Task Contract
1. Public branch metadata: `query_id`
2. Supported public `query_id`: `earliest_time_label`, `latest_time_label`
3. Supported non-semantic visual axes:
   - `scene_variant`: `classic|minimal|outline`
   - `style_variant`: `accented|marker|studio`
   - `accent_color_name`: shared symbolic clock colors
4. `answer_gt.type`: `string`
5. Answer schema: one visible clock label.
6. `annotation_gt.type`: `bbox`
7. Annotation schema: scalar `bbox`
8. Scene contract:
   - six analog clocks are labeled with unique letters,
   - exactly one clock has the requested extremum time,
   - the extremum clock is separated by at least the configured comparison gap.

## 3) Prompt Contract
1. Bundle: `symbolic_clock_v1`
2. `scene_key`: `clock`
3. `task_key`: `clock_time_extremum_label_query`
4. Required slots:
   - scene: `object_description`
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
5. Modes: `answer_only`, `answer_and_annotation`

## 4) Annotation + Trace Contract
1. Prompt-facing annotation is the scalar bbox for the selected clock face.
2. Clock labels, non-winning clocks, hands on non-winning clocks, and panel background are not prompt-facing annotation.
3. `projected_annotation` includes `bbox` and `pixel_bbox`.
4. `render_map` includes `clocks_by_label`, `winning_label`, and `winning_clock_bbox_px`.
5. `execution_trace` records all visible clock labels and shown times.

## 5) Determinism + Tests
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the finalized rendered clock grid.
3. Behavior/trace/prompt tests: `tests/test_symbolic_clock_compare_tasks.py`
4. Contract/build tests: `tests/test_symbolic_clock_compare_contracts.py`
