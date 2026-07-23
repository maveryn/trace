# `task_icons__mirror_grid__missing_mirror_cell_label`

## 1) Identity
1. Domain: `icons`
2. Scene id: `mirror_grid`
3. Task id: `task_icons__mirror_grid__missing_mirror_cell_label`
4. Objective contract: `missing_mirror_cell_label`

## 2) Scene + task contract
1. Entities/relations: one mirror-symmetry icon grid with a visible mirror axis, one question-mark cell, and labeled visual option cells.
2. Supported `query_id` values: `single`.
3. `answer_gt.type`: `option_letter`.
4. Answer precision/format, if narrower than the registered answer type: one visible option label from the rendered options, sampled from `A..D` or `A..F`.
5. Default `annotation_gt.type`: `bbox`.
6. Annotation schema: `bbox`.
7. Alternate annotation forms: none.
8. Annotation witness policy:
   - annotation contract family: `direct_visible_answer`
   - minimal visual answer-verification witnesses: the selected option cell that supplies the missing icon.
   - derivation/proof details kept in trace metadata: mirror axis, missing cell location, mirrored counterpart cell location, grid cells, option cells, and icon placements.
   - annotation shape choice: scalar `bbox` around the selected option cell.
   - map annotation role names, if used: none.
   - numeric/readout annotation handling: not applicable.
   - answer-option annotation policy: the answer is a complete visual option image, so annotation marks the selected option cell.
9. Overlap/touch policy, if applicable: not applicable.

## Program Contract

Program: `selection.option_completion(scene=mirror_grid, scope=missing_cell_and_option_icons, rule=mirror_axis_reflection, output=option_letter)`

Candidate set: labeled option icon cells.
Operands: the visible mirror axis, the question-mark cell, the mirrored counterpart cell, and candidate option icons.
Operation: choose the option icon that would make the rendered grid mirror-symmetric when placed into the question-mark cell.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final option.
Annotation witnesses: `annotation` uses the scalar `bbox` schema for the selected option cell.
Query ids: `single`.
Internal generation axes: `mirror_axis` is sampled from `vertical` or `horizontal`.

## Reasoning Operations

Families: `transformation`, `matching`

## 3) Prompt contract
1. `prompt_bundle_id`: `icons_mirror_grid_v1`
2. `scene_key`: `missing_mirror_grid`
3. `task_key`: `missing_mirror_cell_label`
4. Optional query-id prompt mapping: none.
5. Required slots:
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
6. JSON example validity rule: examples use a scalar option-cell `bbox` annotation and an option-letter answer.
7. Variant counts: scene/task/output-mode variants come from `icons_mirror_grid_v1`.
8. Output modes: `answer_only`, `answer_and_annotation`.

## 4) Determinism + constraints
1. Seed namespaces used: `spawn_rng(instance_seed, "scene")`, plus task-specific mirror-axis sampling.
2. Unique-answer policy: the correct option is rendered from the missing cell's required reflected icon; every distractor option is checked against the correct option image digest.
3. Reject/resample conditions: unsupported option count, unsupported mirror axis, empty icon pool, palette separation failure, or inability to sample unique visual distractor options.
4. No-auto-relaxation guarantee: semantic constraints are not relaxed; failed construction raises and lets the caller retry within `max_attempts`.

## 5) Tests
1. Determinism test: `tests/test_icons_mirror_grid_missing_cell_tasks.py`
2. Answer/annotation consistency test: `tests/test_icons_mirror_grid_missing_cell_tasks.py`
3. Prompt metadata/placeholder test: `tests/test_icons_mirror_grid_missing_cell_tasks.py`
4. Constraint-specific tests: option counts, vertical/horizontal axes, scalar selected-option annotation.
