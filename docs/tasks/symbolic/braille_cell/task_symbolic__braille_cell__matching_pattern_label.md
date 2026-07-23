# `task_symbolic__braille_cell__matching_pattern_label`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `braille_cell`
3. Scene: `braille_cell`
4. Task id: `task_symbolic__braille_cell__matching_pattern_label`
5. Objective: choose the labeled option cell with the same raised-dot pattern as a reference Braille cell.

## Program Contract
Program: `braille_cell.pattern_match(scene=braille_cell, scope=reference_cell_and_six_visual_options, output=option_label)`

Candidate set: the six visible Braille cell option cards labeled `A..F`.
Operands: the raised-dot pattern of the marked reference cell and the raised-dot pattern of each option cell.
Operation: compare option patterns to the reference pattern and select the unique matching option.
Output binding: `answer` is the selected option letter.
Annotation witnesses: a `bbox_map` with `reference_cell` and `selected_option` roles.
Query ids: `single`.

## Reasoning Operations

Families: `matching`

## 2) Scene + Task Contract
1. Public branch metadata: `query_id`
2. Supported `query_id`: `single`
3. Internal prompt/query key: `matching_pattern_label`
4. Supported non-semantic visual axes:
   - `scene_variant`: `clean_card|notebook_card|exam_scan`
   - shared symbolic background/noise style
5. `answer_gt.type`: `string`
6. Answer schema: `string`, one capital-letter option label from `A..F`.
7. `annotation_gt.type`: `bbox_map`
8. Annotation schema: `bbox_map`
9. Scene contract:
   - one marked reference Braille cell is shown,
   - six visual option cells are labeled `A..F`,
   - exactly one option has the same raised-dot pattern as the reference,
   - no Braille alphabet knowledge is required.

## 3) Prompt Contract
1. Bundle: `symbolic_braille_cell_v1`
2. `scene_key`: `braille_cell`
3. `task_key`: `braille_matching_pattern_label`
4. Required slots:
   - scene: `object_description`
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
5. Modes: `answer_only`, `answer_and_annotation`
6. Options are visual cells in the image, not prompt-only choices.

## 4) Annotation + Trace Contract
1. Prompt-facing annotation is a `bbox_map` with two semantic roles:
   - `reference_cell`: bbox around the reference cell,
   - `selected_option`: bbox around the correct visual option cell.
2. Distractor option bboxes, individual dots, labels, and panel chrome are not prompt-facing annotation.
3. `projected_annotation` includes `bbox_map` and `pixel_bbox_map`.
4. `render_map` includes:
   - `item_bboxes_px`
   - `dot_centers_px`
   - `raised_dot_centers_px`
   - `cell_dot_centers_px`
5. `execution_trace.braille_metadata` records the reference pattern, correct option label/id, and option patterns.

## 5) Determinism + Tests
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the same finalized reference/option layout.
3. Behavior/trace/prompt tests: `tests/test_symbolic_braille_tasks.py`
4. Source-layout contract tests: `tests/test_public_source_layout_contracts.py`
