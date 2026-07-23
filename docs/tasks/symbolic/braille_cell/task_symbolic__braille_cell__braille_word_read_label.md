# `task_symbolic__braille_cell__braille_word_read_label`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `braille_cell`
3. Scene: `braille_cell`
4. Task id: `task_symbolic__braille_cell__braille_word_read_label`
5. Objective: choose the word option encoded by a multi-cell Braille plate.

## Program Contract
Program: `braille_cell.word_read(scene=braille_cell, scope=source_braille_word_plate_and_four_word_options, output=option_label)`

Candidate set: the four visible word option cards labeled `A..D`.
Operands: the raised-dot patterns in each source Braille cell and the candidate lowercase option words.
Operation: decode the source Grade 1 Braille word and select the unique option word that matches it.
Output binding: `answer` is the selected option letter.
Annotation witnesses: a `bbox_map` with `source_plate` and `selected_option` roles.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`

## 2) Scene + Task Contract
1. Public branch metadata: `query_id`
2. Supported `query_id`: `single`
3. Internal prompt/query key: `braille_word_read_label`
4. Supported non-semantic visual axes:
   - `scene_variant`: `clean_card|notebook_card|exam_scan`
   - shared symbolic background/noise style
5. `answer_gt.type`: `string`
6. Answer schema: `string`, one capital-letter option label from `A..D`.
7. `annotation_gt.type`: `bbox_map`
8. Annotation schema: `bbox_map`
9. Scene contract:
   - one source Braille word plate is shown,
   - four visual word options are labeled `A..D`,
   - every option word has the same length as the source word,
   - all option words share at least the first character, so the first Braille cell alone cannot identify the answer,
   - exactly one option is the lowercase word encoded by the Braille plate,
   - words use Grade 1 uncontracted lowercase Braille letters only,
   - generated words are sampled from `trace_tasks.tasks.shared.word_assets`,
   - word length support is `3..5`.

## 3) Prompt Contract
1. Bundle: `symbolic_braille_cell_v1`
2. `scene_key`: `braille_cell`
3. `task_key`: `braille_word_read_label`
4. Required slots:
   - scene: `object_description`
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
5. Modes: `answer_only`, `answer_and_annotation`
6. Options are visible word cards in the image, not prompt-only choices.

## 4) Annotation + Trace Contract
1. Prompt-facing annotation is a `bbox_map` with two semantic roles:
   - `source_plate`: bbox around the source Braille word plate,
   - `selected_option`: bbox around the correct visual word option.
2. Distractor option bboxes, individual Braille cells, individual dots, text bboxes, labels, and panel chrome are not prompt-facing annotation.
3. `projected_annotation` includes `bbox_map` and `pixel_bbox_map`.
4. `render_map` includes:
   - `item_bboxes_px`
   - `dot_centers_px`
   - `raised_dot_centers_px`
   - `cell_dot_centers_px`
5. `execution_trace.braille_metadata` records the target word, word length, target cell patterns, correct option label/id, option words, and shared-prefix sampling metadata.

## 5) Determinism + Tests
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the same finalized source/option layout.
3. Behavior/trace/prompt tests: `tests/test_symbolic_braille_tasks.py`
4. Source-layout contract tests: `tests/test_public_source_layout_contracts.py`
