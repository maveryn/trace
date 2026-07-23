# `task_symbolic__braille_cell__word_braille_match_label`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `braille_cell`
3. Scene: `braille_cell`
4. Task id: `task_symbolic__braille_cell__word_braille_match_label`
5. Objective: choose the Braille word-plate option that encodes a displayed source word.

## Program Contract
Program: `braille_cell.word_to_braille_match(scene=braille_cell, scope=source_word_and_four_braille_word_plate_options, output=option_label)`

Candidate set: the four visible Braille word-plate option cards labeled `A..D`.
Operands: the displayed source word and the raised-dot cell patterns shown in each option plate.
Operation: encode the source word as Grade 1 Braille and select the unique option plate whose cells match that encoding.
Output binding: `answer` is the selected option letter.
Annotation witnesses: a `bbox_map` with `source_word` and `selected_option` roles.
Query ids: `single`.

## Reasoning Operations

Families: `matching`

## 2) Scene + Task Contract
1. Public branch metadata: `query_id`
2. Supported `query_id`: `single`
3. Internal prompt/query key: `word_braille_match_label`
4. Supported non-semantic visual axes:
   - `scene_variant`: `clean_card|notebook_card|exam_scan`
   - shared symbolic background/noise style
5. `answer_gt.type`: `string`
6. Answer schema: `string`, one capital-letter option label from `A..D`.
7. `annotation_gt.type`: `bbox_map`
8. Annotation schema: `bbox_map`
9. Scene contract:
   - one source word is shown,
   - four visual Braille word-plate options are labeled `A..D`,
   - every option plate has the same number of Braille cells as the source word length,
   - all candidate words share at least the first character, so all plate options share at least the first Braille cell,
   - exactly one option encodes the source word,
   - words use Grade 1 uncontracted lowercase Braille letters only,
   - generated words are sampled from `trace_tasks.tasks.shared.word_assets`,
   - word length support is `3..5`.

## 3) Prompt Contract
1. Bundle: `symbolic_braille_cell_v1`
2. `scene_key`: `braille_cell`
3. `task_key`: `word_braille_match_label`
4. Required slots:
   - scene: `object_description`
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
5. Modes: `answer_only`, `answer_and_annotation`
6. Options are visual Braille word plates in the image, not prompt-only choices.

## 4) Annotation + Trace Contract
1. Prompt-facing annotation is a `bbox_map` with two semantic roles:
   - `source_word`: bbox around the displayed source word card,
   - `selected_option`: bbox around the correct Braille word-plate option.
2. Distractor option bboxes, individual Braille cells, individual dots, labels, and panel chrome are not prompt-facing annotation.
3. `projected_annotation` includes `bbox_map` and `pixel_bbox_map`.
4. `render_map` includes:
   - `item_bboxes_px`
   - `dot_centers_px`
   - `raised_dot_centers_px`
   - `cell_dot_centers_px`
5. `execution_trace.braille_metadata` records the source word, word length, source cell patterns, correct option label/id, option words, option cell patterns, and shared-prefix sampling metadata.

## 5) Determinism + Tests
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the same finalized source/option layout.
3. Behavior/trace/prompt tests: `tests/test_symbolic_braille_tasks.py`
4. Source-layout contract tests: `tests/test_public_source_layout_contracts.py`
