# `task_symbolic__morse_code__morse_word_read_label`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `morse_code`
3. Scene: `morse_code`
4. Task id: `task_symbolic__morse_code__morse_word_read_label`
5. Objective: choose the word option encoded by a Morse-code word card.

## Program Contract
Program: `morse_code.word_read(scene=morse_code, scope=source_morse_word_code_and_four_word_options, output=option_label)`

Candidate set: the four visible word option cards labeled `A..D`.
Operands: the source Morse symbol groups and the candidate lowercase option words.
Operation: decode the source International Morse word and select the unique option word that matches it.
Output binding: `answer` is the selected option letter.
Annotation witnesses: a `bbox_map` with `source_code` and `selected_option` roles.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`

## 2) Scene + Task Contract
1. Public branch metadata: `query_id`
2. Supported `query_id`: `single`
3. Internal prompt/query key: `morse_word_read_label`
4. Supported non-semantic visual axes:
   - `scene_variant`: `clean_card|notebook_card|exam_scan`
   - shared symbolic background/noise style
5. `answer_gt.type`: `string`
6. Answer schema: `string`, one capital-letter option label from `A..D`.
7. `annotation_gt.type`: `bbox_map`
8. Annotation schema: `bbox_map`
9. Scene contract:
   - one source Morse-code word card is shown,
   - four visual word options are labeled `A..D`,
   - every option word has the same length as the source word,
   - all option words share at least the first character, so the first Morse letter alone cannot identify the answer,
   - exactly one option is the lowercase word encoded by the Morse code,
   - words use International Morse code over lowercase English letters,
   - generated words are sampled from `trace_tasks.tasks.shared.word_assets`,
   - word length support is `3..5`.

## 3) Prompt Contract
1. Bundle: `symbolic_morse_code_v1`
2. `scene_key`: `morse_code`
3. `task_key`: `morse_word_read_label`
4. Required slots:
   - scene: `object_description`
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
5. Modes: `answer_only`, `answer_and_annotation`
6. Options are visible word cards in the image, not prompt-only choices.

## 4) Annotation + Trace Contract
1. Prompt-facing annotation is a `bbox_map` with two semantic roles:
   - `source_code`: bbox around the source Morse-code word card,
   - `selected_option`: bbox around the correct visual word option.
2. Distractor option bboxes, individual Morse symbols, individual letters, text bboxes, labels, and panel chrome are not prompt-facing annotation.
3. `projected_annotation` includes `bbox_map` and `pixel_bbox_map`.
4. `render_map` includes:
   - `item_bboxes_px`
   - `symbol_bboxes_px`
5. `execution_trace.morse_metadata` records the target word, word length, target letter codes, correct option label/id, option words, and shared-prefix sampling metadata.

## 5) Determinism + Tests
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the same finalized source/option layout.
3. Behavior/trace/prompt tests: `tests/test_symbolic_morse_code_tasks.py`
4. Source-layout contract tests: `tests/test_public_source_layout_contracts.py`
