# `task_symbolic__morse_code__word_morse_match_label`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `morse_code`
3. Scene: `morse_code`
4. Task id: `task_symbolic__morse_code__word_morse_match_label`
5. Objective: choose the Morse-code option that encodes a source word.

## Program Contract
Program: `morse_code.word_to_morse_match(scene=morse_code, scope=source_word_and_four_morse_code_options, output=option_label)`

Candidate set: the four visible Morse-code option cards labeled `A..D`.
Operands: the displayed source word and the Morse symbol groups shown in each option card.
Operation: encode the source word as International Morse code and select the unique option whose symbols match that encoding.
Output binding: `answer` is the selected option letter.
Annotation witnesses: a `bbox_map` with `source_word` and `selected_option` roles.
Query ids: `single`.

## Reasoning Operations

Families: `matching`

## 2) Scene + Task Contract
1. Public branch metadata: `query_id`
2. Supported `query_id`: `single`
3. Internal prompt/query key: `word_morse_match_label`
4. Supported non-semantic visual axes:
   - `scene_variant`: `clean_card|notebook_card|exam_scan`
   - shared symbolic background/noise style
5. `answer_gt.type`: `string`
6. Answer schema: `string`, one capital-letter option label from `A..D`.
7. `annotation_gt.type`: `bbox_map`
8. Annotation schema: `bbox_map`
9. Scene contract:
   - one visible source word is shown,
   - four visual Morse-code word cards are labeled `A..D`,
   - every option code corresponds to a same-length candidate word,
   - all candidate words share at least the first character, so all code options share at least the first Morse letter,
   - exactly one option encodes the source word,
   - words use International Morse code over lowercase English letters,
   - generated words are sampled from `trace_tasks.tasks.shared.word_assets`,
   - word length support is `3..5`.

## 3) Prompt Contract
1. Bundle: `symbolic_morse_code_v1`
2. `scene_key`: `morse_code`
3. `task_key`: `word_morse_match_label`
4. Required slots:
   - scene: `object_description`
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
5. Modes: `answer_only`, `answer_and_annotation`
6. Options are visible Morse-code cards in the image, not prompt-only choices.

## 4) Annotation + Trace Contract
1. Prompt-facing annotation is a `bbox_map` with two semantic roles:
   - `source_word`: bbox around the source word card,
   - `selected_option`: bbox around the correct visual Morse-code option.
2. Distractor option bboxes, individual Morse symbols, individual letters, text bboxes, labels, and panel chrome are not prompt-facing annotation.
3. `projected_annotation` includes `bbox_map` and `pixel_bbox_map`.
4. `render_map` includes:
   - `item_bboxes_px`
   - `symbol_bboxes_px`
5. `execution_trace.morse_metadata` records the source word, word length, source letter codes, correct option label/id, option words, option letter codes, and shared-prefix sampling metadata.

## 5) Determinism + Tests
1. Deterministic sampling/rendering from `instance_seed`.
2. Answers and annotation come from the same finalized source/option layout.
3. Behavior/trace/prompt tests: `tests/test_symbolic_morse_code_tasks.py`
4. Source-layout contract tests: `tests/test_public_source_layout_contracts.py`
