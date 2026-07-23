# `task_symbolic__abacus__displayed_value_readout`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `abacus`
3. Task id: `task_symbolic__abacus__displayed_value_readout`
4. Objective: select the labeled numeric option matching the integer value represented by a three-column soroban-style abacus.

## Program Contract
Program: `selection.option_match(scene=abacus, scope=three_place_value_columns, rule=soroban_digit_sum, output=option_letter)`

Candidate set: the three visible place-value columns labeled `100`, `10`, and `1`.
Operands: the active upper and lower bead states in each column after final layout, plus the six visible numeric option cards.
Operation: compute each soroban digit from the active beads, combine the digits as `100 * hundreds + 10 * tens + ones`, then select the unique option card whose displayed number equals that value.
Output binding: `answer` is the selected option letter.
Annotation witnesses: the selected option-card bounding box.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## 2) Scene + Task Contract
1. Supported public `query_id` values: `single`
2. `answer_gt.type`: `option_letter`
3. Annotation schema: `bbox`
4. `annotation_gt.type`: `bbox`
5. Non-semantic visual axes: `scene_variant=clean_card|wood_frame|worksheet`, plus shared symbolic background/noise.
6. Scene contract:
   - exactly three place-value columns labeled `100`, `10`, and `1`,
   - each column has one upper bead worth `5` and four lower beads worth `1`,
   - active beads are moved toward the center beam,
   - six visible option cards are labeled `A..F`,
   - exactly one option card displays `hundreds_digit * 100 + tens_digit * 10 + ones_digit`,
   - answer support is the six visible option labels.

## 3) Prompt Contract
1. Bundle: `symbolic_abacus_v1`
2. `scene_key`: `abacus`
3. `task_key`: `abacus_displayed_value_query`
4. Required task slot: `question_text`
5. The prompt explicitly states the abacus rule: beads moved toward the center bar are active, an active upper bead counts as `5`, each active lower bead counts as `1`, and columns are read as `100`, `10`, and `1` from left to right.
6. Prompt-facing answer is the single capital-letter option label.

## 4) Annotation + Trace Contract
1. Annotation is the scalar bbox around the selected option card.
2. Annotation marks the selected option card, not the abacus beads, rods, labels, or frame.
3. The trace records the computed integer value, digits by column, option values by label, and selected option label.
4. Answer and annotation are bound from the same finalized rendered option set.

## 5) Tests
1. Behavior/trace/prompt tests: `tests/test_symbolic_abacus_tasks.py`
2. Prompt/config tests: `tests/test_prompt_system.py`, `tests/test_symbolic_core_scene_config.py`
3. Source-layout contract tests: `tests/test_public_source_layout_contracts.py`
