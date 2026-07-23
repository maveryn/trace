# `task_symbolic__abacus__place_digit_readout`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `abacus`
3. Task id: `task_symbolic__abacus__place_digit_readout`
4. Objective: select the labeled numeric option matching the digit shown in one queried place-value column of a three-column soroban-style abacus.

## Program Contract
Program: `selection.option_match(scene=abacus, scope=queried_place_value_column, rule=soroban_digit_sum, output=option_letter)`

Candidate set: the three visible place-value columns labeled `100`, `10`, and `1`.
Operands: the queried place-value label, the active upper/lower bead states in that column after final layout, and the six visible numeric option cards.
Operation: compute the soroban digit for the queried column, then select the unique option card whose displayed digit equals that value.
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
   - the prompt identifies the queried column by its visible place-value label,
   - six visible option cards are labeled `A..F`,
   - exactly one option card displays the digit represented by that column alone.

## 3) Prompt Contract
1. Bundle: `symbolic_abacus_v1`
2. `scene_key`: `abacus`
3. `task_key`: `abacus_place_digit_query`
4. Required task slot: `question_text`
5. The prompt explicitly states the abacus rule: beads moved toward the center bar are active, an active upper bead counts as `5`, and each active lower bead counts as `1`.
6. Prompt-facing answer is the single capital-letter option label.

## 4) Annotation + Trace Contract
1. Annotation is the scalar bbox around the selected option card.
2. Annotation marks the selected option card, not the abacus beads, rods, labels, inactive beads, or frame.
3. The trace records `target_column_role`, `target_place_label`, `target_place_value`, `digits_by_role`, the computed digit, option values by label, and the selected option label.
4. Answer and annotation are bound from the same finalized rendered option set.

## 5) Tests
1. Behavior/trace/prompt tests: `tests/test_symbolic_abacus_tasks.py`
2. Prompt/config tests: `tests/test_prompt_system.py`, `tests/test_symbolic_core_scene_config.py`
3. Source-layout contract tests: `tests/test_public_source_layout_contracts.py`
