# `task_symbolic__abacus__target_value_match_label`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `abacus`
3. Task id: `task_symbolic__abacus__target_value_match_label`
4. Objective: select the labeled abacus option whose bead positions represent the target integer named in the prompt.

## Program Contract
Program: `selection.option_match(scene=abacus, scope=six_abacus_options, rule=target_value_equal_displayed_abacus_value, output=option_letter)`

Candidate set: the six visible abacus option cards labeled `A..F`.
Operands: the prompt target integer and the value represented by each option's three-column soroban-style abacus.
Operation: compute every option value from its bead states and select the unique option whose value equals the prompt target.
Output binding: `answer` is the selected option letter.
Annotation witnesses: the scalar bbox of the selected option card.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`, `matching`

## 2) Scene + Task Contract
1. Supported public `query_id` values: `single`
2. `answer_gt.type`: `option_letter`
3. Annotation schema: `bbox`
4. `annotation_gt.type`: `bbox`
5. Non-semantic visual axes: `scene_variant=clean_card|wood_frame|worksheet`, plus shared symbolic background/noise.
6. Scene contract:
   - exactly six visual option cards labeled `A..F`,
   - each option card contains one compact three-column soroban-style abacus with columns labeled `100`, `10`, and `1`,
   - each abacus column has one upper bead worth `5` and four lower beads worth `1`,
   - the target value appears in the prompt, not as target text inside the image,
   - exactly one option represents the target value,
   - target value support is `0..999`.

## 3) Prompt Contract
1. Bundle: `symbolic_abacus_v1`
2. `scene_key`: `abacus`
3. `task_key`: `abacus_target_value_match_query`
4. Required task slot: `question_text`
5. Prompt-facing answer is the single matching option label.

## 4) Annotation + Trace Contract
1. Annotation is the scalar pixel-space bounding box `[x0,y0,x1,y1]` around the selected option card.
2. Annotating the selected option card is allowed because this is a visual option-panel task.
3. Option cards are non-overlapping and arranged in fixed row-major order.
4. Answer and annotation are bound from the same finalized option record.

## 5) Tests
1. Behavior/trace/prompt tests: `tests/test_symbolic_abacus_tasks.py`
2. Prompt/config tests: `tests/test_prompt_system.py`, `tests/test_symbolic_core_scene_config.py`
3. Source-layout contract tests: `tests/test_public_source_layout_contracts.py`
