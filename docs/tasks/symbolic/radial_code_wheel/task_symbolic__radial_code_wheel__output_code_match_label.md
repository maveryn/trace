# `task_symbolic__radial_code_wheel__output_code_match_label`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `radial_code_wheel`
3. Task id: `task_symbolic__radial_code_wheel__output_code_match_label`
4. Objective contract: choose the code option that reaches a shown terminal output label.

## Program Contract
Program: `radial_code_wheel.output_to_code_match(scene=radial_code_wheel, scope=target_output_and_six_code_options, output=option_label)`

Candidate set: the six visible three-symbol code option cards labeled `A..F`.
Operands: the target output label and the three-ring radial code wheel mapping from code symbols to terminal output labels.
Operation: evaluate each option code through the inner, middle, and outer rings and select the unique code that reaches the target output label.
Output binding: `answer` is the selected option letter.
Annotation witnesses: a `point_map` with `inner_ring_symbol`, `middle_ring_symbol`, and `outer_ring_symbol` centers for the selected code.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `matching`

## 2) Scene + task contract
1. Entities/relations:
   - one three-ring radial code wheel with symbols `A|B|C|D`;
   - one visible target output card;
   - 64 unique terminal output labels on the wheel;
   - six visual three-symbol code options `A..F`.
2. Supported `query_id` values: `single`
3. `answer_gt.type`: `string`
4. Default `annotation_gt.type`: `point_map`
5. Annotation schema: `point_map`
6. Alternate annotation forms: none
7. Annotation witness policy:
   - minimal object/primitive witnesses: the inner, middle, and outer ring-symbol centers for the code that reaches the target output.
   - annotation shape choice: `point_map`, because the task binds three ordered semantic ring roles.
   - map annotation role names: `inner_ring_symbol`, `middle_ring_symbol`, `outer_ring_symbol`
   - numeric/readout annotation handling: not applicable.
   - answer-option annotation policy: the selected option card is not annotated; the answer field already provides the chosen option label.
8. Overlap/touch policy: terminal labels and option cards are rendered from fixed non-overlapping layout slots.

## 3) Prompt contract
1. `prompt_bundle_id`: `symbolic_radial_code_wheel_v1`
2. `scene_key`: `radial_code_wheel`
3. `task_key`: `output_code_match_label`
4. Optional query-id prompt mapping: none
5. Required slots:
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
6. JSON example validity rule: examples use a capital option letter answer and a three-role point-map annotation.
7. Variant counts: 5 scene templates, 5 task templates, 5 output-mode templates.
8. Output modes: `answer_only`, `answer_and_annotation`

## 4) Determinism + constraints
1. Seed namespaces used: task dataset, scene variant, prompt variants, and symbolic background/noise namespaces.
2. Unique-answer policy: all 64 terminal labels are unique; six code options are unique; exactly one option reaches the target output label.
3. Reject/resample conditions: invalid code length/symbols, duplicate terminal labels, unsupported option count, or insufficient distractors.
4. No-auto-relaxation guarantee: invalid constraints raise and retry with the next deterministic retry seed.

## 5) Tests
1. Determinism test: `tests/test_symbolic_radial_code_wheel_tasks.py`
2. Answer/annotation consistency test: same focused test file validates output-to-code lookup and annotation roles.
3. Prompt metadata/placeholder test: same focused test file validates prompt bundle keys and required slots.
4. Constraint-specific tests: same focused test file validates unique terminal/code option construction.
