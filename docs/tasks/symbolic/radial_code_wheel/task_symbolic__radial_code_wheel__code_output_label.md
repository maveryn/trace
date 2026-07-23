# `task_symbolic__radial_code_wheel__code_output_label`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `radial_code_wheel`
3. Task id: `task_symbolic__radial_code_wheel__code_output_label`
4. Objective contract: choose the output-label option reached by a three-symbol radial code.

## Program Contract
Program: `radial_code_wheel.code_to_output_lookup(scene=radial_code_wheel, scope=source_code_and_six_output_options, output=option_label)`

Candidate set: the six visible output-label option cards labeled `A..F`.
Operands: the shown three-symbol source code and the three-ring radial code wheel mapping to terminal output labels.
Operation: follow the source code through the inner, middle, and outer ring symbols to find the terminal output, then select the option containing that output label.
Output binding: `answer` is the selected option letter.
Annotation witnesses: a `point_map` with `inner_ring_symbol`, `middle_ring_symbol`, and `outer_ring_symbol` centers for the shown code.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`

## 2) Scene + task contract
1. Entities/relations:
   - one three-ring radial code wheel with symbols `A|B|C|D`;
   - one visible source code card;
   - 64 unique terminal output labels on the wheel;
   - six visual output-label options `A..F`.
2. Supported `query_id` values: `single`
3. `answer_gt.type`: `string`
4. Default `annotation_gt.type`: `point_map`
5. Annotation schema: `point_map`
6. Alternate annotation forms: none
7. Annotation witness policy:
   - minimal object/primitive witnesses: the inner, middle, and outer ring-symbol centers followed by the shown code.
   - annotation shape choice: `point_map`, because the task binds three ordered semantic ring roles.
   - map annotation role names: `inner_ring_symbol`, `middle_ring_symbol`, `outer_ring_symbol`
   - numeric/readout annotation handling: not applicable.
   - answer-option annotation policy: the selected option card is not annotated; the answer field already provides the chosen option label.
8. Overlap/touch policy: terminal labels and option cards are rendered from fixed non-overlapping layout slots.

## 3) Prompt contract
1. `prompt_bundle_id`: `symbolic_radial_code_wheel_v1`
2. `scene_key`: `radial_code_wheel`
3. `task_key`: `code_output_label`
4. Optional query-id prompt mapping: none
5. Required slots:
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
6. JSON example validity rule: examples use a capital option letter answer and a three-role point-map annotation.
7. Variant counts: 5 scene templates, 5 task templates, 5 output-mode templates.
8. Output modes: `answer_only`, `answer_and_annotation`

## 4) Determinism + constraints
1. Seed namespaces used: task dataset, scene variant, prompt variants, and symbolic background/noise namespaces.
2. Unique-answer policy: all 64 terminal labels are unique; six option labels are unique; exactly one option contains the reached output label.
3. Reject/resample conditions: invalid code length/symbols, duplicate terminal labels, unsupported option count, or insufficient distractors.
4. No-auto-relaxation guarantee: invalid constraints raise and retry with the next deterministic retry seed.

## 5) Tests
1. Determinism test: `tests/test_symbolic_radial_code_wheel_tasks.py`
2. Answer/annotation consistency test: same focused test file validates code-to-terminal output and annotation roles.
3. Prompt metadata/placeholder test: same focused test file validates prompt bundle keys and required slots.
4. Constraint-specific tests: same focused test file validates unique terminal/output option construction.
