# `task_symbolic__radial_code_wheel__missing_code_symbol_label`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `radial_code_wheel`
3. Task id: `task_symbolic__radial_code_wheel__missing_code_symbol_label`
4. Objective contract: identify the missing symbol in a three-symbol radial code.

## Program Contract
Program: `radial_code_wheel.missing_code_symbol(scene=radial_code_wheel, scope=target_output_and_incomplete_code, output=code_symbol)`

Candidate set: the four code symbols `A|B|C|D`.
Operands: the target output label, the incomplete three-symbol code with one `?`, and the radial code wheel mapping from code symbols to terminal output labels.
Operation: find the code path that reaches the target output label, then identify the symbol that belongs in the `?` position of the shown code.
Output binding: `answer` is the missing code symbol.
Annotation witnesses: one `point` marking the matching symbol center on the missing symbol's ring.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `matching`

## 2) Scene + task contract
1. Entities/relations:
   - one three-ring radial code wheel with symbols `A|B|C|D`;
   - one visible target output card;
   - one visible incomplete code card with exactly one `?`;
   - 64 unique terminal output labels on the wheel.
2. Supported `query_id` values: `single`
3. `answer_gt.type`: `string`
4. Default `annotation_gt.type`: `point`
5. Annotation schema: `point`
6. Alternate annotation forms: none
7. Annotation witness policy:
   - minimal object/primitive witness: the center point of the missing code symbol on the corresponding wheel ring.
   - annotation shape choice: `point`, because the answer witness is one compact ring symbol.
   - numeric/readout annotation handling: not applicable.
   - answer-option annotation policy: the selected symbol card is not annotated; the decisive witness is the ring symbol that completes the code path.
8. Overlap/touch policy: target output card, incomplete code card, and terminal labels are rendered from fixed non-overlapping layout slots.

## 3) Prompt contract
1. `prompt_bundle_id`: `symbolic_radial_code_wheel_v1`
2. `scene_key`: `radial_code_wheel`
3. `task_key`: `missing_code_symbol_label`
4. Optional query-id prompt mapping: none
5. Required slots:
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
6. JSON example validity rule: examples use one code-symbol answer and one point annotation.
7. Variant counts: 5 scene templates, 5 task templates, 5 output-mode templates.
8. Output modes: `answer_only`, `answer_and_annotation`

## 4) Determinism + constraints
1. Seed namespaces used: task dataset, scene variant, prompt variants, and symbolic background/noise namespaces.
2. Unique-answer policy: the target output label maps to exactly one terminal code; the incomplete code has exactly one missing position.
3. Reject/resample conditions: invalid code length/symbols, duplicate terminal labels, unsupported missing position, or empty missing-position support.
4. No-auto-relaxation guarantee: invalid constraints raise and retry with the next deterministic retry seed.

## 5) Tests
1. Registration/taxonomy test: `tests/test_symbolic_radial_code_wheel_tasks.py`
2. Answer/annotation consistency test: same focused test file validates the missing symbol, target output, and point annotation.
3. Prompt metadata/placeholder test: same focused test file validates prompt bundle keys and required slots.
4. Config-default test: `tests/test_symbolic_core_scene_config.py`
