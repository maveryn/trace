# `task_physics__bulb_circuit__brightness_extremum_label`

## Summary
- Domain: `physics`
- Scene id: `bulb_circuit`
- Implementation scene: `bulb_circuit`
- Implementation source: `src/trace_tasks/tasks/physics/bulb_circuit/brightness_extremum_label.py`

## Task Contract
Selects the labeled bulb that is brightest or dimmest in a visible ideal-battery circuit.

## Program Contract

Program: `label(arg_extreme(bulbs, power_in_visible_circuit, direction=brightest_or_dimmest)); scene=bulb_circuit; scope=brightness_extremum_label`

Candidate set: the visible bulb symbols, wire topology, switch states, and component labels inside the `brightness_extremum_label` objective scope.
Operands: `bulbs` (semantic_role, allowed `visible_labeled_bulbs_b1_through_b5_with_resistance_labels`, source `program_schema_concrete`); `circuit_topology` (semantic_role, allowed `visible_series_parallel_bulb_topology`, source `program_schema_concrete`); `target_direction` (query_operand, allowed `brightest|dimmest`, source `query_id`); active `query_id` branch when present.
Operation: evaluate `label` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; The answer value is the selected visible bulb label, for example `B2`.
Annotation witnesses: `bbox` witnesses from the finalized render. Annotation marks one bounding box around the selected answer bulb symbol and its resistance label. Other bulb boxes remain visible context in the image and trace metadata, but they are not public annotation witnesses.
Query ids: `brightest_bulb_label`, `dimmest_bulb_label`.

## Reasoning Operations

Families: `ranking`, `topology`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `brightest_bulb_label` | `label(arg_extreme(bulbs, power_in_visible_circuit, direction=brightest)); scene=bulb_circuit; scope=brightness_extremum_label; query_branch=brightest_bulb_label` |
| `dimmest_bulb_label` | `label(arg_extreme(bulbs, power_in_visible_circuit, direction=dimmest)); scene=bulb_circuit; scope=brightness_extremum_label; query_branch=dimmest_bulb_label` |

## Program Metadata
- Program signatures: `physics.bulb_brightness_extremum_label`
- Base program contract: `label(arg_extreme(bulbs, power_in_visible_circuit, direction=brightest_or_dimmest)); scene=bulb_circuit; scope=brightness_extremum_label`
- Parameter axes: `query_id`, `scene_variant`
- Arguments:
  - `bulbs`: semantic_role; allowed `visible_labeled_bulbs_b1_through_b5_with_resistance_labels`; source `program_schema_concrete`
  - `circuit_topology`: semantic_role; allowed `visible_series_parallel_bulb_topology`; source `program_schema_concrete`
  - `target_direction`: query_operand; allowed `brightest|dimmest`; source `query_id`
- Argument metadata status: `curated`
- Supported `query_id`s: `brightest_bulb_label`, `dimmest_bulb_label`

## Answer Contract
- Answer schema: `string`
- Generator `answer_gt.type`: `string`
- The answer value is the selected visible bulb label, for example `B2`.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation marks one bounding box around the selected answer bulb symbol and its resistance label.
- Other bulb boxes remain visible context in the image and trace metadata, but they are not public annotation witnesses.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the physics prompt bundles, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, visible resistance values, computed powers, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep the circuit topology, bulb labels, resistance labels, and ideal battery visible; glow intensity must not encode the answer.
