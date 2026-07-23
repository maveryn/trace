# `task_physics__thermal_mixing__final_temperature_value`

## Summary
- Domain: `physics`
- Scene id: `thermal_mixing`
- Implementation source: `src/trace_tasks/tasks/physics/thermal_mixing/final_temperature_value.py`

## Task Contract
Computes the final equilibrium temperature after equal amounts of the same liquid are mixed in an insulated container.

## Program Contract

Program: `integer(average(initial_temperatures_same_liquid_equal_amounts_insulated_system)); scene=thermal_mixing; scope=final_temperature_value`

Candidate set: the visible containers, initial temperature labels, masses or mixture labels, and final-state cue inside the `final_temperature_value` objective scope.
Operands: `initial_temperatures` (visual_operand_set, allowed `2..4 visible Celsius cup labels`, source `program_schema_concrete`); `same_liquid` (semantic_condition, allowed `true`, source `prompt_and_scene_contract`); `equal_amounts` (semantic_condition, allowed `true`, source `prompt_and_scene_contract`); `insulated_system` (semantic_condition, allowed `true`, source `prompt_and_scene_contract`).
Operation: evaluate `integer` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; The answer value is the integer final equilibrium temperature in degrees Celsius.
Annotation witnesses: `bbox_set` witnesses from the finalized render. Annotation is an unordered set of bounding boxes around the visible initial temperature labels used in the average. Annotation must not mark the cup bodies, final mixing container, hidden derived final temperature, decorative arrows, title text, background grid, or prompt-only assumptions.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `integer(average(initial_temperatures_same_liquid_equal_amounts_insulated_system)); scene=thermal_mixing; scope=final_temperature_value` |

## Program Metadata
- Program signatures: `physics.final_temperature_value`
- Base program contract: `integer(average(initial_temperatures_same_liquid_equal_amounts_insulated_system)); scene=thermal_mixing; scope=final_temperature_value`
- Parameter axes: `cup_count`, `initial_temperature_set`, `final_temperature_c`
- Arguments:
  - `initial_temperatures`: visual_operand_set; allowed `2..4 visible Celsius cup labels`; source `program_schema_concrete`
  - `same_liquid`: semantic_condition; allowed `true`; source `prompt_and_scene_contract`
  - `equal_amounts`: semantic_condition; allowed `true`; source `prompt_and_scene_contract`
  - `insulated_system`: semantic_condition; allowed `true`; source `prompt_and_scene_contract`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer value is the integer final equilibrium temperature in degrees Celsius.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation is an unordered set of bounding boxes around the visible initial temperature labels used in the average.
- Annotation must not mark the cup bodies, final mixing container, hidden derived final temperature, decorative arrows, title text, background grid, or prompt-only assumptions.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the `physics_thermal_mixing_v1` prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, cup count, initial temperature set, final average, and verifier payloads must be explicit in the instance trace.
- Initial temperatures must be constructed so their average is an integer. Cup order, liquid color, cup count, and layout must remain non-semantic visual variation axes.
