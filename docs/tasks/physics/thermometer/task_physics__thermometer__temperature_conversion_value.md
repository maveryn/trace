# `task_physics__thermometer__temperature_conversion_value`

## Summary
- Domain: `physics`
- Scene id: `thermometer`
- Implementation scene: `thermometer`
- Implementation source: `src/trace_tasks/tasks/physics/thermometer/temperature_conversion_value.py`

## Task Contract
Reads a source temperature from a visible thermometer scale and converts it to the other temperature unit.

## Program Contract

Program: `integer(convert_temperature(read_scale_value(liquid_level, thermometer_scale, source_unit), source_unit, target_unit)); scene=thermometer; scope=temperature_conversion_value`

Candidate set: the visible thermometer tube, liquid level, numeric tick scale, and source unit label inside the `temperature_conversion_value` objective scope.
Operands: `liquid_level` (semantic_role, allowed `visible_liquid_column_top`, source `program_schema_concrete`); `scale_region` (query_operand, allowed `visible_tick_scale_with_numeric_labels`, source `program_schema_concrete`); `source_unit_label` (query_operand, allowed `C|F`, source `program_schema_concrete`); active `query_id` branch when present.
Operation: evaluate `integer` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; The answer value is the integer converted temperature in the target unit requested by the prompt.
Annotation witnesses: `segment` witnesses from the finalized render. Annotation is the visible liquid-level segment on the thermometer tube. Annotation must not mark the whole scale, source-unit label, derived answer text, hidden conversion result, or decorative thermometer casing alone.
Query ids: `celsius_to_fahrenheit_value`, `fahrenheit_to_celsius_value`.

## Reasoning Operations

Families: `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `celsius_to_fahrenheit_value` | `integer(convert_temperature(read_scale_value(liquid_level, thermometer_scale, unit=C), C, F)); scene=thermometer; scope=temperature_conversion_value` |
| `fahrenheit_to_celsius_value` | `integer(convert_temperature(read_scale_value(liquid_level, thermometer_scale, unit=F), F, C)); scene=thermometer; scope=temperature_conversion_value` |

## Program Metadata
- Program signatures: `physics.thermometer_temperature_conversion`
- Base program contract: `integer(convert_temperature(read_scale_value(liquid_level, thermometer_scale, source_unit), source_unit, target_unit)); scene=thermometer; scope=temperature_conversion_value`
- Parameter axes: `query_id`, `scale_profile`, `source_temperature`, `target_answer`
- Arguments:
  - `liquid_level`: semantic_role; allowed `visible_liquid_column_top`; source `program_schema_concrete`
  - `scale_region`: query_operand; allowed `visible_tick_scale_with_numeric_labels`; source `program_schema_concrete`
  - `source_unit_label`: query_operand; allowed `C|F`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `celsius_to_fahrenheit_value`, `fahrenheit_to_celsius_value`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer value is the integer converted temperature in the target unit requested by the prompt.

## Annotation Contract
- Annotation schema: `segment`
- Generator `annotation_gt.type`: `segment`
- Annotation is the visible liquid-level segment on the thermometer tube.
- Annotation must not mark the whole scale, source-unit label, derived answer text, hidden conversion result, or decorative thermometer casing alone.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the physics thermometer prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- The prompt must include the relevant temperature-conversion formula so the task does not depend on hidden unit-conversion assumptions.
- Render randomness, sampled fonts/styles, query id, scale profile, source temperature, source unit, target unit, target answer, and verifier payloads must be explicit in the instance trace.
- First-version diagrams keep source temperatures tick-aligned and target answers integer-valued.
