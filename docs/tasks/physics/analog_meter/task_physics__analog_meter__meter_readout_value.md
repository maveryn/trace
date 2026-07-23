# `task_physics__analog_meter__meter_readout_value`

## Summary
- Domain: `physics`
- Scene id: `analog_meter`
- Implementation scene: `analog_meter`
- Implementation source: `src/trace_tasks/tasks/physics/analog_meter/meter_readout_value.py`

## Task Contract
Reads the integer value shown by an analog meter needle from the visible tick scale and unit label.

## Program Contract

Program: `integer(read_analog_meter(needle_position, tick_scale, displayed_unit)); scene=analog_meter; scope=meter_readout_value`

Candidate set: the visible meter needle, tick scale, numeric labels, and unit label inside the `meter_readout_value` objective scope.
Operands: `needle` (semantic_role, allowed `visible_meter_needle`, source `program_schema_concrete`); `scale_region` (query_operand, allowed `visible_tick_scale_with_numeric_labels`, source `program_schema_concrete`); `unit_label` (query_operand, allowed `A|mA|V`, source `program_schema_concrete`); active `query_id` branch when present.
Operation: evaluate `integer` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; The answer value is the integer readout in the displayed unit.
Annotation witnesses: `segment` witnesses from the finalized render. Annotation marks the visible meter needle as one pixel segment from tail to tip. Scale ticks and the unit label remain visible context in the image and trace metadata, but they are not public annotation witnesses.
Query ids: `ammeter_readout`, `voltmeter_readout`.

## Reasoning Operations

Families: `direct_retrieval`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `ammeter_readout` | `integer(read_analog_meter(needle, scale_region, unit_label)); scene=analog_meter; meter=ammeter; scope=meter_readout_value` |
| `voltmeter_readout` | `integer(read_analog_meter(needle, scale_region, unit_label)); scene=analog_meter; meter=voltmeter; scope=meter_readout_value` |

## Program Metadata
- Program signatures: `physics.analog_meter_readout`
- Base program contract: `integer(read_analog_meter(needle_position, tick_scale, displayed_unit)); scene=analog_meter; scope=meter_readout_value`
- Parameter axes: `query_id`, `meter_profile`, `readout_value`
- Arguments:
  - `needle`: semantic_role; allowed `visible_meter_needle`; source `program_schema_concrete`
  - `scale_region`: query_operand; allowed `visible_tick_scale_with_numeric_labels`; source `program_schema_concrete`
  - `unit_label`: query_operand; allowed `A|mA|V`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported `query_id`s: `ammeter_readout`, `voltmeter_readout`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer value is the integer readout in the displayed unit.

## Annotation Contract
- Annotation schema: `segment`
- Generator `annotation_gt.type`: `segment`
- Annotation marks the visible meter needle as one pixel segment from tail to tip.
- Scale ticks and the unit label remain visible context in the image and trace metadata, but they are not public annotation witnesses.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the physics analog-meter v1 prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query id, meter profile, readout value, needle angle, unit, and verifier payloads must be explicit in the instance trace.
- First-version diagrams must use conventional clockwise left-to-right analog scales and integer tick-aligned needle positions.
