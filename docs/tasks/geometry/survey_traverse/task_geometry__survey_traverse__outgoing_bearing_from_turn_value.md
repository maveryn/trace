# `task_geometry__survey_traverse__outgoing_bearing_from_turn_value`

## Contract
1. Domain: `geometry`
2. Scene id: `survey_traverse`
3. Task id: `task_geometry__survey_traverse__outgoing_bearing_from_turn_value`
4. Supported `query_id`: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox_map`

## Program Contract
- `survey_outgoing_bearing_from_turn(visible_station_line, visible_north_reference, incoming_bearing, turn_angle, turn_direction=left|right) -> outgoing_bearing_degrees; scene=survey_traverse; scope=outgoing_bearing_from_turn_value`

## Reasoning Operations

Families: `spatial_relations`, `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the v1 scene prompt bundle configured for `survey_traverse`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space bbox witnesses. Map annotation binds the visible turn diagram and field-note region:

- `turn_diagram`
- `field_note_region`

Numeric bearing labels, turn labels, station labels, and field-note text remain visible annotations plus private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/survey_traverse.yaml`
- Task module: `src/trace_tasks/tasks/geometry/survey_traverse/outgoing_bearing_from_turn_value.py`
