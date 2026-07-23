# `task_geometry__survey_traverse__station_elevation_value`

## Contract
1. Domain: `geometry`
2. Scene id: `survey_traverse`
3. Task id: `task_geometry__survey_traverse__station_elevation_value`
4. Supported public `query_id`: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox_map`

## Program Contract
- `survey_station_elevation_value(visible_station_profile, visible_field_note, branch=leveling_station_elevation) -> target_station_elevation; scene=survey_traverse; scope=station_elevation_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the v1 scene prompt bundle configured for `survey_traverse`.
- Prompt modes: `answer_only` and `answer_and_annotation`.
- Internal trace branch: `leveling_station_elevation`.

## Annotation
Prompt-facing annotation uses pixel-space bbox witnesses. Map annotation binds the visible station profile and field-note region:

- `station_profile`
- `field_note_region`

Numeric elevations, staff readings, station labels, and field-note text remain visible diagram content plus private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/survey_traverse.yaml`
- Task module: `src/trace_tasks/tasks/geometry/survey_traverse/station_elevation_value.py`
