# `task_geometry__survey_traverse__traverse_area_value`

## Contract
1. Domain: `geometry`
2. Scene id: `survey_traverse`
3. Task id: `task_geometry__survey_traverse__traverse_area_value`
4. Supported public `query_id`: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox_map`

## Program Contract
- `survey_traverse_area_value(visible_traverse_shape, visible_field_note, branch=offset_trapezoid_area) -> enclosed_area; scene=survey_traverse; scope=traverse_area_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the v1 scene prompt bundle configured for `survey_traverse`.
- Prompt modes: `answer_only` and `answer_and_annotation`.
- Internal trace branch: `offset_trapezoid_area`.

## Annotation
Prompt-facing annotation uses pixel-space box witnesses. Map annotation binds the visual area witnesses:

- `traverse_region`
- `field_note_region`
Chainages, offsets, station labels, and field-note text remain visible diagram content plus private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/survey_traverse.yaml`
- Task module: `src/trace_tasks/tasks/geometry/survey_traverse/traverse_area_value.py`
