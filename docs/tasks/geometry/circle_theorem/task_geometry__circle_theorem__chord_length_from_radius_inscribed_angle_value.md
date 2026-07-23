# `task_geometry__circle_theorem__chord_length_from_radius_inscribed_angle_value`

## Contract
1. Domain: `geometry`
2. Scene id: `circle_theorem`
3. Query id: `single`
4. Answer schema: `number`
5. Answer precision: `one_decimal`
6. Annotation schema: `point_map`

## Program Contract
- `solve_formula(visible_circle_radius_and_inscribed_angle, unknown_role=chord_length, formula_schema=inscribed_angle_to_central_angle_then_chord_length); scene=circle_theorem; scope=chord_length_from_radius_inscribed_angle_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the geometry circle prompt bundle configured for this scene/task override.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses keyed pixel points for `O`, `A`, `B`, and `C`, where `O` is the circle center, `A`/`B` are the chord endpoints, and `C` is the inscribed-angle vertex. Radius labels, angle labels, and the `?` chord cue are visible annotations plus verifier metadata, not separate annotation objects.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/circle_theorem.yaml`
- Task module: `src/trace_tasks/tasks/geometry/circle_theorem/chord_length_from_radius_inscribed_angle_value.py`
