# `task_geometry__polygon_equation_diagram__equal_side_length_value`

## Contract
1. Domain: `geometry`
2. Scene id: `polygon_equation_diagram`
3. Supported `query_id`: `single`
4. Answer schema: `integer`
5. Annotation schema: `point_map`

## Program Contract
- `solve_formula(visible_equal_side_polygon_equation, unknown_role=target_side_length, formula_schema=equal_side_expression_side_length_value); scene=polygon_equation_diagram; scope=equal_side_length_value`

## Reasoning Operations

Families: `formula_evaluation`

## Internal Construction Families
The public task has no semantic query branch. The sampled polygon side count is recorded as trace metadata:

- `triangle`
- `quadrilateral`
- `pentagon`
- `hexagon`

## Prompt Bundle
- Prompt text is loaded from `geometry_polygon_equation_diagram_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses a `point_map` keyed by the visible polygon vertex labels, such as `A`, `B`, `C`, and any additional visible vertices. Each value is that labeled vertex's pixel coordinate after final layout and rotation.

The target equal-side pair uses matching two-tick side marks. The diagram may also include non-target
side-expression distractors with one-tick or three-tick marks; those distractors are trace metadata, not
semantic query branches.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/polygon_equation_diagram.yaml`
- Task module: `src/trace_tasks/tasks/geometry/polygon_equation_diagram/equal_side_length_value.py`
