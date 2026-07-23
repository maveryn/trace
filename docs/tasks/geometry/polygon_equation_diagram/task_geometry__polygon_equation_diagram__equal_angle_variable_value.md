# `task_geometry__polygon_equation_diagram__equal_angle_variable_value`

## Contract
1. Domain: `geometry`
2. Scene id: `polygon_equation_diagram`
3. Supported `query_id`: `single`
4. Answer schema: `integer`
5. Annotation schema: `point_map`

## Program Contract
- `solve_formula(visible_equal_angle_polygon_equation, unknown_role=variable_value, formula_schema=equal_angle_expression_variable_value); scene=polygon_equation_diagram; scope=equal_angle_variable_value`

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

The target equal-angle pair uses matching two-arc angle marks. This task does not add non-target
angle-expression distractors; distractor sampling is reserved for sibling equal-angle/side objectives.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/polygon_equation_diagram.yaml`
- Task module: `src/trace_tasks/tasks/geometry/polygon_equation_diagram/equal_angle_variable_value.py`
