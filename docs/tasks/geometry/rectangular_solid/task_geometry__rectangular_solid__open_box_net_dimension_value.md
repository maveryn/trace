# `task_geometry__rectangular_solid__open_box_net_dimension_value`

## Contract
1. Domain: `geometry`
2. Scene id: `rectangular_solid`
3. Task id: `task_geometry__rectangular_solid__open_box_net_dimension_value`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_value`
6. Annotation schema: `segment`
7. Scalar annotation checked: `true` (exactly one target base-dimension segment witness)

## Program Contract
- `solve_formula(corner_cut_open_box_net, unknown_role=base_dimension, formula_schema=open_box_corner_cut_dimensions); scene=rectangular_solid; scope=open_box_net_dimension_value`

## Reasoning Operations

Families: `formula_evaluation`

## Query Semantics
- `single` asks for one marked resulting base dimension after equal corner squares are removed and the sides fold up.
- The sampled target base dimension role, sheet dimensions, cut size, style, font, and layout jitter are internal replay metadata.

## Prompt Bundle
- Prompt text is loaded from `src/trace_tasks/resources/prompts/geometry/rectangular_solid/geometry_rectangular_solid_v1.json`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is the scalar pixel-space segment marking the requested resulting base dimension. Sheet labels, cut labels, hatching, and the `?` marker remain visible diagram content and private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/rectangular_solid.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/geometry/rectangular_solid/geometry_rectangular_solid_v1.json`
- Task module: `src/trace_tasks/tasks/geometry/rectangular_solid/open_box_net_dimension_value.py`
