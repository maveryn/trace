# `task_geometry__rectangular_solid__cube_edge_from_frame_length_value`

## Contract
1. Domain: `geometry`
2. Scene id: `rectangular_solid`
3. Task id: `task_geometry__rectangular_solid__cube_edge_from_frame_length_value`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_value`
6. Annotation schema: `bbox`
7. Scalar annotation checked: `true` (exactly one highlighted frame-region witness)

## Program Contract
- `solve_formula(cube_wire_frame_length_measurement, unknown_role=cube_edge, formula_schema=edge_length_from_frame_edge_count); scene=rectangular_solid; scope=cube_edge_from_frame_length_value`

## Reasoning Operations

Families: `counting`, `formula_evaluation`

## Query Semantics
- `single` asks for the cube edge length from the bright highlighted portion of the cube frame; pale frame edges are visual context only.
- The sampled edge length, highlighted path, highlighted edge count, style, font, and layout jitter are internal replay metadata.

## Prompt Bundle
- Prompt text is loaded from `src/trace_tasks/resources/prompts/geometry/rectangular_solid/geometry_rectangular_solid_v1.json`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is the scalar pixel-space bounding box around the highlighted cube-frame path. Frame-length readouts, edge labels, and the `?` marker remain visible diagram content and private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/rectangular_solid.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/geometry/rectangular_solid/geometry_rectangular_solid_v1.json`
- Task module: `src/trace_tasks/tasks/geometry/rectangular_solid/cube_edge_from_frame_length_value.py`
