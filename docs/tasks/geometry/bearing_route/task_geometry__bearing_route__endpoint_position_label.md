# `task_geometry__bearing_route__endpoint_position_label`

## Contract
1. Domain: `geometry`
2. Scene id: `bearing_route`
5. Query id: `single`
6. Answer schema: `option_letter`
7. Annotation schema: `point_map`

## Program Contract
- `follow_cardinal_bearing_steps_on_graph_paper(route_steps, candidate_endpoint_labels, unknown_role=selected_label); scene=bearing_route; scope=endpoint_position_label`
- The candidate panel is rendered as graph paper; each square is one visible step.
- Route instructions show bearing plus step count, and candidate endpoints sit on grid intersections.

## Reasoning Operations

Families: `spatial_relations`, `state_update`

## Prompt Bundle
- Prompt text is loaded from the geometry prompt bundle configured for this scene/task override.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space witnesses only. Map annotation is used where witness roles matter; graph coordinates, formulas, labels, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/bearing_route.yaml`
- Task module: `src/trace_tasks/tasks/geometry/bearing_route/endpoint_position_label.py`
