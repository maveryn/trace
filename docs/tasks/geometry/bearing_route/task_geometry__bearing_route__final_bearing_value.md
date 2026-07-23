# `task_geometry__bearing_route__final_bearing_value`

## Contract
1. Domain: `geometry`
2. Scene id: `bearing_route`
5. Query id: `single`
6. Answer schema: `option_letter`
7. Annotation schema: `point_map`

## Program Contract
- `solve_formula_then_select_visual_option(visible_bearing_route_measurements, visible_bearing_options, unknown_role=direct_bearing_option, formula_schema=final_bearing_value); scene=bearing_route; scope=final_bearing_value`

## Reasoning Operations

Families: `spatial_relations`, `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from the geometry prompt bundle configured for this scene/task override.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses pixel-space witnesses only. Map annotation is used where witness roles matter; graph coordinates, formulas, labels, and construction metadata remain private verifier metadata unless they are themselves visual witnesses.
The six visible options are answer choices, not annotation witnesses; annotation marks only the start and finish points needed to ground the direct-bearing relation.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/bearing_route.yaml`
- Task module: `src/trace_tasks/tasks/geometry/bearing_route/final_bearing_value.py`
