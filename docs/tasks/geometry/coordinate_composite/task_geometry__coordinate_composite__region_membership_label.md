# `task_geometry__coordinate_composite__region_membership_label`

## Contract
1. Domain: `geometry`
2. Scene id: `coordinate_composite`
3. Query id: `inside_circle_outside_polygon`, `inside_polygon_outside_circle`, `inside_circle_above_line`, or `inside_polygon_below_line`
4. Answer schema: `option_letter`
5. Annotation schema: `point`

## Program Contract
- `select(candidate_point where region_predicate(candidate_point, drawn_objects)); scene=coordinate_composite; scope=region_membership_label`

## Reasoning Operations

Families: `spatial_relations`

## Prompt Bundle
- Prompt text is loaded from `geometry_coordinate_composite_v0`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is the pixel point `[x,y]` for the selected marked candidate point. The image shows four marked candidate points labeled `A` through `D`. Candidate labels, object outlines, graph-paper grid, and symbolic object definitions remain visible annotations plus private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/coordinate_composite.yaml`
- Task module: `src/trace_tasks/tasks/geometry/coordinate_composite/region_membership_label.py`
