# `task_geometry__coordinate_plane__locus_point_label`

## Contract
1. Domain: `geometry`
2. Scene id: `coordinate_plane`
5. Query id: `circle_region_point`, `annulus_region_point`, `vertical_strip_region_point`, `half_plane_intersection_region_point`
6. Answer schema: `option_letter`
7. Annotation schema: `point`

## Program Contract
- `label(select_point(lettered_candidate_points, predicate=inside_shaded_region, region_rule_family)); scene=coordinate_plane; scope=locus_point_label`

## Reasoning Operations

Families: `spatial_relations`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `coordinate_plane`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is the scalar pixel point `[x,y]` at the center of the selected candidate point. Graph coordinates, formulas, labels, and construction metadata remain private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/coordinate_plane.yaml`
- Task module: `src/trace_tasks/tasks/geometry/coordinate_plane/locus_point_label.py`
