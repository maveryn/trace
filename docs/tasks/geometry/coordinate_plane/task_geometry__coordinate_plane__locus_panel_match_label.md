# `task_geometry__coordinate_plane__locus_panel_match_label`

## Contract
1. Domain: `geometry`
2. Scene id: `coordinate_plane`
5. Query id: `circle_inequality_panel_match`, `vertical_strip_panel_match`, `horizontal_halfplane_panel_match`, `two_inequality_panel_match`
6. Answer schema: `option_letter`
7. Annotation schema: `bbox`

## Program Contract
- `label(select_panel(candidate_region_panels, condition_box, region_rule_family)); scene=coordinate_plane; scope=locus_panel_match_label`

## Reasoning Operations

Families: `comparison`, `logical_composition`, `spatial_relations`, `matching`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `coordinate_plane`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is the scalar pixel bounding box `[x0,y0,x1,y1]` around the selected panel. Graph coordinates, formulas, labels, and construction metadata remain private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/coordinate_plane.yaml`
- Task module: `src/trace_tasks/tasks/geometry/coordinate_plane/locus_panel_match_label.py`
