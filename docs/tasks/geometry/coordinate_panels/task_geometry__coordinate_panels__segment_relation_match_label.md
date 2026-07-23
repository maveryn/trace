# `task_geometry__coordinate_panels__segment_relation_match_label`

## Contract
1. Domain: `geometry`
2. Scene id: `coordinate_panels`
5. Query id: `equal_length_segments_match_label`, `parallel_segments_match_label`, `perpendicular_segments_match_label`
6. Answer schema: `option_letter`
7. Annotation schema: `segment_set`

## Program Contract
- `label(select_panel(candidate_coordinate_panels_6, segment_relation)); scene=coordinate_panels; scope=segment_relation_match_label`

## Reasoning Operations

Families: `spatial_relations`, `matching`

## Prompt Bundle
- Prompt text is loaded from the scene prompt bundle configured for `coordinate_panels`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation is the two pixel-space line segments in the selected panel. Each segment is `[[x0,y0],[x1,y1]]`; endpoint order is not semantically meaningful. Panel boxes, graph coordinates, relation flags, and construction metadata remain private verifier metadata.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/coordinate_panels.yaml`
- Task module: `src/trace_tasks/tasks/geometry/coordinate_panels/segment_relation_match_label.py`
