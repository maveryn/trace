# `task_geometry__circle_centerline_overlap__segment_length_value`

## Contract
1. Domain: `geometry`
2. Scene id: `circle_centerline_overlap`
4. Query id: `center_distance_from_overlap` or `boundary_segment_from_overlap`
5. Answer schema: `integer_value`
6. Annotation schema: `segment`

## Program Contract
- `solve_formula(visible_collinear_circle_overlap_measurements, unknown_role=target_centerline_segment, formula_schema=circle_centerline_overlap_segment_length); scene=circle_centerline_overlap; scope=segment_length_value`

## Reasoning Operations

Families: `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_circle_centerline_overlap_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
Prompt-facing annotation uses one pixel-space segment for the requested target segment only: `[[x0,y0],[x1,y1]]`. For example, if the prompt asks for `AB` or `AC`, the segment endpoints are the visible center points. If it asks for a boundary segment such as `AP` or `QB`, the endpoints are the visible center/boundary points named in the target segment. Numeric labels, point labels, radius readouts, and known segments remain visible context plus private verifier metadata.

## Sampling
Default generation samples two-circle chains 75% of the time and three-circle chains 25% of the time as internal construction metadata. It samples circle radii and adjacent overlap lengths from deterministic constrained integer ranges instead of a small fixed case bank. The constraints preserve proper adjacent circle overlaps, keep non-adjacent circles separated for three-circle chains, and ensure boundary-segment answers remain at least 3. Boundary labels are left-to-right along the centerline: `P,Q` for the first overlap and `R,S` for the second overlap when present. Circle measurement labels are radius labels.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/circle_centerline_overlap.yaml`
- Task module: `src/trace_tasks/tasks/geometry/circle_centerline_overlap/segment_length_value.py`
