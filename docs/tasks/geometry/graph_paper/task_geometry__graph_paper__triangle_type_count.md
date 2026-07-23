# `task_geometry__graph_paper__triangle_type_count`

## Contract
1. Domain: `geometry`
2. Scene id: `graph_paper`
3. Task id: `task_geometry__graph_paper__triangle_type_count`
4. Supported `query_id`: `equilateral_triangle_count`, `right_triangle_count`, `scalene_triangle_count`, `non_equilateral_isosceles_triangle_count`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`

## Program Contract
- `count_triangle_class(query_id={equilateral_triangle_count|right_triangle_count|scalene_triangle_count|non_equilateral_isosceles_triangle_count}, target_class={equilateral|right|scalene|non_equilateral_isosceles}, output_role=count); scene=graph_paper; scope=triangle_set`
- Matching uses standard triangle predicates from the rendered vertices. In particular, `scalene_triangle_count` includes any right triangle whose three side lengths are unequal, while `non_equilateral_isosceles_triangle_count` excludes equilateral triangles.
- Non-equilateral classes use integer graph-paper vertices. Exact equilateral triangles keep a graph-paper-aligned base but are the square-lattice exception, since a nondegenerate exact equilateral triangle cannot place all vertices on square grid intersections.
- Triangles use class-preserving variation in scale, orientation, and side ratios; repeated instances of the same class are not required to be congruent.
- Count objects are placed using their actual graph-unit bounds so independent triangles do not overlap.

## Reasoning Operations

Families: `filtering`, `counting`

## Prompt Bundle
- Prompt text is loaded from `geometry_graph_paper_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
The annotation is the unordered set of bounding boxes for every matching triangle.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.
