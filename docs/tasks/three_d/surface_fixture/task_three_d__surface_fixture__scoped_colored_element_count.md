# `task_three_d__surface_fixture__scoped_colored_element_count`

## Summary
- Domain: `three_d`
- Scene id: `surface_fixture`
- Scene package: `surface_fixture`
- Query ids: `row_scoped_color_count`, `column_scoped_color_count`
- Answer type: `integer`
- Annotation type: unordered `bbox_set`
- Annotation schema: `bbox_set`

## Program Contract

Program: `count(filter(surface_fixture_elements, present=true, scope_axis=scope_axis, scope_index=scope_index, color_name=target_color_name)); scene=surface_fixture; scope=scoped_colored_element_count`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `scoped_colored_element_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `surface_fixture_elements`, `present`, `true`, `scope_axis`, `scope_index`, `color_name`, `target_color_name`, `surface_fixture`, `scoped_colored_element_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `row_scoped_color_count`, `column_scoped_color_count`.

## Reasoning Operations

Families: `filtering`, `counting`

## Contract
The image shows one projected fixture surface arranged in rows and columns with
repeated colored elements. The prompt asks for the number of elements of the
requested color within one sampled row or column.
Generated named-color instances use readout-safe fixture variants and avoid
near-color distractors for the requested semantic color.

The answer is the integer count of finalized present cells matching both the
requested scope and `color_name == target_color_name`.
The `row_scoped_color_count` query binds `scope_axis=row`; the
`column_scoped_color_count` query binds `scope_axis=column`.

## Annotation Contract
Annotation is a `bbox_set` containing one `[x0, y0, x1, y1]` pixel box around
each counted colored surface element in the requested row or column.
Same-color elements outside the scope, other colors, the fixture panel, and
decorative context are not annotation.

## Prompt And Trace
The prompt bundle is `three_d_surface_fixture_v1` under `src/trace_tasks/resources/prompts/three_d/surface_fixture/`.
The trace records scene variant, target element type, target color, scope axis,
scope index, explicit cell metadata, projected element boxes and centers, and
the solver count predicate.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config
defaults, prompt bundle, and code versions. Answers and annotation come from the
same finalized fixture trace.
