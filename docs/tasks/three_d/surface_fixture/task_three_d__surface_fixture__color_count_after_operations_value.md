# `task_three_d__surface_fixture__color_count_after_operations_value`

## Summary
- Domain: `three_d`
- Scene id: `surface_fixture`
- Scene package: `surface_fixture`
- Query id: `single`
- Answer type: `integer`
- Annotation type: unordered `bbox_set`
- Annotation schema: `bbox_set`

## Program Contract

Program: `initial_count(filter(surface_fixture_elements, present=true, element_type=target_element_type, color_name=target_color_name)) + sum(signed_count(operation) for operation in operations if operation.color_name=target_color_name); scene=surface_fixture; scope=color_count_after_operations_value`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `color_count_after_operations_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `surface_fixture_elements`, `present`, `true`, `element_type`, `target_element_type`, `color_name`, `target_color_name`, `sum`, `signed_count`, `operation`, `operations`.
Operation: evaluate `initial_count` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `aggregation`, `state_update`, `formula_evaluation`

## Contract
The image shows one projected fixture surface with repeated colored elements.
The prompt gives exactly three hypothetical add/remove operations over the same
element family and asks for the final number of elements of the requested target
color after all operations are applied.
Generated named-color instances use readout-safe fixture variants and avoid
near-color distractors among the active semantic colors.

The answer is the integer final count:

1. Count original visible present cells whose
   `element_type == target_element_type` and
   `color_name == target_color_name`.
2. Add counts from hypothetical target-color add operations.
3. Subtract counts from hypothetical target-color remove operations.

Operations for non-target colors are distractors and do not change the answer.
Generation guarantees the target-color count changes and remains nonnegative.

## Annotation Contract
Annotation is a `bbox_set` containing one `[x0, y0, x1, y1]` pixel box around
each original visible target-color surface element used as the starting count.
Hypothetical added elements, removed elements after the text operations,
non-target colors, the fixture panel, and decorative context are not
annotation.

## Prompt And Trace
The prompt bundle is `three_d_surface_fixture_v1` under
`src/trace_tasks/resources/prompts/three_d/surface_fixture/`. The trace records scene variant, target
element type, target color, active colors, initial color counts, operation list,
final color counts, explicit cell metadata, projected element boxes and
centers, and the solver count predicate.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config
defaults, prompt bundle, and code versions. Answers and annotation come from the
same finalized fixture trace.
