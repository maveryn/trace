# `task_three_d__surface_fixture__colored_element_count`

## Summary
- Domain: `three_d`
- Scene id: `surface_fixture`
- Scene package: `surface_fixture`
- Query id: `single`
- Answer type: `integer`
- Annotation type: unordered `bbox_set`
- Annotation schema: `bbox_set`

## Program Contract

Program: `count(filter(surface_fixture_elements, present=true, element_type=target_element_type, color_name=target_color_name)); scene=surface_fixture; scope=colored_element_count`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `colored_element_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `surface_fixture_elements`, `present`, `true`, `element_type`, `target_element_type`, `color_name`, `target_color_name`, `surface_fixture`, `colored_element_count`.
Operation: evaluate `count` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Contract
The image shows one projected fixture surface with repeated colored elements.
The prompt asks for the number of elements of the sampled family that have the
requested semantic color.
Generated named-color instances use readout-safe fixture variants and avoid
near-color distractors for the requested semantic color.

The answer is the integer count of finalized present cells whose
`element_type == target_element_type` and `color_name == target_color_name`.

## Annotation Contract
Annotation is a `bbox_set` containing one `[x0, y0, x1, y1]` pixel box around
each counted colored surface element. Non-target colors, empty cells, the
fixture panel, and decorative context are not annotation.

## Prompt And Trace
The prompt bundle is `three_d_surface_fixture_v1` under `src/trace_tasks/resources/prompts/three_d/surface_fixture/`.
The trace records scene variant, target element type, target color, explicit
cell metadata, projected element boxes and centers, and the solver count
predicate.

## Determinism
Generation is deterministic from `instance_seed`, explicit params, config
defaults, prompt bundle, and code versions. Answers and annotation come from the
same finalized fixture trace.
