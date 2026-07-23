# `task_three_d__surface_fixture__element_count_extremum_label`

## Summary
- Domain: `three_d`
- Scene id: `surface_fixture`
- Scene package: `surface_fixture`
- Query ids: `highest_element_count`, `lowest_element_count`
- Answer type: `option_letter`
- Annotation type: `bbox`
- Annotation schema: `bbox`

## Contract
The image shows four labeled option panels, `A` through `D`, arranged in a 2x2
grid. Each option panel contains the same type of surface fixture and repeated
surface elements, but the visible element count differs across panels. No
numeric counts are printed in the image.

The answer is the capital letter of the option panel whose visible element
count is highest or lowest, depending on the query id.

Visible elements are assigned canonical named colors for visual variety. For
this task, color is recorded as `non_semantic_visual_variation`; it is not part
of the extremum predicate. The selected option is determined only by total
visible element count.

## Program Contract

Program: `label(select_panel(candidate_surface_fixture_panels, extremum(total_visible_element_count, highest|lowest))); scene=surface_fixture; scope=element_count_extremum_label`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `element_count_extremum_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `select_panel`, `candidate_surface_fixture_panels`, `extremum`, `total_visible_element_count`, `highest`, `lowest`, `surface_fixture`, `element_count_extremum_label` plus the active `query_id` branch.
Operation: evaluate `label` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `highest_element_count`, `lowest_element_count`.

## Reasoning Operations

Families: `counting`, `ranking`

## Annotation Contract
Annotation is the pixel box around the selected option panel. Individual
repeated elements and the option label badge are trace metadata but are not
prompt-facing annotation.

## Prompt Bundle
- Prompt text is loaded from `src/trace_tasks/resources/prompts/three_d/surface_fixture/three_d_surface_fixture_v1.json`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle
version. Answers and annotation come from the same finalized option-grid render
trace.
