# `task_three_d__surface_fixture__color_frequency_option_label`

## Summary
- Domain: `three_d`
- Scene id: `surface_fixture`
- Scene package: `surface_fixture`
- Query ids: `most_frequent_color`, `absent_color`
- Answer type: `option_letter`
- Annotation type: `bbox`
- Annotation schema: `bbox`

## Program Contract

Program: `label(select_text_option(option_cards, color_name = argmax(count(surface_fixture_elements by color)))); scene=surface_fixture; scope=color_frequency_option_label; query=most_frequent_color`

Candidate set: the visible 3D objects, surfaces, room/street/warehouse structures, spatial anchors, markers, and labeled options inside the `color_frequency_option_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `select_text_option`, `option_cards`, `color_name`, `argmax`, `surface_fixture_elements`, `by`, `color`, `surface_fixture`, `color_frequency_option_label`, `query`, `most_frequent_color` plus the active `query_id` branch.
Operation: evaluate `label` over the candidate set using the finalized 3D scene state, camera projection, object identities, spatial relations, counts, distances, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `most_frequent_color`, `absent_color`.

## Reasoning Operations

Families: `counting`, `ranking`

## Contract
The image shows one projected fixture surface containing repeated colored
surface elements, plus six labeled text option cards `A` through `F`. Each
option card names one candidate color using neutral text; the cards are not
filled with that candidate color.
Generated named-color instances use readout-safe fixture variants. Generated
option sets avoid near-color distractors for the answer color.

For `most_frequent_color`, every option color appears on the fixture and
exactly one option color has the highest visible element count.

For `absent_color`, exactly one option color has zero visible elements on the
fixture; the other five option colors appear at least once.

The answer is the capital letter of the matching text option. The answer is not
the color name.

## Annotation Contract
Annotation is the pixel box around the selected text option card. Counted
surface elements, the fixture panel, the option label badge, and the option text
box are trace metadata but are not prompt-facing annotation.

## Prompt Bundle
- Prompt text is loaded from `src/trace_tasks/resources/prompts/three_d/surface_fixture/three_d_surface_fixture_v1.json`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle
version. Answers and annotation come from the same finalized fixture trace.
