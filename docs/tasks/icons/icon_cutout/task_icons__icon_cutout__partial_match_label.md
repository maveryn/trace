# `task_icons__icon_cutout__partial_match_label`

## 1) Identity
1. Domain: `icons`
2. Scene id: `icon_cutout`
3. Scene: `icon_cutout`
4. Task id: `task_icons__icon_cutout__partial_match_label`
5. Objective: select the labeled full-icon option that generated the partial icon fragment.

## Program Contract

Program: `selection.option_match(scene=icon_cutout, scope=partial_fragment_options, rule=source_fragment_shape_match, output=option_letter)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `partial_fragment_options` objective scope.
Operands: visible scene state and prompt-bound operands named by `icon_cutout`, `partial_fragment_options`, `source_fragment_shape_match`.
Operation: evaluate `selection.option_match` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `matching`

## 2) Scene + task contract
1. Entities/relations: one two-panel image with a partial icon fragment on the left and six labeled full-icon options on the right.
2. Supported `query_id` value: `single`
3. Answer type: `answer_gt.type = option_letter`.
4. Annotation type: `annotation_gt.type = bbox_map` with `source_fragment` and `selected_option`.
5. Option policy: the Scene grid uses labels `A..F`; exactly one full-icon option shares the hidden curated `icon_id` that produced the fragment.
6. Fragment policy: the source fragment is a rectangular, rounded, or elliptical window over the correct icon sprite. The window keeps a visible alpha ratio in the configured range and rejects near-blank crops.
7. Distractor policy: distractors are different curated icon ids with distinct full-icon alpha signatures from the correct option.
8. Asset policy: options use the curated icon pool from `src/trace_tasks/resources/assets/icons/all_icons.txt`; names are never shown in the prompt.
9. Styling policy: all options share the same tint and rotation within one instance so the task depends on shape-part matching rather than color or orientation.
10. Noise policy: subtle per-icon edits may be sampled before compositing and are recorded per option. The correct full option uses the same edits as the source sprite.

## 3) Prompt contract
1. `prompt_bundle_id`: `icons_icon_cutout_v1`
2. `scene_key`: `icon_cutout_relation`
3. `task_key`: `relation_query`
4. Answer+annotation JSON shape: `{"annotation":{"source_fragment":[84,168,248,332],"selected_option":[492,112,662,282]},"answer":"C"}`
5. Answer JSON shape: `{"answer":"C"}`
6. Required slots:
   - shared: `object_description`, `question_text_partial_icon_match_label`
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
7. Variant counts (scene/task/mode): exactly 5 templates per required key.
8. Prompt style: the question asks which labeled full icon the partial fragment comes from; it does not expose icon names.

## 4) Determinism + constraints
1. Seed namespaces used: scene-level RNG via `spawn_rng(instance_seed, "scene")`, plus per-icon noise namespaces.
2. Unique-answer policy: the correct option index is sampled first, the correct icon id is inserted exactly once, and distractors are signature-distinct.
3. Reject/resample conditions: unsupported option count, empty icon pool, palette-separation failures, too few distinct distractors, or inability to sample a visible fragment.
4. No-auto-relaxation guarantee: generation fails on unmet fragment/option constraints instead of accepting ambiguous options.
5. Semantic-unit rule: map annotation binds the source fragment frame to the selected full-icon option cell because the task is a visual option-image match.
6. Trace style metadata records the sampled palette, fragment window style, visible alpha ratio, crop coordinates, cell styling, validated text-legibility metadata, and per-icon noise edits.

## 5) Complexity + tests
1. Complexity definition/components: option count, visible-fragment fraction, fragment window style, and option-cell clutter.
2. Behavior/trace/prompt tests: `tests/test_icons_relation_partial_match_label_tasks.py`
3. Implementation: `src/trace_tasks/tasks/icons/icon_cutout/partial_match_label.py`
3. Prompt bundle/config tests: `tests/test_prompt_system.py`, `tests/test_scene_config.py`
