# `task_icons__mirror_grid__mirror_symmetry_match_label`

## 1) Identity
1. Domain: `icons`
2. Scene id: `mirror_grid`
3. Scene: `mirror_grid`
4. Task id: `task_icons__mirror_grid__mirror_symmetry_match_label`
5. Objective: select the labeled option cell with the same mirror symmetry as the Reference cell.

## Program Contract

Program: `selection.option_match(scene=mirror_grid, scope=reference_and_option_cells, rule=mirror_symmetry_signature_match, output=option_letter)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `reference_and_option_cells` objective scope.
Operands: visible scene state and prompt-bound operands named by `mirror_grid`, `reference_and_option_cells`, `mirror_symmetry_signature_match`.
Operation: evaluate `selection.option_match` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `matching`

## 2) Scene + task contract
1. Entities/relations: one two-panel image with a `Reference` cell on the left and labeled option cells on the right.
2. Query ids: `single`.
3. Internal `mirror_signature` values: `mirror_vertical`, `mirror_horizontal`, `mirror_diagonal_main`, `mirror_diagonal_anti`, `mirror_both_axes`.
4. Answer type: `answer_gt.type = option_letter`.
5. Annotation type: `annotation_gt.type = bbox_map` with keys `reference_cell` and `matching_option_cell`.
6. Annotation schema: `bbox_map`.
7. Option policy: option count is sampled from `4` or `6`; labels are fixed row-major as `A..D` or `A..F`.
8. Unique-answer policy: exactly one option cell has the same exact mirror-symmetry signature as the Reference cell.
9. Asset policy: cells use the curated asymmetric icon subset from `src/trace_tasks/resources/assets/icons/non_symmetry.txt`; icon identity is not part of the query.
10. Symmetry policy: matching requires the same exact symmetry signature as the Reference cell; distractors are exact other-symmetry cells or cells with no supported mirror symmetry.
11. Noise policy: mirrored counterparts are derived from the edited seed sprite, so per-icon visual noise does not break the exact symmetry contract.

## 3) Prompt contract
1. `prompt_bundle_id`: `icons_mirror_grid_v1`
2. `scene_key`: `reference_mirror_grid`
3. `task_key`: `mirror_symmetry_match_label`
4. Answer+annotation JSON shape: `{"annotation":{"reference_cell":[44,190,268,414],"matching_option_cell":[590,120,790,320]},"answer":"C"}`
5. Answer JSON shape: `{"answer":"C"}`
6. Prompt wording asks which labeled option cell has the same mirror symmetry as the Reference cell.

## 4) Determinism + constraints
1. Seed namespaces used: scene-level RNG via `spawn_rng(instance_seed, "scene")`.
2. Unique-answer policy: the answer label is selected first, then the rendered option layout is constrained so only that label matches.
3. Reject/resample conditions: unsupported option config, empty icon pool, palette-separation failures, or inability to render exact symmetric / exact non-symmetric cell patches under configured gap and margin constraints.
4. Semantic-unit rule: annotation boxes surround the whole Reference cell and whole matching option cell because the task asks about cell-level mirror symmetry.

## 5) Complexity + tests
1. Complexity definition/components: option count + internal mirror signature.
2. Determinism/build tests: `tests/test_icons_relation_mirror_symmetry_contracts.py`
3. Behavior/trace/prompt tests: `tests/test_icons_relation_mirror_symmetry_tasks.py`
4. Prompt bundle/config tests: `tests/test_icons_prompt_wording.py`, `tests/test_icons_scene_config.py`
