# `task_puzzles__cell_board__symmetry_violation_count`

## Program Contract

Program: `count_mirror_mismatches(cell_board, axis=vertical|horizontal, counted_side=left|top); scene=cell_board; scope=symmetry_violation_count`

Candidate set: the visible grid cells, cell colors/states, labels, walls, start/goal markers, and mirror or connectivity cues inside the `symmetry_violation_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `cell_board`, `axis`, `vertical`, `horizontal`, `counted_side`, `left`, `top`, `symmetry_violation_count`.
Operation: evaluate `count_mirror_mismatches` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `segment_set` schema; one image-pixel segment from each counted-side violating cell center to its mirror-cell center.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `transformation`, `matching`

## 2) Scene + task contract
1. Entities/relations: A rectangular colored cell board with a sampled vertical or horizontal mirror check.
2. Supported `query_id` values: `single`
3. `answer_gt.type`: `integer`
4. Annotation schema: `segment_set`
5. Alternate annotation forms: none
6. Annotation witness policy: one image-pixel segment from each counted-side violating cell center to its mirror-cell center.
7. Overlap/touch policy: only the side named in the prompt determines answer cardinality; mirror cells appear only as segment endpoints.

## 3) Prompt contract
1. `prompt_bundle_id`: `puzzles_cell_board_v1`
2. `scene_key`: `cell_board`
3. `task_key`: `cell_board_symmetry_query`
4. Prompt query key: `symmetry_violation_count`; mirror axis and counted side are trace metadata and prompt slots, not public query ids.
5. Required slots: `mirror_axis`, `counted_side`, plus output-mode slots from the prompt bundle.
6. JSON example validity rule: segment-set cardinality equals the integer answer.
7. Output modes: `answer_only`, `answer_and_annotation`

## 4) Determinism + constraints
1. Seed namespaces used: task-local symmetry namespaces plus shared cell-board layout/style/font/noise namespaces.
2. Unique-answer policy: construction fixes and verifies the counted-side mismatch total on a 3x3 to 5x5 board.
3. Reject/resample conditions: insufficient mirror pairs or mismatch-count drift raises and retries.
4. No-auto-relaxation guarantee: semantic constraints are not relaxed.
