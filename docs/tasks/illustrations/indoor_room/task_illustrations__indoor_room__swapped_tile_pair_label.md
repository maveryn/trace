# `task_illustrations__indoor_room__swapped_tile_pair_label`

## Summary
- Domain: `illustrations`
- Scene id: `indoor_room`
- Implementation source scene: `indoor_room`
- Implementation source: `src/trace_tasks/tasks/illustrations/indoor_room/swapped_tile_pair_label.py`

## Task Contract
Renders an indoor-room source illustration as a numbered `3x3` grid, swaps exactly two grid cells, and shows four lettered options naming possible swapped cell pairs. The model selects the option letter for the actual swapped pair.

## Program Contract

Program: `select_option(identify_swapped_tile_pair(numbered_tile_grid, option_pairs)); scene=indoor_room; scope=swapped_tile_pair_label`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `swapped_tile_pair_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `identify_swapped_tile_pair`, `numbered_tile_grid`, `option_pairs`, `indoor_room`, `swapped_tile_pair_label`.
Operation: evaluate `select_option` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; one of the visible option letters `A` through `D`.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `select_option(identify_swapped_tile_pair(numbered_tile_grid, option_pairs)); scene=indoor_room; scope=swapped_tile_pair_label` |

## Program Metadata
- Program signatures: `selection.visual_anomaly`
- Base program contract: `select_option(identify_swapped_tile_pair(numbered_tile_grid, option_pairs)); scene=indoor_room; scope=swapped_tile_pair_label`
- Parameter axes: `theme_id`, `source_object_count`, `swapped_pair`, `canvas_profile`
- Arguments:
  - `numbered_tile_grid`: semantic_role; allowed `indoor_room_3x3_numbered_tile_grid`; source `program_schema_concrete`
  - `option_pairs`: semantic_role; allowed `A_D_lettered_cell_pair_options`; source `program_schema_concrete`
  - `swapped_pair`: sampled_relation; allowed `two_distinct_cells_from_1_to_9`; source `trace_metadata`
  - `canvas_profile`: render_parameter; allowed `landscape`, `square`, `portrait`; source `trace_metadata`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is one of the visible option letters `A` through `D`.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation contains exactly two final-image pixel boxes around the two swapped numbered grid cells. Do not include option boxes, option labels, all grid cells, room objects, or context-only regions.

## Prompt And Trace Requirements
- Prompt text must come from the `illustrations_indoor_room_v0` prompt bundle, with scene/task/output layers selected deterministically and recorded in metadata.
- Render randomness, sampled room theme/style, source scene object count, cell-label font, selected swapped pair, option pairs, and verifier payloads must be explicit in the instance trace.
- The task always uses a fixed `3x3` grid with visible numbered cells `1` through `9`.
- The selected pair is sampled only from cells with enough visual detail and enough pairwise difference to make the swap visible.
- The rendered source grid must use only functional grid lines, cell numbers, and option marks; do not add decorative source-image borders or frames.
- The swapped cell bboxes, answer label, selected pair, and option pairs must all come from the same `compose_swapped_tile_pair_mcq` execution trace.
