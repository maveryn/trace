# `task_illustrations__pixel_village__swapped_tile_pair_label`

## Summary
- Domain: `illustrations`
- Scene id: `pixel_village`
- Implementation source scene: `pixel_village`
- Implementation source: `src/trace_tasks/tasks/illustrations/pixel_village/swapped_tile_pair_label.py`

## Task Contract
Renders a pixel-village source illustration as a numbered 3x3 tile grid, swaps a pair of tile contents, and shows four lettered pair options. The model selects the option letter naming the two numbered cells that were swapped.

## Program Contract

Program: `select_option(identify_swapped_tile_pair(numbered_tile_grid, pair_options)); scene=pixel_village; scope=swapped_tile_pair_label`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `swapped_tile_pair_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `identify_swapped_tile_pair`, `numbered_tile_grid`, `pair_options`, `pixel_village`, `swapped_tile_pair_label`.
Operation: evaluate `select_option` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; one of the visible option letters `A` through `D`.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `select_option(identify_swapped_tile_pair(numbered_tile_grid, pair_options)); scene=pixel_village; scope=swapped_tile_pair_label` |

## Program Metadata
- Program signatures: `selection.spatial_transform`
- Base program contract: `select_option(identify_swapped_tile_pair(numbered_tile_grid, pair_options)); scene=pixel_village; scope=swapped_tile_pair_label`
- Parameter axes: `source_size`, `canvas_profile`, `swapped_pair`, `correct_index`
- Arguments:
  - `numbered_tile_grid`: semantic_role; allowed `numbered_3x3_pixel_village_tile_grid_with_one_swapped_pair`; source `program_schema_concrete`
  - `pair_options`: semantic_role; allowed `A_D_lettered_tile_pair_options`; source `program_schema_concrete`
  - `swapped_pair`: operation_parameter; allowed `visually_usable_unordered_tile_pair`; source `parameter_axes`
  - `correct_index`: render_parameter; allowed `0_3`; source `trace_metadata`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is one of the visible option letters `A` through `D`.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation has multiple bbox witnesses: the unordered set of two final-image pixel boxes around the two swapped numbered cells.
- Do not include option cards, option labels, all grid cells, source-scene objects, or context-only regions.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/pixel_village/illustrations_pixel_village_v0.json`.
- Runtime query id is the single-query sentinel `single`; answer option letter, option-pair mapping, swapped pair, source render modes, source tile boxes, style, and option-label font are trace parameters.
- The pixel-village source size uses the shared 48px top-down RPG tile profiles and is aligned to both the 3x3 task grid and the pixel-village tile size so tile boundaries do not introduce blank strips or partial pixel tiles.
- The composed grid uses the source image full-bleed with functional tile grid lines, numbered cell badges, and compact pair-option cards only; it must not add a decorative scene background frame.
- The swapped-cell bboxes, answer label, chosen pair option, source tile boxes, and candidate-pair support must all come from the same `compose_swapped_tile_pair_mcq` execution trace.
