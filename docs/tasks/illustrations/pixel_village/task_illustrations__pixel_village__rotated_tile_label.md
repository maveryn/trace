# `task_illustrations__pixel_village__rotated_tile_label`

## Summary
- Domain: `illustrations`
- Scene id: `pixel_village`
- Implementation source scene: `pixel_village`
- Implementation source: `src/trace_tasks/tasks/illustrations/pixel_village/rotated_tile_label.py`

## Task Contract
Renders a pixel-village source illustration as a profile-aware grid of lettered square tiles with exactly one tile rotated. The model selects the letter of the rotated tile.

## Program Contract

Program: `select_label(find_rotated_tile(tile_grid, labels=visible_letters)); scene=pixel_village; scope=rotated_tile_label`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `rotated_tile_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `find_rotated_tile`, `tile_grid`, `labels`, `visible_letters`, `pixel_village`, `rotated_tile_label`.
Operation: evaluate `select_label` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; one of the visible tile letters.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `select_label(find_rotated_tile(tile_grid, labels=visible_letters)); scene=pixel_village; scope=rotated_tile_label` |

## Program Metadata
- Program signatures: `selection.spatial_transform`
- Base program contract: `select_label(find_rotated_tile(tile_grid, labels=visible_letters)); scene=pixel_village; scope=rotated_tile_label`
- Parameter axes: `rotation_degrees`, `source_size`, `canvas_profile`
- Arguments:
  - `tile_grid`: semantic_role; allowed `profile_aware_pixel_village_tile_grid`; source `program_schema_concrete`
  - `labels`: semantic_role; allowed `A_F_or_A_I_lettered_tiles`; source `program_schema_concrete`
  - `rotation_degrees`: operation_parameter; allowed `90`, `270`; source `parameter_axes`
  - `canvas_profile`: render_parameter; allowed `landscape`, `square`, `portrait`; source `trace_metadata`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is one of the visible tile letters.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation is the final-image pixel box around the rotated tile.
- Do not include all tiles, the tile label badge, source-scene objects, or context-only regions.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/pixel_village/illustrations_pixel_village_v0.json`.
- Runtime query id is the single-query sentinel `single`; rotation angle, usable tile indices, selected tile, source render modes, and label font are trace parameters.
- The composed grid uses the source image full-bleed with functional tile grid lines and option letters only; it must not add decorative outer margins, borders, or background frames.
- Quarter-turn rotations require square source cells; the source profile chooses a landscape 2x3, square 3x3, or portrait 3x2 grid.
- The selected tile bbox, answer label, rotation angle, and usable-tile set must all come from the same `compose_rotated_tile_grid` execution trace.
