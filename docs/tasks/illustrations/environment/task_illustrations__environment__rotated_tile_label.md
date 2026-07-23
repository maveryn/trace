# `task_illustrations__environment__rotated_tile_label`

## Summary
- Domain: `illustrations`
- Scene id: `environment`
- Implementation source scene: `environment`
- Implementation source: `src/trace_tasks/tasks/illustrations/environment/rotated_tile_label.py`

## Task Contract
Renders an environment source illustration as a profile-aware grid of lettered square tiles, with exactly one tile rotated. The model selects the letter of the rotated tile.

## Program Contract

Program: `select_label(find_rotated_tile(tile_grid, labels=visible_letters)); scene=environment; scope=rotated_tile_label`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `rotated_tile_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `find_rotated_tile`, `tile_grid`, `labels`, `visible_letters`, `environment`, `rotated_tile_label`.
Operation: evaluate `select_label` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; one of the visible tile letters.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `select_label(find_rotated_tile(tile_grid, labels=visible_letters)); scene=environment; scope=rotated_tile_label` |

## Program Metadata
- Program signatures: `selection.visual_anomaly`
- Base program contract: `select_label(find_rotated_tile(tile_grid, labels=visible_letters)); scene=environment; scope=rotated_tile_label`
- Parameter axes: `theme_id`, `source_object_count`, `rotation_degrees`, `canvas_profile`
- Arguments:
  - `tile_grid`: semantic_role; allowed `lettered_environment_tile_grid`; source `program_schema_concrete`
  - `labels`: semantic_role; allowed `A_F_or_A_I`; source `program_schema_concrete`
  - `rotation_degrees`: render_parameter; allowed `90`, `270`; source `trace_metadata`
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
- Annotation contains exactly one final-image pixel box around the rotated tile. Do not include all tile options, tile labels, foreground objects, features, or context-only regions.

## Prompt And Trace Requirements
- Prompt text must come from the `illustrations_environment_v1` prompt bundle, with scene/task/output layers selected deterministically and recorded in metadata.
- Render randomness, sampled environment theme/style, tile-label font, grid style, selected tile, and verifier payloads must be explicit in the instance trace.
- The selected tile is sampled only from tiles with enough visual detail and rotation difference to make the anomaly visible.
- Quarter-turn rotations require square source cells; the source profile chooses a landscape 2x3, square 3x3, or portrait 3x2 grid.
- The composed grid must be full-bleed over the source image with no decorative outer margin, border, or background frame.
- The selected tile bbox, answer label, and rotated tile index must all come from the same `compose_rotated_tile_grid` execution trace.
