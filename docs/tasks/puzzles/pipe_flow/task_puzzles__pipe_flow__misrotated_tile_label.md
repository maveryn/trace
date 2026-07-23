# `task_puzzles__pipe_flow__misrotated_tile_label`

## Public Taxonomy
1. Domain: `puzzles`
2. Scene id: `pipe_flow`
3. Source scene: `pipe_flow`
4. Task id: `task_puzzles__pipe_flow__misrotated_tile_label`

## Query Contract
1. Supported `query_id`: `single`
2. Internal question format: `pipe_flow_misrotated_tile_label`
3. Prompt asks for the labeled pipe tile that should be rotated so flow connects the green start marker to the red triangular finish flag.
4. Internal variation:
   - `grid_size_variant`: `5x5|6x6|7x7`
   - `scene_variant`: `water_pipe|circuit_trace|industrial_conduit`
   - candidate labels: exactly four labeled board cells `A..D`
   - board topology: one visible start-to-finish path with no branch offshoots
   - exactly one candidate tile has current openings rotated away from its required openings
   - rotating exactly that tile by a quarter-turn amount reconnects start to finish.

## Program Contract

Program: `select_label(misrotated_pipe_tile, rule=single_tile_rotation_reconnects_start_to_finish); scene=pipe_flow; scope=pipe_flow_misrotated_tile_label`

Candidate set: the visible pipe tiles, pipe connectors, start/finish markers, misrotated or missing tile cue, and labeled options inside the `pipe_flow_misrotated_tile_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `misrotated_pipe_tile`, `single_tile_rotation_reconnects_start_to_finish`, `pipe_flow`, `pipe_flow_misrotated_tile_label`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; the capital-letter label drawn inside the tile that should be rotated.
Annotation witnesses: `annotation` uses the `bbox` schema; the image-pixel bbox around the labeled tile that should be rotated.
Query ids: `single`.

## Reasoning Operations

Families: `topology`, `transformation`

## Answer And Annotation
1. `answer_gt.type = option_letter`
2. `answer_gt.value` is the capital-letter label drawn inside the tile that should be rotated.
3. `annotation_gt.type = bbox`
4. Annotation schema: `bbox`
5. Annotation is the image-pixel bbox around the labeled tile that should be rotated.
6. `scalar_annotation_checked = true`; scalar bbox annotation is used because the task has exactly one tile witness.

## Trace Contract
1. `scene_ir.entities` includes one `pipe_flow_panel`, one `pipe_flow_tile` per visible pipe/conduit tile, one `pipe_flow_start_marker`, one `pipe_flow_finish_flag`, and four `pipe_flow_candidate_tile` entities.
2. Internal `render_map.item_bboxes_px` contains all candidate tile ids used for verifier projection.
3. Trace sidecars expose the projected scalar pixel box through `annotation_gt` / `projected_annotation`; internal item ids may be sanitized from persisted `execution_trace`.
4. `execution_trace.tiles` records each visible tile's current openings and required path openings.
5. `execution_trace.branch_cells` and `execution_trace.branch_terminal_cells` are empty because branch offshoots are disabled for this task.
6. `execution_trace.candidate_specs` records each candidate label, cell, required openings, current openings, repair rotation turns, and whether rotating that tile reconnects the path.
7. `execution_trace.misrotated_tile_id` identifies the unique tile whose current openings differ from the required path openings and whose rotation solves the puzzle.

## Prompt Contract
1. Bundle: `puzzles_pipe_flow_v1`
2. Scene key: `pipe_flow`
3. Task key: `pipe_flow_misrotated_tile_label_query`
4. Query key: `pipe_flow_misrotated_tile_label`
5. Prompt wording should explain that flow passes through matching tile-edge openings and that exactly one labeled tile should be rotated to connect the green start marker to the red triangular finish flag.
