# `task_puzzles__pipe_flow__pipe_flow_repair_tile_label`

## Public Taxonomy
1. Domain: `puzzles`
2. Scene id: `pipe_flow`
3. Source scene: `pipe_flow`
4. Task id: `task_puzzles__pipe_flow__pipe_flow_repair_tile_label`

## Query Contract
1. Supported `query_id`: `single`
2. Internal question format: `pipe_flow_repair_tile_label`
3. Prompt asks for the labeled 2x2 pipe/conduit repair option that fills the black missing region as drawn, without rotation or flipping, and repairs flow from the green start marker to the red triangular finish flag.
4. Internal variation:
   - `grid_size_variant`: `5x5|6x6|7x7`
   - `gap_size_variant`: fixed `2x2`
   - `scene_variant`: `water_pipe|circuit_trace|industrial_conduit`
   - option labels: exactly four options `A..D`
   - board topology: one visible start-to-finish path with no branch offshoots
   - rotations and flips are not allowed when testing whether an option fits.
   - option occupied cells are complete pipe tiles with two or more openings; one-opening partial pipe cells are rejected.

## Program Contract

Program: `select_label(as_drawn_2x2_pipe_repair_option, rule=connect_start_to_finish_without_rotation); scene=pipe_flow; scope=pipe_flow_repair_tile_label`

Candidate set: the visible pipe tiles, pipe connectors, start/finish markers, misrotated or missing tile cue, and labeled options inside the `pipe_flow_repair_tile_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `as_drawn_2x2_pipe_repair_option`, `connect_start_to_finish_without_rotation`, `pipe_flow`, `pipe_flow_repair_tile_label`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; the capital-letter label on the correct 2x2 repair option panel.
Annotation witnesses: `annotation` uses the `bbox_map` schema; not used because the task has two semantically distinct visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`

## Answer And Annotation
1. `answer_gt.type = option_letter`
2. `answer_gt.value` is the capital-letter label on the correct 2x2 repair option panel.
3. `annotation_gt.type = bbox_map`
4. Annotation schema: `bbox_map`
5. Annotation contains exactly two keys:
   - `selected_option`: bbox around the correct option panel.
   - `missing_gap`: bbox around the black 2x2 missing region.
6. `scalar_annotation_checked = true`; scalar annotation is not used because the task has two semantically distinct visual witnesses.

## Trace Contract
1. `scene_ir.entities` includes one `pipe_flow_panel`, one `pipe_flow_missing_region`, one `pipe_flow_tile` per visible pipe/conduit tile, one `pipe_flow_start_marker`, one `pipe_flow_finish_flag`, and four `pipe_flow_option_panel` entities.
2. Internal `render_map.item_bboxes_px` contains all option panel ids and the missing-region id used for verifier projection.
3. Trace sidecars expose the projected pixel boxes through `annotation_gt` / `projected_annotation`; internal item ids may be sanitized from persisted `execution_trace`.
4. `execution_trace.tiles` records each visible tile's current openings and required path openings. For this task, every visible pipe tile belongs to the single main path.
5. `execution_trace.branch_cells` and `execution_trace.branch_terminal_cells` are empty because branch offshoots are disabled for this repair task.
6. `execution_trace.gap_size_variant` records the fixed `2x2` missing-region size.
7. `execution_trace.option_specs` records the four 2x2 option pieces, records that rotation is not allowed, stores whether each option connects in place, rejects one-opening partial pipe cells, and identifies the unique correct option as drawn.

## Prompt Contract
1. Bundle: `puzzles_pipe_flow_v1`
2. Scene key: `pipe_flow`
3. Task key: `pipe_flow_repair_tile_label_query`
4. Query key: `pipe_flow_repair_tile_label`
5. Prompt wording should explain that flow passes through matching tile-edge openings, that options must be used as drawn without rotation or flipping, and that exactly one 2x2 option fills the black gap to connect the green start marker to the red triangular finish flag.
