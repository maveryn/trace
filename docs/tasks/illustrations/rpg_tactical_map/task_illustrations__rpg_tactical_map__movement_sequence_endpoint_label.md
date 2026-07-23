# `task_illustrations__rpg_tactical_map__movement_sequence_endpoint_label`

## Summary
- Domain: `illustrations`
- Scene id: `rpg_tactical_map`
- Implementation source scene: `rpg_tactical_map`
- Implementation source: `src/trace_tasks/tasks/illustrations/rpg_tactical_map/movement_sequence_endpoint_label.py`

## Task Contract
Selects the single lettered terrain tile where the blue unit ends after following an explicit up/down/left/right move sequence.

## Program Contract

Program: `select(tile, endpoint_after_cardinal_move_sequence(unit_start, move_sequence)); scene=rpg_tactical_map; scope=movement_sequence_endpoint_label`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `movement_sequence_endpoint_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `tile`, `endpoint_after_cardinal_move_sequence`, `unit_start`, `move_sequence`, `rpg_tactical_map`, `movement_sequence_endpoint_label`.
Operation: evaluate `select` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `spatial_relations`, `state_update`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `select(tile, endpoint_after_cardinal_move_sequence(unit_start, move_sequence)); scene=rpg_tactical_map; scope=movement_sequence_endpoint_label` |

## Program Metadata
- Program signatures: `select.endpoint_after_cardinal_move_sequence`
- Base program contract: `select(tile, endpoint_after_cardinal_move_sequence(unit_start, move_sequence)); scene=rpg_tactical_map; scope=movement_sequence_endpoint_label`
- Parameter axes: `move_sequence`, `sequence_length`, `candidate_tile_set`, `terrain_layout`, `water_feature_style`, `canvas_profile`
- Arguments:
  - `unit_start`: blue_unit_start_tile; the visible tile containing the blue unit; source `scene_ir.units`
  - `move_sequence`: ordered list of cardinal moves; allowed directions `up|down|left|right`; source `query_spec.params.move_sequence`
  - `tile`: candidate_tile; visible lettered terrain tile; source `render_map.candidate_tile_ids_by_label`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer is the letter of the only candidate tile equal to the endpoint after applying the visible move sequence.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation contains one pixel bounding box around the selected ending terrain tile.
- Annotation excludes the letter badge, the blue unit, non-selected candidate tiles, and the intermediate path tiles.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/rpg_tactical_map/illustrations_rpg_tactical_map_v0.json`.
- Public prompts must state the ordered move sequence and ask for the ending lettered tile.
- Public prompts must not introduce movement-point costs or terrain-cost rules; this task executes the listed directions one tile per step.
- The task has no semantic query branch beyond `single`; sampled map layout, water feature style, sequence length, tile labels, terrain colors, and canvas profile are trace metadata, not public query ids.
- Candidate tile ids, tile bboxes, label bboxes, move sequence, path tile ids, selected label, selected tile id, and scalar bbox annotation must be recorded in the trace.
