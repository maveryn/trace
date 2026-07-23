# `task_illustrations__rpg_tactical_map__water_barrier_unreachable_tile_label`

## Summary
- Domain: `illustrations`
- Scene id: `rpg_tactical_map`
- Implementation source scene: `rpg_tactical_map`
- Implementation source: `src/trace_tasks/tasks/illustrations/rpg_tactical_map/water_barrier_unreachable_tile_label.py`

## Task Contract
Selects the single lettered tile that the blue unit cannot reach because a continuous water barrier crosses the full tactical RPG map.

## Program Contract

Program: `select(tile, unreachable_by_water_barrier_connectivity(unit, tile)); scene=rpg_tactical_map; scope=water_barrier_unreachable_tile_label`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `water_barrier_unreachable_tile_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `tile`, `unreachable_by_water_barrier_connectivity`, `unit`, `rpg_tactical_map`, `water_barrier_unreachable_tile_label`.
Operation: evaluate `select` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `topology`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `select(tile, unreachable_by_water_barrier_connectivity(unit, tile)); scene=rpg_tactical_map; scope=water_barrier_unreachable_tile_label` |

## Program Metadata
- Program signatures: `select.unreachable_tile_across_full_water_barrier`
- Base program contract: `select(tile, unreachable_by_water_barrier_connectivity(unit, tile)); scene=rpg_tactical_map; scope=water_barrier_unreachable_tile_label`
- Parameter axes: `barrier_orientation`, `water_feature_style`, `barrier_position`, `candidate_tile_set`, `canvas_profile`
- Arguments:
  - `unit`: blue_unit; the single reference unit visible in the scene; source `scene_ir.units`
  - `tile`: candidate_tile; visible lettered terrain tile; source `render_map.candidate_tile_ids_by_label`
  - `water_barrier`: full map-crossing water strip or meandering barrier; source `render_map.water_barrier_tile_ids`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer is the letter of the only candidate tile not connected to the blue unit by orthogonal moves through non-water tiles.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation contains one pixel bounding box around the selected unreachable terrain tile.
- Annotation excludes the letter badge, the blue unit, non-selected candidate tiles, and water-barrier tiles.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/rpg_tactical_map/illustrations_rpg_tactical_map_v0.json`.
- Public prompts must state that water tiles cannot be crossed, all non-water tiles can be crossed, and movement is only up, down, left, or right.
- Public prompts must not mention movement budgets or terrain movement costs.
- The task has no semantic query branch beyond `single`; sampled map layout, barrier orientation/style, tile labels, terrain colors, and canvas profile are trace metadata, not public query ids.
- Candidate tile ids, tile bboxes, label bboxes, candidate reachability flags, water-barrier tile ids, barrier orientation/style, selected label, selected tile id, reachable tile ids, and scalar bbox annotation must be recorded in the trace.
