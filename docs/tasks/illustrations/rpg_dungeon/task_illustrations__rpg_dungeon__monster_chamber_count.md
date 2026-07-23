# `task_illustrations__rpg_dungeon__monster_chamber_count`

## Summary
- Domain: `illustrations`
- Scene id: `rpg_dungeon`
- Implementation source scene: `rpg_dungeon`
- Implementation source: `src/trace_tasks/tasks/illustrations/rpg_dungeon/monster_chamber_count.py`

## Task Contract
Counts the treasure chambers that contain a visible monster in a top-down RPG dungeon.

## Program Contract

Program: `count(chamber, exists(monster in chamber) and chamber.layout_role=chest_room); scene=rpg_dungeon; scope=monster_chamber_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `monster_chamber_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `chamber`, `exists`, `monster`, `layout_role`, `chest_room`, `rpg_dungeon`, `monster_chamber_count`.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `count(chamber, exists(monster in chamber) and chamber.layout_role=chest_room); scene=rpg_dungeon; scope=monster_chamber_count` |

## Program Metadata
- Program signatures: `count.containment_rooms`
- Base program contract: `count(chamber, exists(monster in chamber) and chamber.layout_role=chest_room); scene=rpg_dungeon; scope=monster_chamber_count`
- Parameter axes: `total_chest_count`, `monster_chamber_count`, `layout_orientation`, `side_counts`, `monster_object_type`, `canvas_profile`
- Arguments:
  - `chamber`: treasure chamber; allowed visible generated chest chambers; source `scene_ir.chambers`
  - `monster`: visible RPG dungeon monster; allowed generated slime, bat, or spider monsters; source `scene_ir.entities`
  - `total_chest_count`: integer; allowed `4|5|6`; source `parameter_axes`
  - `monster_chamber_count`: integer; allowed `1|2|3|4` and never greater than `total_chest_count`; source `parameter_axes`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer is the number of visible treasure chambers containing at least one monster. Each monster-containing chamber has exactly one visible monster by construction.
- Answer range: `1..4`

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation contains one bounding box around each visible monster in a counted treasure chamber.
- Annotation excludes chests, player, walls, floors, corridors, and empty chambers.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/rpg_dungeon/illustrations_rpg_dungeon_v0.json`.
- Public prompts use `chambers`, not `rooms`, for this scene.
- Render-only attributes such as palette, monster type, chamber positions, layout orientation, side counts, and canvas profile must not be query ids.
- Chamber ids, monster entity ids, monster chamber ids, monster bboxes, layout orientation, side counts, total chest count, projected bbox-set annotation, and diagnostic entity bboxes must be recorded in the trace.
