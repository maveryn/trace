# `task_illustrations__rpg_dungeon__missing_patch_label`

## Summary
- Domain: `illustrations`
- Scene id: `rpg_dungeon`
- Implementation source scene: `rpg_dungeon`
- Implementation source: `src/trace_tasks/tasks/illustrations/rpg_dungeon/missing_patch_label.py`

## Task Contract
Shows a top-down RPG dungeon source panel with one rectangular region removed and several lettered patch options. The model must select the option letter for the patch that restores the removed region.

## Program Contract

Program: `select(option_letter, option.patch_pixels = source.crop(missing_region) and unique(option)); scene=rpg_dungeon; scope=missing_patch_label`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `missing_patch_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `patch_pixels`, `source`, `crop`, `missing_region`, `unique`, `rpg_dungeon`, `missing_patch_label`.
Operation: evaluate `select` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `matching`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `select(option_letter, option.patch_pixels = source.crop(missing_region) and unique(option)); scene=rpg_dungeon; scope=missing_patch_label` |

## Program Metadata
- Program signatures: `select.visual_missing_patch_option`
- Base program contract: `select(option_letter, option.patch_pixels = source.crop(missing_region) and unique(option)); scene=rpg_dungeon; scope=missing_patch_label`
- Parameter axes: `source_chest_count`, `source_reachable_chest_count`, `source_monster_count`, `option_count`, `correct_index`, `patch_size`, `crop_margin_px`, `canvas_profile`
- Arguments:
  - `missing_region`: rectangular source-image crop removed from the top dungeon panel; source `render_map.missing_region_bbox_px`
  - `option_letter`: visible candidate patch label; allowed generated option letters; source `render_map.option_bboxes_px_by_label`
  - `selected_option`: the visible patch option whose pixels match the removed crop; source `render_map.selected_option_bbox_px`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer is the letter of the patch option that restores the missing region in the dungeon source panel.

## Annotation Contract
- Annotation schema: `bbox_map`
- Generator `annotation_gt.type`: `bbox_map`
- Annotation key `missing_region` contains one bounding box around the blanked source-panel region.
- Annotation key `selected_option` contains one bounding box around the selected patch option.
- Annotation excludes distractor patch options, non-missing source regions, labels, and background.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/rpg_dungeon/illustrations_rpg_dungeon_v0.json`.
- Public prompts must ask for the lettered patch option that restores the missing region.
- Render-only attributes such as palette, chamber placement, source counts, patch size, crop position, correct option index, label font, and canvas profile must not be query ids.
- Source scene trace, missing-region bbox, option bboxes by label, selected-option bbox, crop source boxes, projected keyed bbox-map annotation, and prompt-template metadata must be recorded in the trace.
