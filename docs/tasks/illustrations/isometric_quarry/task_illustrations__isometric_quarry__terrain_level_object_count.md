# `task_illustrations__isometric_quarry__terrain_level_object_count`

## Summary
- Domain: `illustrations`
- Scene id: `isometric_quarry`
- Implementation source scene: `isometric_quarry`
- Implementation source: `src/trace_tasks/tasks/illustrations/isometric_quarry/terrain_level_object_count.py`

## Task Contract
Counts quarry objects of a sampled subtype on either the highest or lowest terrain level in an isometric quarry scene.

## Program Contract

Program: `count(object where subtype in {ore_vein,mine_cart} and level(object_base)=extremum(levels), mode=highest|lowest); scene=isometric_quarry; scope=terrain_level_object_count`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `terrain_level_object_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `object`, `where`, `subtype`, `ore_vein`, `mine_cart`, `level`, `object_base`, `extremum`, `levels`, `mode`, `highest`, `lowest` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `highest_terrain_object_count`, `lowest_terrain_object_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`, `logical_composition`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `highest_terrain_object_count` | `count(object where subtype=target_object_type and level(object_base)=max(levels)); scene=isometric_quarry; scope=terrain_level_object_count` |
| `lowest_terrain_object_count` | `count(object where subtype=target_object_type and level(object_base)=min(levels)); scene=isometric_quarry; scope=terrain_level_object_count` |

## Program Metadata
- Program signatures: `count.spatial_elevation_object_filter`
- Base program contract: `count(object where subtype=target_object_type and level(object_base)=extremum(levels)); scene=isometric_quarry; scope=terrain_level_object_count`
- Parameter axes: `canvas_profile`, `target_object_type`, `target_count`, `active_level_range`, `layout_family`, `context_object_placement`
- Arguments:
  - `target_object_type`: sampled quarry object subtype; allowed `ore_vein|mine_cart`; source `query_spec.params`
  - `level`: terrain elevation of the object's base tile; source `scene_ir.entities[].level`
  - `mode`: extremum operator; allowed `highest|lowest`; source `query_id`
- Argument metadata status: `curated`
- Supported query ids: `highest_terrain_object_count`, `lowest_terrain_object_count`
- `scalar_annotation_checked`: not applicable; annotation schema is `bbox_set`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- Answer range: `0..5`

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation contains one bounding box for each counted quarry object; use `[]` when the answer is `0`.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/isometric_quarry/illustrations_isometric_quarry_v0.json`.
- Prompts must name the sampled object label (`ore veins` or `mine carts`) and the elevation extremum.
- The object-count renderer uses a clean quarry object pool: countable `ore_vein|mine_cart` objects plus visually distinct `barrel|crate|sign` distractors.
- The object-count renderer disables quarry patch terrain so rocks, ore-dust patches, rails, and support clutter cannot be mistaken for count targets.
- Render-only attributes such as palette, canvas profile, active max level, layout family, and distractor context object placement must not be query ids.
