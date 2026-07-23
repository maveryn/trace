# `task_physics__refraction_layers__medium_speed_order_label`

## Summary
- Domain: `physics`
- Scene id: `refraction_layers`
- Implementation scene: `refraction_layers`
- Implementation source: `src/trace_tasks/tasks/physics/refraction_layers/medium_speed_order_label.py`

## Program Contract

Program: `option_letter(order_by_speed(media_m1_m2_m3, inferred_from=ray_bending_at_interfaces)); scene=refraction_layers; scope=medium_speed_order_label`

Candidate set: the visible media layers, interface normals, ray segments, and medium labels inside the `medium_speed_order_label` objective scope.
Operands: `media_m1_m2_m3` (semantic_role, allowed `visible_labeled_media_regions`, source `program_schema_concrete`); `ray_bending_at_interfaces` (semantic_role, allowed `visible_ray_path_and_normals`, source `program_schema_concrete`); `option_map` (semantic_role, allowed `visible_speed_order_options`, source `program_schema_concrete`).
Operation: evaluate `option_letter` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; The answer value is the selected visible option letter.
Annotation witnesses: `bbox_set` witnesses from the finalized render. Annotation contains two pixel boxes around the two visible ray-bend regions where the ray crosses the media interfaces. Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, full media regions, or derived hidden speed ranks.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `option_letter(order_by_speed(media_m1_m2_m3, inferred_from=ray_bending_at_interfaces)); scene=refraction_layers; scope=medium_speed_order_label; query_branch=three_medium_speed_order` |

## Program Metadata
- Program signatures: `physics.refraction_medium_speed_order_label`
- Base program contract: `option_letter(order_by_speed(media_m1_m2_m3, inferred_from=ray_bending_at_interfaces)); scene=refraction_layers; scope=medium_speed_order_label`
- Parameter axes: `layer_orientation`, `entry_side`, `speed_order`, `option_map`
- Arguments:
  - `media_m1_m2_m3`: semantic_role; allowed `visible_labeled_media_regions`; source `program_schema_concrete`
  - `ray_bending_at_interfaces`: semantic_role; allowed `visible_ray_path_and_normals`; source `program_schema_concrete`
  - `option_map`: semantic_role; allowed `visible_speed_order_options`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported `query_id`s: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is the selected visible option letter.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation contains two pixel boxes around the two visible ray-bend regions where the ray crosses the media interfaces.
- Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, full media regions, or derived hidden speed ranks.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the physics prompt bundles, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, hidden medium speed ranks, option placement, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep the media labels, ray path, interface normals, and answer options visible; numeric refractive-index or speed labels are intentionally omitted.
