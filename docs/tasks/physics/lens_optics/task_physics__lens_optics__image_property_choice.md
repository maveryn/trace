# `task_physics__lens_optics__image_property_choice`

## Summary
- Domain: `physics`
- Scene id: `lens_optics`
- Implementation scene: `lens_optics`
- Implementation source: `src/trace_tasks/tasks/physics/lens_optics/image_property_choice.py`

## Program Contract

Program: `option_letter(classify_converging_lens_image_property(object_position_relative_to_focal_marks)); scene=lens_optics; scope=image_property_choice`

Candidate set: the visible lens, object arrow, focal marks, principal rays, and image-property option cues inside the `image_property_choice` objective scope.
Operands: `lens` (semantic_role, allowed `visible_converging_thin_lens`, source `program_schema_concrete`); `object_arrow` (semantic_role, allowed `single_visible_left_side_object_arrow`, source `program_schema_concrete`); `focal_marks` (semantic_role, allowed `F_and_2F_marks_on_both_sides`, source `program_schema_concrete`); `option_map` (semantic_role, allowed `visible_image_property_option_cards`, source `program_schema_concrete`); `object_position_case` (query_operand, allowed `beyond_2f|at_2f|between_f_2f|inside_f`, source `sampled_axis`).
Operation: evaluate `option_letter` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; The answer value is the selected visible option letter. The four supported image properties are `real_inverted_smaller`, `real_inverted_same_size`, `real_inverted_larger`, and `virtual_upright_larger`.
Annotation witnesses: `bbox_map` witnesses from the finalized render. Annotation keys are `lens`, `object_arrow`, and `focal_marks`. Annotation must mark minimal visual witnesses from the final rendered diagram. It must not mark option cards, option letters, title text, decorative grid lines, hidden image-position metadata, or a solved image arrow.
Query ids: `single`.

## Reasoning Operations

Families: `spatial_relations`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `option_letter(classify_converging_lens_image_property(object_position_relative_to_focal_marks)); scene=lens_optics; scope=image_property_choice; query_branch=single` |

## Program Metadata
- Program signatures: `physics.lens_optics_image_property_choice`
- Base program contract: `option_letter(classify_converging_lens_image_property(object_position_relative_to_focal_marks)); scene=lens_optics; scope=image_property_choice`
- Parameter axes: `scene_variant`, `object_position_case`, `correct_option_letter`, `accent_color_name`
- Arguments:
  - `lens`: semantic_role; allowed `visible_converging_thin_lens`; source `program_schema_concrete`
  - `object_arrow`: semantic_role; allowed `single_visible_left_side_object_arrow`; source `program_schema_concrete`
  - `focal_marks`: semantic_role; allowed `F_and_2F_marks_on_both_sides`; source `program_schema_concrete`
  - `option_map`: semantic_role; allowed `visible_image_property_option_cards`; source `program_schema_concrete`
  - `object_position_case`: query_operand; allowed `beyond_2f|at_2f|between_f_2f|inside_f`; source `sampled_axis`
- Argument metadata status: `curated`
- Supported `query_id`s: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is the selected visible option letter. The four supported image properties are `real_inverted_smaller`, `real_inverted_same_size`, `real_inverted_larger`, and `virtual_upright_larger`.

## Annotation Contract
- Annotation schema: `bbox_map`
- Generator `annotation_gt.type`: `bbox_map`
- Annotation keys are `lens`, `object_arrow`, and `focal_marks`.
- Annotation must mark minimal visual witnesses from the final rendered diagram. It must not mark option cards, option letters, title text, decorative grid lines, hidden image-position metadata, or a solved image arrow.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from the physics prompt bundles, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, object-position case, option mapping, option letter, accent color, and verifier payloads must be explicit in the instance trace.
- V0 is converging-lens only and excludes diverging-lens, object-at-F, and no-image cases.
- The rendered prompt image must show the lens, principal axis, `F` and `2F` marks on both sides, one object arrow, and visible image-property option cards. It must not draw the final image arrow or name the sampled object-position case in visible text.
