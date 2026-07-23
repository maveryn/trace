# `task_illustrations__pixel_village__missing_patch_label`

## Summary
- Domain: `illustrations`
- Scene id: `pixel_village`
- Implementation source scene: `pixel_village`
- Implementation source: `src/trace_tasks/tasks/illustrations/pixel_village/missing_patch_label.py`

## Task Contract
Renders a pixel-village source panel with one missing visual region and four or six same-size lettered patch options. The model selects the option letter that exactly restores the missing region.

## Program Contract

Program: `select_option(match_patch(source_image, missing_region, options, transform=none)); scene=pixel_village; scope=missing_patch_label`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `missing_patch_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `match_patch`, `source_image`, `missing_region`, `transform`, `none`, `pixel_village`, `missing_patch_label`.
Operation: evaluate `select_option` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; one of the visible option letters.
Annotation witnesses: `annotation` uses the `bbox_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `matching`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `select_option(match_patch(source_image, missing_region, options, transform=none)); scene=pixel_village; scope=missing_patch_label` |

## Program Metadata
- Program signatures: `selection.option_match`
- Base program contract: `select_option(match_patch(source_image, missing_region, options, transform=none)); scene=pixel_village; scope=missing_patch_label`
- Parameter axes: `option_count`, `source_relative_patch_size`, `source_size`, `canvas_profile`
- Arguments:
  - `source_image`: semantic_role; allowed `pixel_village_source_panel`; source `program_schema_concrete`
  - `missing_region`: semantic_role; allowed `masked_source_region`; source `program_schema_concrete`
  - `options`: semantic_role; allowed `lettered_patch_options`; source `program_schema_concrete`
  - `transform`: operation_parameter; allowed `none`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is one of the visible option letters.

## Annotation Contract
- Annotation schema: `bbox_map`
- Generator `annotation_gt.type`: `bbox_map`
- Annotation keys are `missing_region` and `selected_option`.
- Annotation boxes are final-image pixel boxes around the missing source region and the selected patch option. Do not include all options, option labels, village entities, territories, or context-only regions.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/pixel_village/illustrations_pixel_village_v0.json`.
- Runtime query id is the single-query sentinel `single`; option count, patch size, crop box, style, source render modes, and option-label font are trace parameters.
- Missing-region size is sampled relative to the resolved source image: width `15%-30%`, height `15%-26%`, area at most `6.5%`.
- The composed image uses a frameless functional layout: no source-panel title, decorative outer border, option-card outline, or extra scene background.
- The selected option bbox, answer label, missing-region bbox, and crop boxes must all come from the same `compose_patch_options` execution trace.
