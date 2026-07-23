# `task_illustrations__rpg_house__missing_patch_label`

## Summary
- Domain: `illustrations`
- Scene id: `rpg_house`
- Implementation source scene: `rpg_house`
- Implementation source: `src/trace_tasks/tasks/illustrations/rpg_house/missing_patch_label.py`

## Task Contract
Render a top-down RPG house source panel with one missing rectangular region and four or six lettered patch options. The model selects the option letter that restores the missing region.

## Program Contract

Program: `select_option(match_patch(source_image, missing_region, options, transform=none)); scene=rpg_house; scope=missing_patch_label`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `missing_patch_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `match_patch`, `source_image`, `missing_region`, `transform`, `none`, `rpg_house`, `missing_patch_label`.
Operation: evaluate `select_option` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; one of the visible patch option letters.
Annotation witnesses: `annotation` uses the `bbox_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `matching`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `select_option(match_patch(source_image, missing_region, options, transform=none)); scene=rpg_house; scope=missing_patch_label` |

## Program Metadata
- Program signatures: `selection.option_match`
- Base program contract: `select_option(match_patch(source_image, missing_region, options, transform=none)); scene=rpg_house; scope=missing_patch_label`
- Parameter axes: `source_room_count`, `option_count`, `source_relative_patch_size`, `crop_box`, `canvas_profile`
- Arguments:
  - `source_image`: semantic role; allowed `rpg_house_source_panel`; source `program_schema_concrete`
  - `missing_region`: semantic role; allowed `masked_source_region`; source `program_schema_concrete`
  - `options`: semantic role; allowed `lettered_patch_options`; source `program_schema_concrete`
  - `canvas_profile`: render parameter; allowed `landscape`, `square`, `portrait`; source `trace_metadata`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is one of the visible patch option letters.

## Annotation Contract
- Annotation schema: `bbox_map`
- Generator `annotation_gt.type`: `bbox_map`
- Annotation keys are `missing_region` and `selected_option`.
- Annotation boxes are final-image pixel boxes around the missing source region and selected patch option. Do not include all options, option labels, room fixtures, or context-only source objects.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/rpg_house/illustrations_rpg_house_v0.json`.
- Render randomness, source room count, sampled label font, option order, crop box, and verifier payloads must be explicit in the instance trace.
- Missing-region size is sampled relative to the resolved source image: width `15%-30%`, height `15%-26%`, area at most `6.5%`.
- Patch crops are sampled from visible room, door, or fixture regions to avoid blank floor-only ambiguity.
- The selected option bbox, answer label, and missing-region bbox must all come from the same `compose_patch_options` execution trace.
