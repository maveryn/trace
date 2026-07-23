# `task_illustrations__construction_site__missing_patch_label`

## Summary
- Domain: `illustrations`
- Scene id: `construction_site`
- Implementation scene: `construction_site`
- Implementation source: `src/trace_tasks/tasks/illustrations/construction_site/missing_patch_label.py`

## Contract
1. Domain: `illustrations`
2. Scene id: `construction_site`
3. Public task id: `task_illustrations__construction_site__missing_patch_label`
4. Supported `query_id` values: `single`
5. Query ids: `single`
6. Answer schema: `option_letter`
7. Annotation schema: `bbox_map`
8. Program schema: `select_option(match_patch(source_image, missing_region, options)); scene=construction_site; scope=missing_patch_label`

## Program Contract

Program: `select_option(match_patch(source_image, missing_region, options)); scene=construction_site; scope=missing_patch_label`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `missing_patch_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `match_patch`, `source_image`, `missing_region`, `construction_site`, `missing_patch_label`.
Operation: evaluate `select_option` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; one of the visible option letters.
Annotation witnesses: `annotation` uses the `bbox_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `matching`

## Task Contract
Renders a construction-site source panel with one missing visual region and four or six same-size lettered patch options. The model selects the option letter that restores the missing region.

This public task id is a stable scene-owned visual-option contract. The single public query uses exact patch matching only.

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `select_option(match_patch(source_image, missing_region, options, transform=none)); scene=construction_site; scope=missing_patch_label` |

## Program Metadata
- Program signatures: `selection.option_match`
- Base program contract: `select_option(match_patch(source_image, missing_region, options)); scene=construction_site; scope=missing_patch_label`
- Parameter axes: `none`
- Arguments:
  - `source_image`: semantic_role; allowed `construction_site_source_panel`; source `program_schema_concrete`
  - `missing_region`: semantic_role; allowed `masked_source_region`; source `program_schema_concrete`
  - `options`: semantic_role; allowed `lettered_patch_options`; source `program_schema_concrete`
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
- Annotation boxes are final-image pixel boxes around the missing source region and the selected patch option. Do not include all options, labels, or context-only source objects.

## Prompt And Trace Requirements
- Prompt text must come from the `illustrations_construction_site_v1` scene prompt bundle, with scene/task/output layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, option order, crop box, and verifier payloads must be explicit in the instance trace.
- Option count is sampled from `4` or `6`; all option bboxes use the same pixel width and height as the missing region.
- Missing-region size is sampled relative to the resolved source image: width `15%-30%`, height `15%-26%`, area at most `6.5%`.
- Source-zone text labels are suppressed in this reconstruction view so the task is patch matching rather than text matching.
- The selected option bbox, answer label, and missing-region bbox must all come from the same `compose_patch_options` execution trace.
