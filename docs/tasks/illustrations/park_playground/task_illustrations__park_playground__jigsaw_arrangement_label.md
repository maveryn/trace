# `task_illustrations__park_playground__jigsaw_arrangement_label`

## Summary
- Domain: `illustrations`
- Scene id: `park_playground`
- Implementation source scene: `park_playground`
- Implementation source: `src/trace_tasks/tasks/illustrations/park_playground/jigsaw_arrangement_label.py`

## Task Contract
Renders a park/playground source illustration, cuts it into a profile-aware tile grid, and shows four lettered complete arrangements of those tiles. Exactly one option preserves the original row-major tile order. The model selects the option letter for the correct arrangement.

## Program Contract

Program: `select_option(match_jigsaw_arrangement(tile_set, arrangement_options, correct_order=row_major)); scene=park_playground; scope=jigsaw_arrangement_label`

Candidate set: the visible illustrated scene objects, people, regions, tiles, patches, labels, and option panels inside the `jigsaw_arrangement_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `match_jigsaw_arrangement`, `tile_set`, `arrangement_options`, `correct_order`, `row_major`, `park_playground`, `jigsaw_arrangement_label`.
Operation: evaluate `select_option` over the candidate set using the visible illustrated objects, regions, layout relationships, counts, patch/tile transforms, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; one of the visible option letters `A` through `D`.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `matching`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `select_option(match_jigsaw_arrangement(tile_set, arrangement_options, correct_order=row_major)); scene=park_playground; scope=jigsaw_arrangement_label` |

## Program Metadata
- Program signatures: `selection.option_match`
- Base program contract: `select_option(match_jigsaw_arrangement(tile_set, arrangement_options, correct_order=row_major)); scene=park_playground; scope=jigsaw_arrangement_label`
- Parameter axes: `person_count`, `equipment_count`, `source_size`, `canvas_profile`
- Arguments:
  - `tile_set`: semantic_role; allowed `profile_aware_park_playground_tiles`; source `program_schema_concrete`
  - `arrangement_options`: semantic_role; allowed `A_D_lettered_tile_arrangements`; source `program_schema_concrete`
  - `correct_order`: operation_parameter; allowed `row_major`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is one of the visible option letters `A` through `D`.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation contains exactly one bbox: the final-image pixel box around the selected jigsaw arrangement option image.
- Do not include the option label badge, all options, source-scene objects, or context-only regions.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/illustrations/park_playground/illustrations_park_playground_v0.json`.
- Render randomness, sampled park setting/style, source scene counts, option-label font, option permutations, selected option, and verifier payloads must be explicit in the instance trace.
- Source panels are accepted only when every source tile has enough visual detail to avoid flat-background jigsaw options.
- The selected option bbox, answer label, option permutations, and source tile boxes must all come from the same `compose_jigsaw_arrangement_options` execution trace.
