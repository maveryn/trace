# `task_physics__electromagnetic_induction__induced_current_direction_count`

## Summary
- Domain: `physics`
- Scene id: `electromagnetic_induction`
- Implementation scene: `electromagnetic_induction`
- Implementation source: `src/trace_tasks/tasks/physics/electromagnetic_induction/induced_current_direction_count.py`

## Task Contract
Counts how many of six mini-panels have the queried induced-current direction from visible magnetic-flux-change cues.

## Program Contract

Program: `count(filter(induction_panels, induced_current_direction(panel)=target_current_class)); scene=electromagnetic_induction; scope=induced_current_direction_count`

Candidate set: the visible loop panels, magnetic-field markers, motion/change cues, and panel labels inside the `induced_current_direction_count` objective scope.
Operands: `induction_panels` (visual_candidate_set, allowed `six_mini_panels`, source `program_schema_concrete`); `target_current_class` (semantic_role, allowed `clockwise`, `counterclockwise`, `no_current`, source `query_id`); `field_orientation` (semantic_role, allowed `into_page`, `out_of_page`, source `program_schema_concrete`); `flux_change_mechanism` (semantic_role, allowed `loop_enters_field`, `loop_leaves_field`, `field_strength_increases`, `field_strength_decreases`, `loop_area_expands`, `loop_area_contracts`, `loop_slides_inside_uniform_field`, `stationary_constant_field`, source `program_schema_concrete`); active `query_id` branch when present.
Operation: evaluate `count` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; The answer value is the number of mini-panels whose induced-current class matches the query. Supported answers are `0..6`.
Annotation witnesses: `bbox_set` witnesses from the finalized render. Annotation is the unordered set of bboxes around the full matching mini-panels. If the answer is `0`, annotation is an empty array.
Query ids: `clockwise_induced_current_count`, `counterclockwise_induced_current_count`, `no_induced_current_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `clockwise_induced_current_count` | `count(filter(induction_panels, induced_current_direction(panel)=clockwise)); scene=electromagnetic_induction; scope=induced_current_direction_count; query_branch=clockwise_induced_current_count` |
| `counterclockwise_induced_current_count` | `count(filter(induction_panels, induced_current_direction(panel)=counterclockwise)); scene=electromagnetic_induction; scope=induced_current_direction_count; query_branch=counterclockwise_induced_current_count` |
| `no_induced_current_count` | `count(filter(induction_panels, induced_current_direction(panel)=no_current)); scene=electromagnetic_induction; scope=induced_current_direction_count; query_branch=no_induced_current_count` |

## Program Metadata
- Program signatures: `physics.electromagnetic_induction_direction_count`
- Base program contract: `count(filter(induction_panels, induced_current_direction(panel)=target_current_class)); scene=electromagnetic_induction; scope=induced_current_direction_count`
- Parameter axes: `query_id`, `target_answer`, `field_orientation`, `flux_change_mechanism`
- Arguments:
  - `induction_panels`: visual_candidate_set; allowed `six_mini_panels`; source `program_schema_concrete`
  - `target_current_class`: semantic_role; allowed `clockwise`, `counterclockwise`, `no_current`; source `query_id`
  - `field_orientation`: semantic_role; allowed `into_page`, `out_of_page`; source `program_schema_concrete`
  - `flux_change_mechanism`: semantic_role; allowed `loop_enters_field`, `loop_leaves_field`, `field_strength_increases`, `field_strength_decreases`, `loop_area_expands`, `loop_area_contracts`, `loop_slides_inside_uniform_field`, `stationary_constant_field`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `clockwise_induced_current_count`, `counterclockwise_induced_current_count`, `no_induced_current_count`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer value is the number of mini-panels whose induced-current class matches the query. Supported answers are `0..6`.

## Annotation Contract
- Annotation schema: `bbox_set`
- Generator `annotation_gt.type`: `bbox_set`
- Annotation is the unordered set of bboxes around the full matching mini-panels.
- If the answer is `0`, annotation is an empty array.
- Annotation must not mark individual field symbols, loop arrows, cue text alone, derived current arrows, decorative grid lines, or panel chrome.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/electromagnetic_induction/physics_electromagnetic_induction_v1.json`, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, target answer, panel flux-change mechanisms, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep the field orientation and flux-change cue visible in every mini-panel.
- The renderer must construct exactly six panels and support the full answer range `0..6`.
