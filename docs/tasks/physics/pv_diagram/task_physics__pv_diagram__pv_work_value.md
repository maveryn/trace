# `task_physics__pv_diagram__pv_work_value`

## Summary
- Domain: `physics`
- Scene id: `pv_diagram`
- Implementation scene: `pv_diagram`
- Implementation source: `src/trace_tasks/tasks/physics/pv_diagram/pv_work_value.py`

## Program Contract

Program: `integer(pressure * (final_volume - initial_volume)); scene=pv_diagram; scope=pv_work_value`

Candidate set: the visible pressure-volume axes, process path, endpoint labels, and shaded/process direction cues inside the `pv_work_value` objective scope.
Operands: `pressure` (semantic_role, allowed `visible_process_pressure`, source `program_schema_concrete`); `initial_volume` (semantic_role, allowed `visible_initial_volume`, source `program_schema_concrete`); `final_volume` (semantic_role, allowed `visible_final_volume`, source `program_schema_concrete`).
Operation: evaluate `integer` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; The answer value is an exact signed integer produced by the symbolic PV construction.
Annotation witnesses: `bbox` witnesses from the finalized render. Annotation is one final-image pixel box around the highlighted PV path and shaded work region. Annotation must mark the visual path/area used for the work calculation, not the answer label, decorative chrome, or unrelated axes text.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `integer(pressure * (final_volume - initial_volume)); scene=pv_diagram; scope=pv_work_value` |

## Program Metadata
- Program signatures: `physics.pv_work_value`
- Base program contract: `integer(pressure * (final_volume - initial_volume)); scene=pv_diagram; scope=pv_work_value`
- Parameter axes: `scene_variant`, `work_mode`, `accent_color_name`, `target_answer`
- Arguments:
  - `pressure`: semantic_role; allowed `visible_process_pressure`; source `program_schema_concrete`
  - `initial_volume`: semantic_role; allowed `visible_initial_volume`; source `program_schema_concrete`
  - `final_volume`: semantic_role; allowed `visible_final_volume`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported `query_id` values: `single`

## Answer Contract
- Answer schema: `integer`
- Generator `answer_gt.type`: `integer`
- The answer value is an exact signed integer produced by the symbolic PV construction.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation is one final-image pixel box around the highlighted PV path and shaded work region.
- Annotation must mark the visual path/area used for the work calculation, not the answer label, decorative chrome, or unrelated axes text.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/pv_diagram/physics_pv_diagram_v1.json`, with scene and task/query layers selected deterministically and recorded in metadata.
- Public `query_id` is `single`; the prompt branch is recorded as `work_value` in trace metadata.
- Render randomness, sampled fonts/styles, query operands, formula quantities, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep all quantities required for the physics computation visible or explicitly stated by the task prompt contract.
