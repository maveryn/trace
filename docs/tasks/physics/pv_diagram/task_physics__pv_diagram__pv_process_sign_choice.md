# `task_physics__pv_diagram__pv_process_sign_choice`

## Summary
- Domain: `physics`
- Scene id: `pv_diagram`
- Implementation scene: `pv_diagram`
- Implementation source: `src/trace_tasks/tasks/physics/pv_diagram/pv_process_sign_choice.py`

## Program Contract

Program: `option_letter(select(candidate_processes, sign(volume_change(process))=target_sign)); scene=pv_diagram; scope=pv_process_sign_choice`

Candidate set: the visible pressure-volume axes, process path, endpoint labels, and shaded/process direction cues inside the `pv_process_sign_choice` objective scope.
Operands: `candidate_processes` (semantic_role, allowed `visible_mini_pv_processes`, source `program_schema_concrete`); `process` (semantic_role, allowed `candidate_pv_process`, source `program_schema_concrete`); `target_sign` (semantic_role, allowed `negative`, `positive`, `zero`, source `program_schema_concrete`).
Operation: evaluate `option_letter` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; The answer value is the selected visible option letter.
Annotation witnesses: `bbox` witnesses from the finalized render. Annotation is one final-image pixel box around the process arrow in the correct mini diagram. Annotation must mark the selected visual process arrow, not the option label alone, decorative chrome, or unrelated candidates.
Query ids: `single`.

## Reasoning Operations

Families: `formula_evaluation`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `option_letter(select(candidate_processes, sign(volume_change(process))=target_sign)); scene=pv_diagram; scope=pv_process_sign_choice` |

## Program Metadata
- Program signatures: `physics.pv_process_sign_choice`
- Base program contract: `option_letter(select(candidate_processes, sign(volume_change(process))=target_sign)); scene=pv_diagram; scope=pv_process_sign_choice`
- Parameter axes: `scene_variant`, `target_sign`, `correct_option_letter`, `accent_color_name`
- Arguments:
  - `candidate_processes`: semantic_role; allowed `visible_mini_pv_processes`; source `program_schema_concrete`
  - `process`: semantic_role; allowed `candidate_pv_process`; source `program_schema_concrete`
  - `target_sign`: semantic_role; allowed `negative`, `positive`, `zero`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported `query_id` values: `single`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is the selected visible option letter.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation is one final-image pixel box around the process arrow in the correct mini diagram.
- Annotation must mark the selected visual process arrow, not the option label alone, decorative chrome, or unrelated candidates.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/pv_diagram/physics_pv_diagram_v1.json`, with scene and task/query layers selected deterministically and recorded in metadata.
- Public `query_id` is `single`; the prompt branch is recorded as `process_sign_choice` in trace metadata.
- Render randomness, sampled fonts/styles, query operands, formula quantities, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep all quantities required for the physics computation visible or explicitly stated by the task prompt contract.
