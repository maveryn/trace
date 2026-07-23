# `task_symbolic__music_staff__articulation_symbol_label`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `music_staff`
3. Task id: `task_symbolic__music_staff__articulation_symbol_label`
4. Objective contract: `articulation_symbol_label`

## Program Contract
Program: `music_staff.articulation_symbol_label(scene=music_staff, scope=numbered_articulation_symbol_excerpt_plus_visible_text_options, output=option_letter)`

Candidate set: the visible text option cards for articulation-symbol names.
Operands: the target numbered articulation mark and the semantic label represented by each option.
Operation: classify the target articulation symbol and select the unique option naming that symbol.
Output binding: `answer` is the selected option letter.
Annotation witnesses: the scalar point at the center of the target articulation symbol.
Query ids: `single`.

## Reasoning Operations

Families: `matching`

## 2) Scene + task contract
1. Entities/relations: a rendered music-staff notation panel with marked notes, chords, measure ranges, option cards, or key signatures depending on the task objective.
2. Supported `query_id` values: `single`
3. `answer_gt.type`: `string`
4. Default `annotation_gt.type`: `point`
5. Annotation schema: scalar `point`
6. Annotation witness policy: annotation marks only the center point of the target articulation symbol; visible text options, the target note, non-target notes/symbols, number-marker bboxes, staff lines, and decorative background are not prompt-facing annotation.
7. The correct semantic articulation-symbol name is rendered as one visible option; `answer_gt.value` is the option letter.

| Query id | User-facing operation |
|---|---|
| `single` | The prompt asks the one stable objective contract for this task. |

## 3) Prompt contract
1. `prompt_bundle_id`: `symbolic_music_staff_v1`
2. `scene_key`: `music_staff`
3. `task_key`: `music_articulation_symbol_label`
4. Query keys: internal branch key from the generated trace; public single-operation tasks expose `query_id=single` after registry normalization.
5. Output modes: `answer_only`, `answer_and_annotation`

## 4) Determinism + constraints
1. Deterministic generation and rendering from `instance_seed`.
2. Answer and annotation are bound from the same finalized notation scene and projected bboxes.
3. Count tasks construct the requested answer count before rendering; label tasks construct a unique target label by scene state.
4. No semantic constraints are auto-relaxed after sampling.

## 5) Source files
1. Task source: `src/trace_tasks/tasks/symbolic/music_staff/articulation_symbol_label.py`
2. Scene shared package: `src/trace_tasks/tasks/symbolic/music_staff/shared/`
3. Config: `src/trace_tasks/resources/configs/domains/symbolic/music_staff.yaml`
4. Prompt asset: `src/trace_tasks/resources/prompts/symbolic/music_staff/symbolic_music_staff_v1.json`
5. Focused tests: `tests/test_symbolic_notation_tasks.py`
