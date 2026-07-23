# `task_symbolic__music_staff__key_signature_label`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `music_staff`
3. Task id: `task_symbolic__music_staff__key_signature_label`
4. Objective contract: `key_signature_label`

## Program Contract
Program: `music_staff.key_signature_label(scene=music_staff, scope=key_signature_in_staff_context_plus_visible_text_options, output=option_letter)`

Candidate set: the visible text option cards for key names.
Operands: the visible key-signature accidental pattern and the key name represented by each option.
Operation: identify the key signature from the positioned accidentals and select the unique matching option.
Output binding: `answer` is the selected option letter.
Annotation witnesses: the scalar bbox of the visible key signature.
Query ids: `single`.

## Reasoning Operations

Families: `matching`

## 2) Scene + task contract
1. Entities/relations: a rendered music-staff notation panel with marked notes, chords, measure ranges, option cards, or key signatures depending on the task objective.
2. Supported `query_id` values: `single`
3. `answer_gt.type`: `string`
4. Default `annotation_gt.type`: `bbox`
5. Annotation schema: `bbox`
6. Annotation witness policy: annotation marks the visible key-signature bbox; visible text options, context notes, staff lines, and decorative background are not prompt-facing annotation.
7. Rendering contract: the key signature is rendered as positioned treble-staff accidentals in standard order, not as a plain `# #` or `b b` text string.
8. The correct semantic key name is rendered as one visible option; `answer_gt.value` is the option letter.

| Query id | User-facing operation |
|---|---|
| `single` | The prompt asks the one stable objective contract for this task. |

## 3) Prompt contract
1. `prompt_bundle_id`: `symbolic_music_staff_v1`
2. `scene_key`: `music_staff`
3. `task_key`: `music_key_signature_label`
4. Query keys: internal branch key from the generated trace; public single-operation tasks expose `query_id=single` after registry normalization.
5. Output modes: `answer_only`, `answer_and_annotation`

## 4) Determinism + constraints
1. Deterministic generation and rendering from `instance_seed`.
2. Answer and annotation are bound from the same finalized notation scene and projected bboxes.
3. Count tasks construct the requested answer count before rendering; label tasks construct a unique target label by scene state.
4. No semantic constraints are auto-relaxed after sampling.

## 5) Source files
1. Task source: `src/trace_tasks/tasks/symbolic/music_staff/key_signature_label.py`
2. Scene shared package: `src/trace_tasks/tasks/symbolic/music_staff/shared/`
3. Config: `src/trace_tasks/resources/configs/domains/symbolic/music_staff.yaml`
4. Prompt asset: `src/trace_tasks/resources/prompts/symbolic/music_staff/symbolic_music_staff_v1.json`
5. Focused tests: `tests/test_symbolic_notation_tasks.py`
