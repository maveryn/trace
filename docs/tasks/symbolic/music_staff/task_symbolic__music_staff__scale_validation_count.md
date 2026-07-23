# `task_symbolic__music_staff__scale_validation_count`

## 1) Identity
1. Domain: `symbolic`
2. Scene id: `music_staff`
3. Task id: `task_symbolic__music_staff__scale_validation_count`
4. Objective contract: `scale_validation_count`

## Program Contract
Program: `music_staff.scale_validation_count(scene=music_staff, scope=four_numbered_scale_fragments, predicate=fits_target_key, output=integer)`

Candidate set: the four visible numbered scale-fragment ranges.
Operands: the target key named in the prompt and the notes in each scale-fragment range.
Operation: test whether each fragment fits the target key and count matching fragments.
Output binding: `answer` is the matching fragment count as an integer.
Annotation witnesses: a homogeneous `bbox_set` of scale-fragment range boxes that fit the target key.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `matching`

## 2) Scene + task contract
1. Entities/relations: a rendered music-staff notation panel with marked notes, chords, measure ranges, option cards, or key signatures depending on the task objective.
2. Supported `query_id` values: `single`
3. `answer_gt.type`: `integer`
4. Default `annotation_gt.type`: `bbox_set`
5. Annotation schema: `bbox_set`
6. Annotation witness policy: annotation marks scale-fragment range boxes that correctly fit the target key; non-target notes, nonmatching ranges, staff lines, and decorative background are not prompt-facing annotation.

| Query id | User-facing operation |
|---|---|
| `single` | The prompt asks the one stable objective contract for this task. |

## 3) Prompt contract
1. `prompt_bundle_id`: `symbolic_music_staff_v1`
2. `scene_key`: `music_staff`
3. `task_key`: `music_scale_validation_count`
4. Query keys: internal branch key from the generated trace; public single-operation tasks expose `query_id=single` after registry normalization.
5. Output modes: `answer_only`, `answer_and_annotation`

## 4) Determinism + constraints
1. Deterministic generation and rendering from `instance_seed`.
2. Answer and annotation are bound from the same finalized notation scene and projected bboxes.
3. Count tasks construct the requested answer count before rendering; label tasks construct a unique target label by scene state.
4. No semantic constraints are auto-relaxed after sampling.

## 5) Source files
1. Task source: `src/trace_tasks/tasks/symbolic/music_staff/scale_validation_count.py`
2. Scene shared package: `src/trace_tasks/tasks/symbolic/music_staff/shared/`
3. Config: `src/trace_tasks/resources/configs/domains/symbolic/music_staff.yaml`
4. Prompt asset: `src/trace_tasks/resources/prompts/symbolic/music_staff/symbolic_music_staff_v1.json`
5. Focused tests: `tests/test_symbolic_notation_tasks.py`
