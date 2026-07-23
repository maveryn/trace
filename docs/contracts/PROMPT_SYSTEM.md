# Trace Prompt System

Prompt text is externalized, deterministic, and traceable.

## 1) Core Contract
1. Task modules must not hardcode user-facing prompt text.
2. Active source-layout scenes use prompt bundles under
   `src/trace_tasks/resources/prompts/<domain>/<scene_id>/<bundle_id>.json`.
3. Composition layers are:
   - scene,
   - task,
   - optional query,
   - output mode (`answer_only`, `answer_and_annotation`).
4. Selection is deterministic from seed namespaces.
5. Required template lists contain exactly 5 high-quality variants unless the
   schema declares an approved exception.
6. All active tasks must provide task-specific JSON-format guidance in both
   output modes:
   - `answer_only` uses `{"answer": ...}`
   - `answer_and_annotation` uses `{"answer": ..., "annotation": ...}`
7. Output-mode instructions should keep task-specific `answer_hint`,
   `annotation_hint`, and JSON examples. Consumers may add a generic final
   JSON-object instruction without changing task semantics.
8. For named colors, include the canonical hex code in the prompt-facing color
   label using `<color_name> [#RRGGBB]`.
10. If the query layer already contains the full question, the task layer may be
    empty only when the bundle declares `allow_empty_task_templates: true`.

## 2) Bundle Schema
Required fields:

1. `bundle_id`
2. `schema_version`
3. `scene_templates`
4. `task_templates`
5. `answer_or_annotation_templates`
6. `required_slots_by_key`

Optional fields:

1. `query_templates`
2. `allow_empty_task_templates`

Required slots should be declared at the narrowest layer that needs them:

- `scene:<scene_key>` for scene-wide visual framing slots;
- `task:<task_key>` for objective-level slots;
- `query:<query_key>` for branch-specific wording slots;
- output-mode keys for answer/annotation examples and hints.

## 3) Metadata Requirements
Trace `query_spec.prompt_variant` should include:

1. bundle id and layer keys;
2. selected variant indices;
3. query id and query-id count when applicable;
4. slot values for declared required slots;
5. output-mode key/index;
6. prompt asset version when available.

Train records should store the active `prompt` and generated `prompt_variants`
when both output modes are materialized.

## 4) Shared Implementation
1. `src/trace_tasks/core/prompts/assets.py` — bundle loading/cache.
2. `src/trace_tasks/core/prompts/schema.py` — schema validation.
3. `src/trace_tasks/core/prompts/select.py` — deterministic variant selection.
4. `src/trace_tasks/core/prompts/render.py` — strict rendering and metadata.
5. `src/trace_tasks/tasks/shared/prompt_variants.py` — task-level dual-mode orchestration.
6. `src/trace_tasks/tasks/shared/prompt_json_example.py` — deterministic
   JSON-example helpers.

## 5) Prompt Quality Policy
1. Keep stems natural and image-focused.
2. Scene layer describes the visible scaffold only. Do not include taxonomy,
   source, or construction-category labels that are not needed to solve the
   task, such as calling a prompt scaffold "special", "synthetic",
   "procedural", "unlettered", or naming an internal scene family when the
   concrete visible object labels already carry the needed information.
3. Task layer states the operation only when needed.
4. The query layer owns the actual question; user-facing wording should come
   from prompt templates.
5. Output-mode layer owns field hints and JSON examples.
6. Avoid repeating broad nouns such as image, chart, table, diagram, board,
   question, or answer across adjacent layers.
7. Template examples show valid output format for the active answer and
   annotation contract, not the sampled instance's actual answer. The example
   annotation should be an answer-verification witness for the task family, not
   a full derivation/proof trace.
8. Template examples must be internally coherent. If the example answer is a
   count and annotation is the counted witness collection, the example
   annotation cardinality must match the example answer. Scalar examples must
   use scalar shapes, not one-item sets.
9. Annotation examples and hints must use final image pixel coordinates, not
   grid indices, scene coordinates, data values, or normalized coordinates.
10. Use `annotation` terminology in prompts.
11. Prefer direct operand-bound questions over indirect scene-mechanics
    wording. When the task asks for a total, comparison, or selection over
    named operands, the query template should name those operands directly
    rather than making the model infer the requested operation from verbose
    visual descriptions.

## 6) Source Of Truth
Do not maintain exhaustive task-to-bundle maps in this document. They drift.

Use:

1. config references under `src/trace_tasks/resources/configs/domains/`;
2. prompt assets under `src/trace_tasks/resources/prompts/`;
3. runtime prompt metadata in `query_spec.prompt_variant`;
4. generated task inventory in `docs/ACTIVE_TASK_INVENTORY.md`;
5. task-level contracts in `docs/tasks/`.

Validate prompt bundles and rendering with:

```bash
pytest -q tests/test_prompt_system.py
```
