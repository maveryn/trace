# `<task_id>` Task Doc Template

## 1) Identity
1. Domain:
2. Scene id:
3. Task id:
4. Objective contract:

## 2) Scene + task contract
1. Entities/relations:
2. Supported `query_id` values:
3. `answer_gt.type`:
4. Answer precision/format, if narrower than the registered answer type:
5. Default `annotation_gt.type`:
6. Alternate annotation forms:
7. Annotation witness policy:
   - annotation contract family (`direct_visible_answer`, `derived_visual_value`, or `diagram_primitives`):
   - minimal visual answer-verification witnesses:
   - derivation/proof details kept in trace metadata:
   - annotation shape choice (`point`, `bbox`, `segment`, `point_set`, `bbox_set`, `segment_set`,
     `point_map`, `bbox_map`, etc.):
     Use `segment` for exactly one visual line/edge/path witness and
     `segment_set` for multiple unordered line/edge/path witnesses.
   - map annotation role names, if used:
   - numeric/readout annotation handling:
   - answer-option annotation policy (only allowed for complete visual
     option-image/panel tasks with a source/reference/original image or
     region; otherwise ground source/candidate objects or primitives):
8. Overlap/touch policy (if applicable):

## 3) Prompt contract
1. `prompt_bundle_id`:
2. `scene_key`:
3. `task_key`:
4. Optional query-id prompt mapping (`query_key` or slot-driven mapping):
5. Required slots:
   - answer output: `json_output_contract_answer_only`, `answer_hint`, `json_example_answer_only`
   - answer+annotation mode: `json_output_contract`, `annotation_hint`, `answer_hint`, `json_example`
6. JSON example validity rule: every documented prompt JSON example must be a valid response for the active task/variant/output mode (keys, value types, and annotation cardinality/semantics).
7. Variant counts (scene/task/query-id/mode):
8. Output modes:
   - `answer_only`
   - `answer_and_annotation`

## 4) Determinism + constraints
1. Seed namespaces used:
2. Unique-answer policy:
3. Reject/resample conditions:
4. No-auto-relaxation guarantee:

## 5) Program and reasoning metadata
1. Concrete `## Program Contract` expression:
2. `## Reasoning Operations` families, in the canonical order from
   `docs/contracts/PROGRAM_SCHEMA_CATALOG.md`:

```markdown
## Reasoning Operations

Families: `ranking`, `aggregation`
```

Reasoning operations are exhaustive analysis metadata, not public taxonomy
nodes. The public task class's literal `reasoning_operations` tuple is the
source of truth; this section is its exact documentation mirror. Record every
meaningful answer-determining operation, but do not count primitive visual
access, output binding, annotation construction, or MCQ letter binding.
`direct_retrieval` is an exclusive fallback and must appear alone.

## 6) Tests
1. Determinism test:
2. Answer/annotation consistency test:
3. Prompt metadata/placeholder test:
4. Constraint-specific tests:

Include only details that are part of the durable public task contract.
