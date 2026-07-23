# Trace Annotation And Reward Contracts

Normative contract for Trace annotation types and answer/annotation reward
dispatch.

Trace v0 uses **annotation** as the public grounding term. Active prompts,
outputs, trace artifacts, and reward contracts must use `annotation` /
`annotation_gt`; do not add alternate public grounding keys, prompt wording, or
compatibility aliases.

Conceptually, Trace annotations are **answer-verification witnesses**: the
minimal final-image visual primitives needed to verify the task answer contract
for that task family. For direct selection and counting tasks, this is usually
the selected/countable answer object or region. For chart, graph, measurement,
or arithmetic tasks, it may be the visible operands used to compute the answer.
For geometry or physics diagrams, it may be canonical diagram primitives such
as points, sides, rays, intervals, axes, or marked regions. Annotation is not a
full reasoning proof, derivation trace, or list of every visual cue a solver
could use. Store derivation/proof/debug details in `trace_payload`,
`witness_symbolic`, `scene_ir`, `render_map`, or task-specific metadata, not in
`annotation_gt`.

## 1) Purpose
Trace has two public output contracts:

- `answer_gt`: task answer value. In v0, answer reward uses exact-match
  normalization for every registered answer type.
- `annotation_gt`: minimal visual answer-verification witness. The annotation
  type determines the generic annotation scorer.

Every built Trace instance carries a compact `reward_contract` payload so RLVR
code can select the generic answer and annotation scorers without task-specific
branching. The payload is instance-local, versioned, and derived from the public
`answer_gt.type` and `annotation_gt.type`.

Task code must not hand-author `reward_contract`. The builder resolves it through
`src/trace_tasks/core/reward_contracts.py`; scoring lives in
`src/trace_tasks/core/reward_scoring.py`.

## 2) ABI Shape
`TrainInstance.reward_contract` and `TraceInstance.reward_contract` use:

```json
{
  "reward_contract_version": "v0",
  "answer": {
    "id": "answer_exact_match_v0",
    "type": "integer"
  },
  "annotation": {
    "id": "bbox_set_soft_iou_v0",
    "type": "bbox_set"
  }
}
```

Rules:

1. `reward_contract_version` is required and currently fixed to `v0`.
2. `answer.id` is required and currently fixed to `answer_exact_match_v0`.
3. `answer.type` must exactly match `answer_gt.type`.
4. `annotation.id` must be the resolver output for `annotation_gt.type`.
5. `annotation.type` must exactly match `annotation_gt.type`.
6. Non-current public reward ids are rejected. Do not add compatibility aliases
   for retired public output contracts.

## 3) Current Resolver Table

| Public type | Reward id | Scoring rule |
| --- | --- | --- |
| any registered answer type | `answer_exact_match_v0` | Exact match after answer-type normalization. Ordered answer types remain sequence-sensitive. |
| `bbox` | `bbox_soft_iou_v0` | One scalar bbox scored by raw IoU. |
| `bbox_set` | `bbox_set_soft_iou_v0` | Unordered Hungarian matching over raw IoU. |
| `bbox_sequence` | `bbox_sequence_soft_iou_v0` | Index-aligned IoU aggregation. |
| `bbox_map` | `bbox_map_soft_iou_v0` | Exact key matching, then one bbox IoU per key. |
| `bbox_set_map` | `bbox_set_map_soft_iou_v0` | Exact key matching, then unordered bbox-set IoU matching inside each key. |
| `point` | `point_soft_distance_v0` | One scalar point scored by soft pixel distance. |
| `point_set` | `point_set_soft_distance_v0` | Unordered Hungarian matching over soft pixel distance. |
| `point_sequence` | `point_sequence_soft_distance_v0` | Index-aligned soft pixel distance. |
| `segment` | `segment_soft_distance_v0` | One undirected segment scored by endpoint distance. |
| `segment_set` | `segment_set_soft_distance_v0` | Unordered matching of undirected segment witnesses. |
| `point_map` | `point_map_soft_distance_v0` | Exact key matching, then one point-distance score per key. |
| `point_set_map` | `point_set_map_soft_distance_v0` | Exact key matching, then unordered point-set distance matching inside each key. |

## 4) Annotation Contract Rules

Public annotation contracts are image-level only. Annotation coordinates are
always pixel coordinates in the provided/exported image, after layout jitter,
scaling, cropping, and any coordinate-preserving post-processing. Do not use
scene-local logical coordinates, grid indices, data values, normalized
coordinates, or model-internal resized tensor coordinates in `annotation_gt`.

The annotation target depends on the task family. The core contract question is:
"Which final-image primitives are minimally sufficient to verify that this
answer was bound to the right visible content?" Use the following hierarchy:

- **Direct visible answer tasks** annotate the selected or counted answer
  witness. Selection and MCQ tasks annotate the selected visual object, option
  panel, tile, card, region, or control. Counting tasks annotate exactly the
  counted visible objects, with an empty set for a valid zero count. Object
  identity tasks annotate the answer object, not distractors or reference
  objects unless the reference itself is the answer.
- **Derived visual-value tasks** annotate the minimal visible operands needed
  to verify the computed answer. Examples include chart bars used in a total,
  cells/amounts used in an arithmetic result, graph intervals/segments/axis
  scale used for an area or rate value, and measurement markings used for a
  readout. Do not annotate extra context, unrelated legends, distractors, or
  intermediate arithmetic steps.
- **Diagram and formula-grounded tasks** annotate the canonical visual
  primitives that bind the problem, such as triangle sides, circle centers,
  rays, vectors, graph paths, force arrows, marked intervals, baseline
  segments, axes, or labeled measurement operands. Formula choices,
  intermediate values, and proof steps belong in trace metadata.
- **Scoped tasks** annotate the selected/countable/operand witnesses inside the
  scope. The scope region itself is included only when it is part of the
  answer-verification witness, such as a section total, a graph interval, or a
  selected region; otherwise keep the scope name/bbox in trace metadata.
- **Option-image tasks** annotate the selected option image when the option
  image itself is the answer-verification witness.

If consumers need the full visual proof path, keep it in trace metadata. Do not
inflate `annotation_gt` to include non-witness context simply because it helped
derive the answer.

Use these global homogeneous annotation type names directly:

- `bbox`
- `bbox_set`
- `bbox_sequence`
- `bbox_map`
- `bbox_set_map`
- `point`
- `point_set`
- `point_sequence`
- `segment`
- `segment_set`
- `point_map`
- `point_set_map`

Do not create domain-specific map annotation names such as
`physics_point_map` or `geometry_bbox_map`. Prefer map annotation when the
reward must verify key binding, for example source versus target, outer shape
versus shaded region, or input versus output measurement. Unordered sets are
appropriate for counting tasks and homogeneous witness collections where key
identity and order do not matter.

Annotation cardinality must match the task contract:

- use scalar `bbox`, `point`, or `segment` when the task guarantees exactly one
  visual witness;
- use `bbox_set`, `point_set`, or `segment_set` only for variable-size or
  multiple unordered homogeneous witnesses;
- use `bbox_sequence` or `point_sequence` only when witness order is part of the
  task contract;
- use map annotation only when key binding matters.

Set annotations are not a fallback for "several things were relevant." The
witnesses in one set must have the same semantic role. If the annotation needs
both an original/source panel and a selected option panel, or both an input
object and an output/candidate object, use a map contract with explicit role
keys instead of a plain `bbox_set` or `point_set`.

Do not use a one-item set for a guaranteed-single witness task. Do not use a
one-key `point_map` or `bbox_map` only to avoid scalar annotation.
Map annotations are for binding multiple semantic roles, not cardinality
workarounds. If there is exactly one witness, use the scalar type even when the
role name is obvious; for example, use `bbox` for one selected endpoint box, not
`bbox_map` with an `answer_endpoint` key.

Choose the annotation geometry from the visual primitive the answer is grounded
in, not from the domain alone:

- use `bbox` / `bbox_set` when the witness is an area-like object or selectable
  region, such as a grid cell, tile, board square, card, GUI control, text box,
  table cell, chart bar, image patch, or page region;
- use `point` / `point_set` when the witness is a localized feature or compact
  object center, such as a chart mark, graph node, dot, vertex, intersection,
  ball, marble, token center, or pointer tip;
- use `segment` / `segment_set` when the witness is a line-like relation, such
  as an edge, route, row/column span, shot path, trend interval, geometry side,
  or vector.

These are defaults, not a substitute for task-specific contract analysis. A task may use a different
annotation type when the program contract makes that geometry more faithful, but
the task doc should explain the exception. Similar scenes inside a domain should
use the same annotation geometry for the same visual primitive unless they
document a real task-specific reason.

Avoid mixed point/box annotation. If a task appears to need mixed geometry,
reconsider the task annotation contract first. Add a new public annotation type
only when the contract cannot be expressed with the current homogeneous types.

## 5) Scoring Semantics

1. Box-set matching uses raw IoU values directly; there is no acceptance
   threshold.
2. Pixel point similarity is
   `exp(-ln(2) * (distance / half_life_px)^2)`.
3. When source image size is available,
   `half_life_px = clamp(0.035 * sqrt(width^2 + height^2), 20, 80)`.
   A supplied `point_half_life_px` overrides this policy, and missing image-size
   metadata falls back to 32 px.
4. Set contracts divide matched similarity by `max(pred_count, gt_count, 1)`,
   so missing and extra witnesses are penalized.
5. Sequence contracts compare by position and divide by the max sequence length.
6. Map contracts divide shared-key similarity by the union of predicted
   and target keys, so missing and extra role keys are penalized.
7. `segment_soft_distance_v0` and `segment_set_soft_distance_v0` treat each
   segment as undirected, so reversed endpoints score as the same witness.

## 6) Update Rules
When a public answer or annotation reward contract changes:

1. update `src/trace_tasks/core/reward_contracts.py`,
2. update `src/trace_tasks/core/reward_scoring.py` if scoring behavior changes,
3. update this document,
4. refresh affected tests and task documentation in the same patch.
