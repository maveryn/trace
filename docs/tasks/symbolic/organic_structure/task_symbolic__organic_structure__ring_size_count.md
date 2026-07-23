# `task_symbolic__organic_structure__ring_size_count`

## Public Taxonomy
1. Domain: `symbolic`
2. Scene id: `organic_structure`
3. Task id: `task_symbolic__organic_structure__ring_size_count`

## Program Contract
Program: `organic_structure.ring_size_count(scene=organic_structure, scope=symbolic, target_ring_size=5|6, output=integer)`

Candidate set: all explicit ring records in one rendered organic line-angle structure.
Operands: each ring's vertex count and the sampled target ring size `5|6`.
Operation: count rings whose explicit ring size equals the target ring size.
Output binding: `answer` is the matching ring count as an integer.
Annotation schema: `bbox_set`.
Annotation witnesses: a homogeneous `bbox_set` of matching ring bboxes.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `topology`

## Answer And Annotation
1. `answer_gt.type = integer`
2. `answer_gt.value` is the number of visible rings whose polygon has the requested vertex count.
3. `annotation_gt.type = bbox_set`
4. Annotation contains one bounding box around each matching ring.
5. Configured answer support is `1..5`.

## Scene Constraints
1. The scene renders explicit pentagonal and hexagonal ring records in bent or branched linked-ring clusters; fused or linked ring boundaries must remain unambiguous.
2. Each counted ring is an explicit renderer record with a stable ring id; the task does not require inferring implicit rings outside those records.
3. Atom/group letters, implicit-carbon totals, molecular identity, formulas, and chemistry naming are not queried.
4. Basic line-angle constraints are enforced: atom valence at most four, no crossed bonds, and ring sizes limited to five and six.

## Trace Contract
1. `query_spec.query_id` is `single`; `query_spec.internal_query_id` is `ring_size_count`.
2. `execution_trace.target_ring_size` records the requested ring size.
3. `execution_trace.matching_ring_item_ids` records the ring ids counted in the answer.
4. `execution_trace.rings` records each ring id, ring size, atom ids, atom indices, and target-ring membership.
5. `render_map.ring_bboxes_px` records ring bboxes used for annotation projection.

## Prompt Contract
1. Bundle: `symbolic_organic_structure_v1`
2. Scene key: `organic_structure`
3. Task key: `organic_structure_ring_size_count`
