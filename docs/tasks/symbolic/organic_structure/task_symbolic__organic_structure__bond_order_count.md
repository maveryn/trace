# `task_symbolic__organic_structure__bond_order_count`

## Public Taxonomy
1. Domain: `symbolic`
2. Scene id: `organic_structure`
3. Task id: `task_symbolic__organic_structure__bond_order_count`

## Program Contract
Program: `organic_structure.bond_order_count(scene=organic_structure, scope=symbolic, target_bond_order=double|triple, output=integer)`

Candidate set: all explicit bond records in one rendered organic line-angle structure.
Operands: each bond's semantic bond order and the sampled target bond order `double|triple`.
Operation: count bonds whose semantic order equals the target bond order.
Output binding: `answer` is the matching bond count as an integer.
Annotation schema: `segment_set`.
Annotation witnesses: a homogeneous `segment_set` of semantic endpoint-to-endpoint bond segments for matching bonds.
Query ids: `single`.

## Reasoning Operations

Families: `counting`

## Answer And Annotation
1. `answer_gt.type = integer`
2. `answer_gt.value` is the number of visible bonds whose order matches `target_bond_order`.
3. `annotation_gt.type = segment_set`
4. Each segment is `[[x0, y0], [x1, y1]]` and connects the semantic endpoints of one matching bond.
5. Segments mark semantic bond endpoints, not every parallel stroke in a double/triple bond.
6. Configured answer support is `1..5`.

## Trace Contract
1. `query_spec.query_id` is `single`; `query_spec.internal_query_id` is `bond_order_count`.
2. `execution_trace.organic_metadata` records supported bond orders, scaffold id/family, and constraint policy.
3. `execution_trace.annotation_item_ids` records the rendered bond ids used for segment projection.
4. `render_map.bond_segments_px` exposes bond endpoints after final layout.
5. `execution_trace.bonds` records each bond and rendered order.

## Prompt Contract
1. Bundle: `symbolic_organic_structure_v1`
2. Scene key: `organic_structure`
3. Task key: `organic_structure_bond_order_count`
4. Prompt wording asks for visible double or triple bond notation. It must not ask for molecular formula, atom labels, implicit-carbon counts, stereochemistry, naming, or reaction reasoning.

## Determinism + Constraints
1. Deterministic generation and rendering from `instance_seed`.
2. Unique-answer policy: the answer is the exact count of rendered bonds with the sampled target order.
3. Reject/resample conditions: generation fails rather than silently changing the semantic contract if the requested answer support cannot fit.
4. The scene grammar enforces basic line-angle plausibility: atom valence at most four, linear unbranched triple-bond atoms, curated pentagon/hexagon rings, optional atom/substituent labels, and no crossed nonadjacent bonds.
