# `task_icons__named_ring__nearest_marker_target_count`

- domain: `icons`
- scene_id: `named_ring`
- task: `nearest_marker_target_count`
- module: `src/trace_tasks/tasks/icons/named_ring/nearest_marker_target_count.py`

## Program Contract

Program: `count.filtered_by_nearest_marker(scene=named_ring, scope=ring_icons_between_markers, markers=A|B, predicate=shape_equals_target and distance_to_A_less_than_distance_to_B, output=integer)`

Candidate set: visible named icons around the ring, excluding the marker labels
themselves.
Operands: marker labels `A` and `B`, the prompt-bound target icon type, and
shortest ring-step distance from each icon to each marker.
Operation: count target-type icons whose shortest ring distance to marker `A`
is smaller than their shortest ring distance to marker `B`; ties are not counted.
Output binding: `answer` uses the `integer` schema; generation binds a unique
count by construction.
Annotation witnesses: `annotation` uses the `bbox_set` schema over counted
target icons.
Annotation schema: `bbox_set`.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`, `logical_composition`, `spatial_relations`

## Contract
1. The image shows one visible ring of procedural named icons.
2. Two endpoint icons are marked with visible labels `A` and `B`.
3. The prompt names one target icon shape in quotes.
4. The answer is the number of target-shape icons closer to marker `A` than to
   marker `B` around the ring.
5. Target-shape icons equally far from `A` and `B` are excluded.
6. `answer_gt.type = integer`.
7. `annotation_gt.type = bbox_set` over the bounding boxes of the counted target
   icons only.

## Query IDs
- `single`

## Generation
Default ring size is `12..22` icons. Default answer support is `0..6`.
The generator samples marker positions with enough non-marker slots closer to
marker `A`, places exactly the requested number of target icons in that region,
and may place additional non-counted target icons closer to marker `B` or at tie
positions.

Fill style and color are rendered as non-semantic visual variation.

## Trace
The trace records:
- every named icon with clockwise ring index, center, bbox, shape id/name, and
  marker role,
- marker indices for `A` and `B`,
- shortest ring distances to each marker,
- close-to-A, close-to-B, and tie index partitions,
- counted target indices and non-counted target distractor indices,
- render-style metadata for marker-label text legibility,
- query, answer, ring-size, target-shape, and fill-style probability metadata.
