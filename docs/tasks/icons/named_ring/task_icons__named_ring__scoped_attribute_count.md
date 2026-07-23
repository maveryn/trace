# `task_icons__named_ring__scoped_attribute_count`

- domain: `icons`
- scene_id: `named_ring`
- task: `scoped_attribute_count`
- module: `src/trace_tasks/tasks/icons/named_ring/scoped_attribute_count.py`

## Program Contract

Program: `count.filtered_on_directed_ring_arc(scene=named_ring, scope=icons_strictly_between_markers, traversal=clockwise|counterclockwise, predicate=shape_equals_target, output=integer)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `icons_strictly_between_markers` objective scope.
Operands: visible scene state and prompt-bound operands named by `named_ring`, `icons_strictly_between_markers`, `traversal`, `clockwise`, `counterclockwise`, `predicate`, `shape_equals_target`.
Operation: evaluate `count.filtered_on_directed_ring_arc` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Annotation schema: `bbox_set`.
Query ids: `clockwise_arc_shape_count`, `counterclockwise_arc_shape_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `topology`

## Contract
1. The image shows one visible ring of procedural named icons.
2. Two endpoint icons are marked with visible labels `A` and `B`.
3. The prompt names one target icon shape in quotes and specifies clockwise or
   counterclockwise traversal from `A` to `B`.
4. The answer is the number of target-shape icons strictly between markers `A`
   and `B` along the specified directed arc.
5. Markers `A` and `B` are excluded from the count.
6. `answer_gt.type = integer`.
7. `annotation_gt.type = bbox_set` over the bounding boxes of the counted
   target-shape icons only. `projected_annotation` mirrors this as typed
   bbox-set annotation with `bbox_set`, `pixel_bbox_set`, and derived
   `pixel_point_set`.

## Query IDs
- `clockwise_arc_shape_count`
- `counterclockwise_arc_shape_count`

## Generation
Default ring size is `12..22` icons. Default answer support is `0..6`.
Default directed arc span is `3..12` icons strictly between the endpoint
markers.

The generator samples the answer first, then selects a feasible directed arc
and places exactly that many target-shape icons along the arc. Additional
target-shape icons appear outside the queried arc as distractors. Marker icons
are never the target shape.

Fill style and color are rendered as non-semantic visual variation.

## Trace
The trace records:
- every named icon with clockwise ring index, center, bbox, shape id/name, and
  marker role,
- marker indices for `A` and `B`,
- traversal direction,
- full clockwise shape order,
- directed arc indices,
- counted target indices and off-arc target distractor indices,
- render-style metadata for panel title and marker-label text legibility,
- query, answer, ring-size, arc-span, off-arc target, shape, and fill-style
  probability metadata.
