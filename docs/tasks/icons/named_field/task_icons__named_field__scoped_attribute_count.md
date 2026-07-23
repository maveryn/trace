# `task_icons__named_field__scoped_attribute_count`

## Identity
- domain: `icons`
- scene_id: `named_field`
- task: `named_shape_region_count`
- module: `src/trace_tasks/tasks/icons/named_field/scoped_attribute_count.py`
- prompt bundle: `src/trace_tasks/resources/prompts/icons/named_field/icons_named_field_v1.json`

## Program Contract

Program: `count.scoped_attribute(scene=named_field, scope=marked_region_or_band_or_quadrant_or_shelf, attribute=shape, output=count)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `marked_region_or_band_or_quadrant_or_shelf` objective scope.
Operands: visible scene state and prompt-bound operands named by `named_field`, `marked_region_or_band_or_quadrant_or_shelf`, `attribute`, `shape`.
Operation: evaluate `count.scoped_attribute` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `inside_shape_count`.

## Reasoning Operations

Families: `filtering`, `counting`

## Scene And Query
The task renders one panel containing procedurally generated
named shape icons plus a visible marked region. The prompt names one procedural
shape and asks how many matching icons are inside or outside the queried region.

Supported query ids:
- `inside_shape_count`: count target-shape icons inside a marked rectangle or
  ellipse.
- `outside_shape_count`: count target-shape icons outside a marked rectangle or
  ellipse.
- `inside_band_count`: count target-shape icons inside the band between two
  parallel lines.
- `outside_band_count`: count target-shape icons outside the band between two
  parallel lines.
- `inside_quadrant_count`: count target-shape icons inside the highlighted
  quadrant.
- `inside_shelf_count`: count target-shape icons inside the highlighted shelf
  row.

The target shape support is the full procedural named-icon vocabulary in
`src/trace_tasks/tasks/icons/shared/procedural_named_icons.py`. Icons also sample a
rendered `fill_style` from `solid`, `striped`, and `dotted`;
fill style is metadata/visual variation only for this task and does not affect
region membership.

## Answer Contract
- `answer_gt.type = integer`
- default answer support is `1..5`
- value is the number of target-shape icons whose centers satisfy the active
  region predicate

## Annotation Contract
- `annotation_gt.type = bbox_set`
- one `[x0, y0, x1, y1]` pixel bounding box for every counted target-shape icon
- target-shape icons are placed with bbox clearance from the queried boundary,
  so counted and non-counted targets do not touch or cross it
- annotation boxes are sorted by the witnesses in reading order

## Trace Contract
- `scene_ir.entities` contains one entity for each rendered procedural icon,
  including its `fill_style` and `inside_region` metadata.
- `scene_ir.region` records the marked region geometry.
- `query_spec.params.query_id` records the selected region query.
- `execution_trace.target_answer` equals the integer answer.
- `render_map.counted_instance_ids`, `witness_symbolic.counted_instance_ids`,
  and `projected_annotation.bbox_set` are derived from the same rendered
  instances.

## Prompt Contract
- `scene_key = single_scene_counting`
- `task_key = counting_query`
- prompts ask for the count of one named shape inside or outside a visible
  region
- Both output modes include contract-valid JSON
  examples
