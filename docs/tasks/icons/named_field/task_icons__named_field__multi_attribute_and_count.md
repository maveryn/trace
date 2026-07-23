# `task_icons__named_field__multi_attribute_and_count`

## Identity
- domain: `icons`
- scene_id: `named_field`
- task: `multi_attribute_and`
- module: `src/trace_tasks/tasks/icons/named_field/multi_attribute_and_count.py`
- prompt bundle: `src/trace_tasks/resources/prompts/icons/named_field/icons_named_field_v1.json`

## Program Contract

Program: `count.multi_attribute_and(scene=named_field, scope=all_icons, predicates=shape_and_secondary_attribute, secondary_attribute=color, output=count)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `all_icons` objective scope.
Operands: visible scene state and prompt-bound operands named by `named_field`, `all_icons`, `predicates`, `shape_and_secondary_attribute`, `secondary_attribute`, `color`.
Operation: evaluate `count.multi_attribute_and` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`

## Scene And Query
The task renders one panel containing procedurally generated
named shape icons. Each icon has a semantic procedural `shape_id` and a semantic
`color_name` sampled from the shared Trace named-color palette. Icons may also
render with non-semantic fill patterns (`solid`, `striped`, or `dotted`) as
visual variation, but fill style is not queried by this task.

The prompt names one shape and one color, then asks for the count satisfying
one Boolean predicate.

Supported public query ids:
- `single`

The fixed internal predicate key is `shape_and_color_count`; trace metadata records it
as `query_spec.internal_query_id` and `query_spec.params.internal_query_id`. It
selects the shape AND color predicate.

The target shape support is the full procedural named-icon vocabulary in
`src/trace_tasks/tasks/icons/shared/procedural_named_icons.py`. Color prompts use the shared named-color label with hex notation, such as
`red [#E63232]`, from `src/trace_tasks/tasks/shared/named_colors.py`.

The scene uses only non-stack named-icon arrangements: jittered/ordered grids,
shelf rows, collision-free scatter, and clusters by shape. Stack arrangements
are intentionally excluded because Boolean shape/color membership should be
answered by attribute filtering rather than row/column stack arithmetic.

## Answer Contract
- `answer_gt.type = integer`
- value is the count of icons satisfying the shape AND color predicate
- default answer support is `1..5`, with lower counts sampled more often to
  keep the generated answer distribution stable

## Annotation Contract
- `annotation_gt.type = bbox_set`
- one `[x0, y0, x1, y1]` pixel bounding box for each counted icon
- annotation boxes are sorted by the witnesses in reading order

## Trace Contract
- `scene_ir.entities` contains one entity for each rendered procedural icon.
- Each entity records `shape_id`, prompt-facing `shape_name`, `color_name`,
  `fill_style`, RGB tint, bbox, size, rotation, placement group, row/column layout
  coordinates, and icon-noise metadata.
- `query_spec.params.query_id` is `single`.
- `query_spec.params.internal_query_id` records the fixed Boolean predicate key.
- `query_spec.params.target_attribute_axis` records whether the secondary
  attribute is `color` or `fill_style`.
- `query_spec.params.partition_counts` records the four target-relative
  partitions: `both`, `shape_only`, `attribute_only`, and `neither`.
- `execution_trace.target_answer` equals the integer answer.
- `render_map.counted_instance_ids`, `witness_symbolic.counted_instance_ids`,
  and `projected_annotation.bbox_set` are derived from the same rendered
  instances.

## Prompt Contract
- `scene_key = single_scene_counting`
- `task_key = counting_query`
- prompts ask for the count of one named shape plus color-or-fill-style
  Boolean condition
- Both output modes include contract-valid JSON
  examples
