# `task_icons__named_field__single_attribute_membership_count`

## Identity
- domain: `icons`
- scene_id: `named_field`
- task: `named_shape_count`
- module: `src/trace_tasks/tasks/icons/named_field/single_attribute_membership_count.py`
- prompt bundle: `src/trace_tasks/resources/prompts/icons/named_field/icons_named_field_v1.json`

## Program Contract

Program: `count.single_attribute_membership(scene=named_field, scope=all_icons, attribute=shape, output=count)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `all_icons` objective scope.
Operands: visible scene state and prompt-bound operands named by `named_field`, `all_icons`, `attribute`, `shape`.
Operation: evaluate `count.single_attribute_membership` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Scene And Query
The task renders one panel containing procedurally generated
named shape icons. The prompt names one shape, such as `star`, `crescent`,
`plus sign`, or `lightning bolt`, and asks how many icons of that shape are
present.

The target shape support is the full procedural named-icon vocabulary in
`src/trace_tasks/tasks/icons/shared/procedural_named_icons.py`. Icons also sample a
non-semantic `fill_style` visual attribute from `solid`, `striped`, and `dotted`; this varies rendering but is not part of this task's
answer.

The scene samples several internal arrangement modes while keeping the same
public task id and answer contract: jittered/ordered grids, shelf rows,
collision-free scatter, clusters by shape, shape stacks, mixed stacks, and a
target-stack-with-oddballs mode. Stack modes may use moderately larger target counts
because the visual strategy is structural row/column counting rather than
pure visual search. Stack-family renderers place same-group icons as compact
row/column blocks with minimal internal gaps so a queried icon type can be
counted from its stack. The `target_stack_with_oddballs` arrangement renders
only one stack: the target shape icons plus exactly one odd non-target icon in
that same stack, with no additional distractor stacks.

## Answer Contract
- `answer_gt.type = integer`
- value is the count of icons whose `shape_id` equals the target shape
- default answer support is layout-dependent: non-stack layouts use smaller
  counts, and stack layouts are capped at moderate row/column counts

## Annotation Contract
- `annotation_gt.type = bbox_set`
- one `[x0, y0, x1, y1]` pixel bounding box for each counted target-shape icon
- annotation boxes are sorted by the witnesses in reading order

## Trace Contract
- `scene_ir.entities` contains one entity for each rendered procedural icon.
- Each entity records `shape_id`, prompt-facing `shape_name`, bbox, size,
  rotation, tint, `fill_style`, placement group, row/column layout
  coordinates, and icon noise metadata.
- `query_spec.params.arrangement_mode` records the sampled arrangement mode.
- `scene_ir.relations.arrangement_details` records stack/oddball details when
  applicable.
- `execution_trace.shape_counts[target_shape_id]` equals the integer answer.
- `render_map.counted_instance_ids`, `witness_symbolic.counted_instance_ids`,
  and `projected_annotation.bbox_set` are derived from the same rendered
  instances.

## Prompt Contract
- `scene_key = single_scene_counting`
- `task_key = counting_query`
- `query_key = named_shape_count`
- prompts ask for the count of one named procedural shape
- Both output modes include contract-valid JSON
  examples
