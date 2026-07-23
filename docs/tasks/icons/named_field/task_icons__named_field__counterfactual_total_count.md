# `task_icons__named_field__counterfactual_total_count`

## Identity
- domain: `icons`
- scene_id: `named_field`
- task: `named_shape_counterfactual_count`
- module: `src/trace_tasks/tasks/icons/named_field/counterfactual_total_count.py`
- prompt bundle: `src/trace_tasks/resources/prompts/icons/named_field/icons_named_field_v1.json`

## Program Contract

Program: `count.counterfactual(scene=named_field, scope=visible_icons, edit=shape_removal, target=total_icon_count, output=count)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `visible_icons` objective scope.
Operands: visible scene state and prompt-bound operands named by `named_field`, `visible_icons`, `edit`, `shape_removal`, `total_icon_count`.
Operation: evaluate `count.counterfactual` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `state_update`

## Scene And Query
The task renders one panel containing procedurally generated
named shape icons. The prompt describes a hypothetical edit and asks for the
integer count after applying that edit mentally.

Supported public query ids:
- `single`

The fixed internal hypothetical branch is `total_count_after_shape_removal`. Trace metadata
records it as `query_spec.internal_query_id` and
`query_spec.params.internal_query_id`; it selects the shape-removal edit.

The target shape support is the full procedural named-icon vocabulary in
`src/trace_tasks/tasks/icons/shared/procedural_named_icons.py`. Icons also sample a
rendered `fill_style` from `solid`, `striped`, and `dotted`;
fill style is metadata/visual variation only for this task and does not affect
the hypothetical edit semantics.

## Answer Contract
- `answer_gt.type = integer`
- default answer support is `1..8`
- value is the final count after applying the stated hypothetical edit

## Annotation Contract
- `annotation_gt.type = bbox_set`
- one `[x0, y0, x1, y1]` pixel bounding box for every visible icon that contributes to
  the final count
- annotation boxes are sorted by the witnesses in reading order
- hypothetical additions are intentionally not sampled because they would not
  have visible bbox annotation

## Trace Contract
- `scene_ir.entities` contains one entity for each rendered procedural icon,
  including its `fill_style`.
- `scene_ir.relations.role_by_instance_id` records each icon's
  counterfactual role and whether it is counted after the edit.
- `query_spec.params.query_id` is `single`.
- `query_spec.params.internal_query_id` records the fixed hypothetical branch.
- `execution_trace.target_answer` equals the integer answer.
- `render_map.counted_instance_ids`, `witness_symbolic.counted_instance_ids`,
  and `projected_annotation.bbox_set` are derived from the same rendered
  instances.

## Prompt Contract
- `scene_key = single_scene_counting`
- `task_key = counting_query`
- prompts ask for the resulting total icon count after a removal edit
- Both output modes include contract-valid JSON
  examples
