# `task_icons__named_field__closer_to_reference_count`

## Identity
- domain: `icons`
- scene_id: `named_field`
- task: `closer_to_reference_count`
- module: `src/trace_tasks/tasks/icons/named_field/closer_to_reference_count.py`
- prompt bundle: `src/trace_tasks/resources/prompts/icons/named_field/icons_named_field_v1.json`

## Program Contract

Program: `count.reference_metric_relation(scene=named_field, scope=target_shape_icons, metric=center_distance_to_two_references, relation=closer_to_queried_reference, output=count)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `target_shape_icons` objective scope.
Operands: visible scene state and prompt-bound operands named by `named_field`, `target_shape_icons`, `metric`, `center_distance_to_two_references`, `relation`, `closer_to_queried_reference`.
Operation: evaluate `count.reference_metric_relation` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `closer_to_reference_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Scene And Query
The task renders one panel containing two larger reference
icons plus several icons of one prompt-named target shape. The references are
identified in the prompt by icon name, and there are no unrelated distractor
icon types in the scene.

Supported query ids:
- `closer_to_reference_count`

The prompt asks how many target-shape icons are closer to one named reference
icon than to the other named reference icon. The selected reference side is a
sampled operand parameter recorded as `queried_reference_label`, not a public
query-id split.

## Answer Contract
- `answer_gt.type = integer`
- default answer support is `0..4`
- default total target-icon support is `4..8`

## Annotation Contract
- `annotation_gt.type = bbox_set`
- annotation contains one `[x0, y0, x1, y1]` pixel bounding box for every counted target-shape icon
- reference icons are recorded in trace metadata but are not included in the
  counting annotation

## Trace Contract
- `scene_ir.entities` contains two reference entities and all target entities.
- Each target entity stores distances to references `A` and `B`, its closer
  internal reference key, and whether it was counted.
- `query_spec.params.closer_count_by_reference` stores both `A` and `B` counts,
  including zero values.
- `render_map.counted_instance_ids`, `witness_symbolic.counted_instance_ids`,
  and `projected_annotation.bbox_set` are derived from the same rendered target
  entities.
- The reference icons are not visibly letter-labeled; internal `A`/`B` keys are
  used only for trace bookkeeping.

## Prompt Contract
- `scene_key = single_scene_counting`
- `task_key = counting_query`
- prompts ask for the target-shape count closer to one prompt-named reference
  icon than to the other prompt-named reference icon
- Both output modes include contract-valid JSON
  examples
