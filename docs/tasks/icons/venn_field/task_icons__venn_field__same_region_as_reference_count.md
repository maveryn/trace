# `task_icons__venn_field__same_region_as_reference_count`

## Program Contract

Program: `counting.dynamic_reference_region_predicate_count(scene=venn_field, scope=procedural_named_icons, reference=marked_icon, region=reference_icon_venn_category, target=shape|color_and_shape, membership=center_in_region, output=count)`

Candidate set: the visible procedural named-icon instances in one `venn_field`
panel.
Operands: two overlapping marked circles, one visually marked reference icon,
the reference icon's Venn category, and a prompt-bound target predicate over
icon shape or color plus shape.
Operation: classify the marked reference icon into exactly one of four Venn
regions by center position: left circle only, right circle only, overlap of
both circles, or outside both circles. Count target icons whose centers fall in
that same region.
Output binding: `answer` uses the `integer` schema; generation binds a unique
final count.
Annotation witnesses: `annotation` uses the `bbox_set` schema and contains only
the counted target icons.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Identity

- Domain: `icons`
- Scene id: `venn_field`
- Task id: `task_icons__venn_field__same_region_as_reference_count`
- Objective contract: count prompt-named procedural icons in the same
  center-based Venn region as one marked reference icon.
- Module: `src/trace_tasks/tasks/icons/venn_field/same_region_as_reference_count.py`
- Prompt bundle: `src/trace_tasks/resources/prompts/icons/venn_field/icons_venn_field_v1.json`

## Contract

- Supported `query_id` values:
  - `single`: count target icons in the same Venn region as the marked
    reference icon.
- Answer schema: `integer`.
- Annotation schema: `bbox_set`.
- The image contains one panel with procedurally generated named icons and two
  overlapping marked circles.
- The target predicate is either one quoted shape name, such as `"bell" icons`,
  or one named color plus quoted shape name, such as `red [#E63232] "bell"
  icons`.
- The prompt defines the four possible Venn regions and says that region
  membership is determined by icon centers.
- The marked reference icon is constructed so it does not satisfy the target
  predicate; it is a scope operand, not a counted answer witness.
- The sampled reference region, target predicate mode, target shape, target
  color, fill style, icon count, and render style are generation metadata, not
  public query ids.

## Generation

- `target_count` defaults to `1..5`.
- `object_count` defaults to `8..13`.
- One marked non-target reference icon is placed in a sampled Venn category.
- Exactly `target_count` target icons are placed in the same category as the
  reference icon.
- At least one target icon is also placed outside the reference category as a
  distractor.
- Non-target distractors include shape and color confounds.
- Generation rejects samples that cannot place all icon boxes with the
  requested Venn membership and overlap constraints.

## Prompt

- Prompt bundle: `icons_venn_field_v1`
- `scene_key`: `venn_field_scene`
- `task_key`: `same_region_as_reference_count`
- `query_key`: `single`
- The query templates explicitly define the four Venn regions and center-based
  membership.
- Answer JSON shape: `{"answer":2}`
- Answer+annotation JSON shape:
  `{"annotation":[[116,128,168,180],[634,302,686,354]],"answer":2}`

## Annotation

- `bbox_set` is used because the number of counted witnesses varies by
  generated answer.
- Boxes mark counted target icons only, sorted top-to-bottom then
  left-to-right.
- The marked reference icon bbox is stored in trace metadata, not annotation,
  because the reference is a scope operand rather than an answer witness.
- The task is not scalar-annotation eligible because counted witness
  cardinality varies by generated answer.

## Trace

- `scene_ir.entities` contains one entity for each rendered procedural icon,
  including `shape_id`, `shape_name`, `color_name`, `fill_style`,
  `venn_category`, `is_reference`, and target/count flags.
- `scene_ir.relations.venn` records both circle geometries.
- `query_spec.params.reference_venn_category` records the sampled reference
  category.
- `query_spec.params.target_attribute_mode` records whether the target is
  shape-only or color+shape.
- `render_map.reference_instance_id` and `render_map.reference_bbox_px` record
  the marked reference icon.
- `render_map.counted_instance_ids`, `witness_symbolic.counted_instance_ids`,
  and `projected_annotation.bbox_set` are derived from the same rendered
  instances.

## Tests

- Behavior and trace tests:
  `tests/test_icons_counting_named_shape_venn_region_count.py`
- Config tests: `tests/test_icons_scene_config.py`
- Prompt bundle tests: `tests/test_prompt_system.py`
- Source-layout contract checks:
  `tests/test_public_source_layout_contracts.py`
