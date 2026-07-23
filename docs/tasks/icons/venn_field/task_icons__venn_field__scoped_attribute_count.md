# `task_icons__venn_field__scoped_attribute_count`

## Program Contract

Program: `counting.set_region_predicate_count(scene=venn_field, scope=procedural_named_icons, region=both_circles|either_circle|exactly_one_circle|outside_both_circles, target=shape|color_and_shape, membership=center_in_region, output=count)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `procedural_named_icons` objective scope.
Operands: visible scene state and prompt-bound operands named by `venn_field`, `procedural_named_icons`, `region`, `both_circles`, `either_circle`, `exactly_one_circle`, `outside_both_circles`, `shape`, `color_and_shape`, `membership`, `center_in_region`.
Operation: evaluate `counting.set_region_predicate_count` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `inside_both_circles_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `spatial_relations`

## Identity

- Domain: `icons`
- Scene id: `venn_field`
- Task id: `task_icons__venn_field__scoped_attribute_count`
- Objective contract: count prompt-named procedural icons whose centers satisfy a visible overlapping-circle region predicate.
- Module: `src/trace_tasks/tasks/icons/venn_field/scoped_attribute_count.py`
- Prompt bundle: `src/trace_tasks/resources/prompts/icons/venn_field/icons_venn_field_v1.json`

## Contract

- Supported `query_id` values:
  - `inside_both_circles_count`: count target icons centered inside both marked circles.
  - `inside_either_circle_count`: count target icons centered inside at least one marked circle.
  - `inside_exactly_one_circle_count`: count target icons centered inside exactly one marked circle.
  - `outside_both_circles_count`: count target icons centered outside both marked circles.
- Answer schema: `integer`.
- Annotation schema: `bbox_set`.
- The image contains one panel with procedurally generated named icons and two overlapping marked circles.
- The target predicate is either one quoted shape name, such as `"bell" icons`, or one named color plus quoted shape name, such as `red [#E63232] "bell" icons`.
- Target predicate mode, target shape, target color, fill style, icon count, and render style are generation metadata, not public query ids.
- Circle boundary margins keep icon boxes away from queried inside/outside boundaries.

## Generation

- `target_count` defaults to `1..5`.
- `object_count` defaults to `8..13`.
- At least one target icon is also placed outside the counted region as an opposite-region distractor.
- Non-target distractors include shape and color confounds.
- Query selection is task-owned and uniform unless `query_id` is explicitly supplied.
- Generation rejects samples that cannot place all icon boxes with the requested Venn membership and overlap constraints.

## Prompt

- Prompt bundle: `icons_venn_field_v1`
- `scene_key`: `venn_field_scene`
- `task_key`: `scoped_attribute_count`
- Query templates ask for one of the four overlapping-circle region predicates.
- Answer JSON shape: `{"answer":2}`
- Answer+annotation JSON shape: `{"annotation":[[116,128,168,180],[634,302,686,354]],"answer":2}`

## Annotation

- `bbox_set` is used because the number of counted witnesses can be one or more.
- Boxes mark counted target icons only, sorted top-to-bottom then left-to-right.
- The task is not scalar-annotation eligible because counted witness cardinality varies by generated answer.

## Trace

- `scene_ir.entities` contains one entity for each rendered procedural icon, including `shape_id`, `shape_name`, `color_name`, `fill_style`, `venn_category`, and target/count flags.
- `scene_ir.relations.venn` records both circle geometries.
- `query_spec.params.target_attribute_mode` records whether the target is shape-only or color+shape.
- `execution_trace.counted_venn_categories` records the Venn categories counted by the active query.
- `render_map.counted_instance_ids`, `witness_symbolic.counted_instance_ids`, and `projected_annotation.bbox_set` are derived from the same rendered instances.

## Tests

- Behavior and trace tests: `tests/test_icons_counting_named_shape_venn_region_count.py`
- Config tests: `tests/test_icons_scene_config.py`
- Prompt bundle tests: `tests/test_prompt_system.py`
- Source-layout contract checks: `tests/test_public_source_layout_contracts.py`
