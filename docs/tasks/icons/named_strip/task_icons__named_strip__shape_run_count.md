# `task_icons__named_strip__shape_run_count`

## Identity

- Domain: `icons`
- Scene id: `named_strip`
- Task id: `task_icons__named_strip__shape_run_count`
- Objective contract: count the number of separate consecutive runs of one prompt-named icon shape in a horizontal strip.
- Module: `src/trace_tasks/tasks/icons/named_strip/shape_run_count.py`
- Prompt bundle: `src/trace_tasks/resources/prompts/icons/named_strip/icons_named_strip_v1.json`

## Program Contract

Program: `counting.sequence_run_count(scene=named_strip, scope=procedural_named_icons, target=quoted_shape_name, adjacency=consecutive_horizontal_cells, output=integer)`

Candidate set: the visible icon instances in one horizontal row of boxed procedural named icons.
Operands: visible icon shape identity in each cell and the prompt-bound target shape.
Operation: partition the target-shape cells into maximal consecutive horizontal runs and count those runs. Generation enforces the final answer in the configured range.
Output binding: `answer` uses the `integer` schema.
Annotation schema: `bbox_set`.
Annotation witnesses: `annotation` contains the icon-object bbox for the first icon in each counted target-shape run, sorted in reading order.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `topology`

## Contract

- The image contains one horizontal row of boxed procedural named icons.
- The prompt names one target icon shape in quotes.
- The answer is the number of separate consecutive runs of the target shape.
- Default answer support is `1..4`.
- Target shape, target-run lengths, fill style, color palette, strip length, and render style are generation metadata, not public query ids.

## Generation

- Default strip length is `12..16`.
- The generator constructs exactly `answer` target-shape runs.
- Adjacent target-shape runs are separated by at least one non-target icon.
- Run lengths vary as non-semantic visual variation; only the number of runs determines the answer.

## Prompt

- Prompt bundle: `icons_named_strip_v1`
- `scene_key`: `named_strip_run_length`
- `task_key`: `shape_run_count`
- `query_key`: `shape_run_count`
- Answer JSON shape: `{"answer":3}`
- Answer+annotation JSON shape:
  `{"annotation":[[116,132,166,182],[364,132,414,182],[580,132,630,182]],"answer":3}`

## Annotation

- `bbox_set` is used because the number of counted run witnesses varies by generated answer.
- Boxes mark only the first icon in each counted target-shape run.
- The task is not scalar-annotation eligible because witness cardinality varies by generated answer.

## Trace

- `scene_ir.entities` contains one entity for each strip cell and rendered procedural named icon.
- `scene_ir.relations.target_runs` records all target-shape runs with start, end, and length.
- `execution_trace.selected_run_start_indices` records the first cell index of each counted run.
- `render_map.selected_run_start_instance_ids` records the rendered icon ids used for annotation.
- `projected_annotation.bbox_set` is derived from the same rendered icons as the integer answer.

## Tests

- Behavior and trace tests: `tests/test_icons_sequence_named_shape_run_length_tasks.py`
- Config tests: `tests/test_icons_scene_config.py`
- Prompt bundle tests: `tests/test_prompt_system.py`
- Source-layout contract checks:
  `tests/test_public_source_layout_contracts.py`
