# `task_icons__named_strip__shape_run_length`

## Program Contract

Program: `counting.sequence_run_extremum_length(scene=named_strip, scope=procedural_named_icons, target=quoted_shape_name, extremum=longest|shortest, adjacency=consecutive_horizontal_cells, output=integer)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `procedural_named_icons` objective scope.
Operands: visible scene state and prompt-bound operands named by `named_strip`, `procedural_named_icons`, `quoted_shape_name`, `extremum`, `longest`, `shortest`, `adjacency`, `consecutive_horizontal_cells`.
Operation: evaluate `counting.sequence_run_extremum_length` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `longest_shape_run_length`.

## Reasoning Operations

Families: `counting`, `ranking`, `topology`

## Identity

- Domain: `icons`
- Scene id: `named_strip`
- Task id: `task_icons__named_strip__shape_run_length`
- Objective contract: measure the length of the longest or shortest consecutive run of the prompt-named icon shape in one horizontal strip.
- Module: `src/trace_tasks/tasks/icons/named_strip/shape_run_length.py`
- Prompt bundle: `src/trace_tasks/resources/prompts/icons/named_strip/icons_named_strip_v1.json`

## Contract

- Supported `query_id` values:
  - `longest_shape_run_length`: return the longest consecutive target-shape run length.
  - `shortest_shape_run_length`: return the shortest consecutive target-shape run length.
- Answer schema: `integer`.
- Annotation schema: `bbox_set`.
- The image contains one horizontal row of boxed procedural named icons.
- The prompt names one target icon shape in quotes.
- The target-shape run that determines the answer is unique by construction.
- Target shape, fill style, color palette, strip length, and render style are generation metadata, not public query ids.

## Generation

- Default strip length is `12..16`.
- Default longest-run answer support is `2..6`.
- Default shortest-run answer support is `1..5`.
- Longest-run rows contain one target run of the answer length and any other target runs are shorter.
- Shortest-run rows contain one target run of the answer length and at least one other target run that is longer.
- Fill style and color are rendered as non-semantic visual variation.

## Prompt

- Prompt bundle: `icons_named_strip_v1`
- `scene_key`: `named_strip_run_length`
- `task_key`: `shape_run_length`
- Query templates ask for either the longest or shortest consecutive run.
- Answer JSON shape: `{"answer":3}`
- Answer+annotation JSON shape: `{"annotation":[[220,132,270,182],[292,132,342,182],[364,132,414,182]],"answer":3}`

## Annotation

- `bbox_set` is used because the selected run can contain multiple icon witnesses.
- Boxes mark only the icons in the unique run that determines the answer, sorted top-to-bottom then left-to-right.
- The task is not scalar-annotation eligible because witness cardinality varies by generated answer.

## Trace

- `scene_ir.entities` contains one entity for each strip cell and rendered procedural named icon.
- `scene_ir.relations.target_runs` records all target-shape runs with start, end, and length.
- `execution_trace.selected_run_indices` and `render_map.selected_run_instance_ids` record the unique witness run.
- `query_spec.params` records query, answer, strip-length, shape, and fill-style probability metadata.
- `projected_annotation.bbox_set` is derived from the same selected rendered icons as the integer answer.

## Tests

- Behavior and trace tests: `tests/test_icons_sequence_named_shape_run_length_tasks.py`
- Config tests: `tests/test_icons_scene_config.py`
- Prompt bundle tests: `tests/test_prompt_system.py`
- Source-layout contract checks: `tests/test_public_source_layout_contracts.py`
