# `task_geometry__graph_paper__length_extremum_label`

## Contract
1. Domain: `geometry`
2. Scene id: `graph_paper`
3. Task id: `task_geometry__graph_paper__length_extremum_label`
4. Supported `query_id`: `largest`, `smallest`
5. Answer schema: `option_letter`
6. Annotation schema: `segment`

## Program Contract
- `select_segment_length_extremum(extremum={largest|smallest}, output_role=segment_label); scene=graph_paper; scope=labeled_segment_set`

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Prompt Bundle
- Prompt text is loaded from `geometry_graph_paper_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
The annotation is the selected visual segment as two pixel endpoints.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.
