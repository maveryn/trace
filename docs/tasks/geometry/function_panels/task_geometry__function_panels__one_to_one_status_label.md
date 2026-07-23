# `task_geometry__function_panels__one_to_one_status_label`

## Contract
1. Domain: `geometry`
2. Scene id: `function_panels`
3. Public task id: `task_geometry__function_panels__one_to_one_status_label`
4. Supported `query_id`: `single`
5. Answer schema: `option_letter`
6. Annotation schema: `bbox`

## Program Contract
- `label(select_panel(candidate_coordinate_relations, relation_is_one_to_one_function)); scene=function_panels; scope=one_to_one_status_label`

## Reasoning Operations

Families: `spatial_relations`, `matching`

## Prompt Bundle
- Prompt text is loaded from `geometry_analytical_function_property_v1`.
- Prompt modes: `answer_only` and `answer_and_annotation`.

## Annotation
The annotation is the scalar pixel bounding box `[x0,y0,x1,y1]` of the selected panel.

## Determinism
Generation is deterministic for a fixed seed, params, config, and prompt bundle version.

## Source
- Config: `src/trace_tasks/resources/configs/domains/geometry/function_panels.yaml`
- Task module: `src/trace_tasks/tasks/geometry/function_panels/one_to_one_status_label.py`
