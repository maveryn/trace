# `task_pages__map__landmark_after_route_step_label`

## Identity
1. Domain: `pages`
2. Scene id: `map`
3. Source scene: `map`
4. Task id: `task_pages__map__landmark_after_route_step_label`

## Contract
1. Objective: identify the landmark reached after a requested ordinal step on the highlighted route.
2. Public task contract: `landmark_after_route_step_label`
3. Supported `query_id` values: `single`
4. Answer type: `string`
5. Annotation schema: `bbox_sequence`
6. Annotation witness: ordered highlighted-route landmark boxes from the route start through the requested landmark.
7. Query argument axes: route start/end landmarks, requested step ordinal, route length, landmark count, and map layout jitter/style.

## Program Contract
- `map_landmark_after_route_step(route_start_label, route_end_label, step_ordinal); output=string_visible_landmark_label; annotation=bbox_sequence(route_landmarks_ordered_to_answer); scene=map; scope=one printed campus map`

## Reasoning Operations

Families: `ranking`, `topology`

## Prompt + Trace
1. Prompt bundle: `pages_map_v1`
2. Scene key: `printed_map`
3. Task key: `map_navigation_query`
4. Prompt query key: `landmark_after_route_step`
5. Trace records the sampled map graph, visible landmark labels, highlighted route ids, requested step ordinal, final rendered bboxes, sampled style metadata, and layout geometry.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized map render metadata.
