# `task_pages__map__destination_after_directions_label`

## Identity
1. Domain: `pages`
2. Scene id: `map`
3. Source scene: `map`
4. Task id: `task_pages__map__destination_after_directions_label`

## Contract
1. Objective: identify the destination landmark reached by starting at a named landmark and following visible map directions.
2. Public task contract: `destination_after_directions_label`
3. Supported `query_id` values: `single`
4. Answer type: `string`
5. Annotation schema: `bbox_sequence`
6. Annotation witness: ordered landmark boxes along the followed route, from start through destination.
7. Query argument axes: start landmark, direction sequence, route length, landmark count, and map layout jitter/style.

## Program Contract
- `map_destination_after_directions(start_label, direction_sequence); output=string_visible_landmark_label; annotation=bbox_sequence(route_landmarks_ordered); scene=map; scope=one printed campus map`

## Reasoning Operations

Families: `spatial_relations`, `topology`, `state_update`

## Prompt + Trace
1. Prompt bundle: `pages_map_v1`
2. Scene key: `printed_map`
3. Task key: `map_navigation_query`
4. Prompt query key: `destination_after_directions`
5. Trace records the sampled map graph, visible landmark labels, route landmark ids, direction text, final rendered bboxes, sampled style metadata, and layout geometry.
6. Generation is deterministic from `instance_seed`; answers and annotation come from the finalized map render metadata.
