# `task_pages__profile_card_grid__filtered_ranked_profile_label`

## Identity
1. Domain: `pages`
2. Scene id: `profile_card_grid`
3. Source scene: `profile_card_grid`
4. Task id: `task_pages__profile_card_grid__filtered_ranked_profile_label`

## Contract
1. Objective: identify the profile at a requested rank after filtering by a visible categorical field and sorting by a visible numeric field.
2. Public task contract: `filtered_ranked_profile_label`
3. Supported `query_id` values: `filtered_highest_profile_label`, `filtered_lowest_profile_label`, `filtered_nth_highest_profile_label`, `filtered_nth_lowest_profile_label`
4. Answer type: `string`
5. Annotation schema: `bbox`
6. Annotation witness: scalar box around the selected profile card; filter field/value boxes, ranked field/value boxes, and filtered candidate values stay in trace metadata.
7. Query argument axes: filter field label, filter field value, rank direction, rank position, target numeric field label, filtered candidate values, card count, and scene variant. Highest/lowest queries bind `rank_position=1`; nth-rank queries sample a supported non-extremal rank position.

## Program Contract
- `profile_card_grid_filtered_ranked_profile_label(filter_field_label, filter_field_value, field_label, rank_direction, rank_position); output=profile_name_string; annotation=bbox(selected_profile_card); scene=profile_card_grid; scope=one profile-card grid page`

## Reasoning Operations

Families: `filtering`, `ranking`

## Prompt + Trace
1. Prompt bundle: `pages_profile_card_grid_v1`
2. Scene key: `profile_card_grid`
3. Task key: `profile_attribute_lookup_query`
4. Prompt query keys: `filtered_highest_profile_label`, `filtered_lowest_profile_label`, `filtered_nth_highest_profile_label`, `filtered_nth_lowest_profile_label`
5. Trace records visible profile cards, repeated categorical filter groups, numeric candidate values inside the selected filter group, selected rank direction/position, target profile payload, final text boxes, layout metadata, and prompt metadata.
6. Generation is deterministic from `instance_seed`; answer and annotation come from the finalized render metadata.
