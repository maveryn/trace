# `task_pages__profile_card_grid__field_ranked_profile_label`

## Identity
1. Domain: `pages`
2. Scene id: `profile_card_grid`
3. Source scene: `profile_card_grid`
4. Task id: `task_pages__profile_card_grid__field_ranked_profile_label`

## Contract
1. Objective: identify the profile at a requested rank after sorting by a visible numeric field.
2. Public task contract: `field_ranked_profile_label`
3. Supported `query_id` values: `highest_field_profile_label`, `lowest_field_profile_label`, `nth_highest_field_profile_label`, `nth_lowest_field_profile_label`
4. Answer type: `string`
5. Annotation schema: `bbox`
6. Annotation witness: scalar box around the selected profile card; target field/value boxes and ranked candidate values stay in trace metadata.
7. Query argument axes: rank direction, rank position, target numeric field label, candidate profile values, card count, and scene variant. Highest/lowest queries bind `rank_position=1`; nth-rank queries sample a supported non-extremal rank position.

## Program Contract
- `profile_card_grid_field_ranked_profile_label(field_label, rank_direction, rank_position); output=profile_name_string; annotation=bbox(selected_profile_card); scene=profile_card_grid; scope=one profile-card grid page`

## Reasoning Operations

Families: `ranking`

## Prompt + Trace
1. Prompt bundle: `pages_profile_card_grid_v1`
2. Scene key: `profile_card_grid`
3. Task key: `profile_attribute_lookup_query`
4. Prompt query keys: `highest_field_profile_label`, `lowest_field_profile_label`, `nth_highest_field_profile_label`, `nth_lowest_field_profile_label`
5. Trace records visible profile cards, numeric candidate values, selected rank direction/position, target profile payload, final text boxes, layout metadata, and prompt metadata.
6. Generation is deterministic from `instance_seed`; answer and annotation come from the finalized render metadata.
