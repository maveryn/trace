# `task_pages__infographic__global_metric_ranked_item_label`

## Identity
1. Domain: `pages`
2. Scene id: `infographic`
3. Source path: `src/trace_tasks/tasks/pages/infographic/global_metric_ranked_item_label.py`
4. Task id: `task_pages__infographic__global_metric_ranked_item_label`

## Program Contract
1. Program schema: `selection.ranked_item(candidate_set=all_metric_cards, metric=printed_value, rank_direction=highest|lowest); scene=infographic; scope=one multi-section metric-card infographic`
2. Contract: rank every visible metric card in the infographic by printed value, then return the label of the card at the requested rank.
3. Public query ids: `nth_highest_metric_label`, `nth_lowest_metric_label`
4. Answer schema: `string`
5. Annotation schema: scalar `bbox` around the selected metric card.
6. Query argument axes: rank direction and rank position.
7. scalar_annotation_checked=true

## Reasoning Operations

Families: `ranking`

## Prompt + Trace
1. Prompt bundle: `pages_infographic_v1`
2. Scene key: `infographic_metric_arithmetic`
3. Task key: `metric_arithmetic_query`
4. Trace records ranked candidates, rank direction, rank position, selected label/value, and the selected metric-card bbox.
5. Generation guarantees unique printed values in the ranked candidate set.
