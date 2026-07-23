# `task_charts__part_whole__subset_denominator_share_value`

## Contract
1. Domain: `charts`
2. Scene id: `part_whole`
3. Supported `query_id`: `single`
4. Answer schema: `integer_value`
5. Annotation schema: `point_map`

## Implementation
1. Registered class: `trace_tasks.tasks.charts.part_whole.subset_denominator_share_value.ChartsCompositionSubsetDenominatorShareValueTask`
2. Prompt lookup domain/scene: `charts/part_whole`
3. Generation is deterministic from `instance_seed`, explicit params, prompt bundle, renderer config, and code versions.
4. Answers and annotation are produced from the same metadata execution trace.

## Program Contract

Program: `round(100 * value(target_category) / sum(value(category) for category in denominator_subset)); scene=part_whole; scope=subset_denominator_share_value`

Candidate set: the visible part-whole segments, slices, and category labels inside the `subset_denominator_share_value` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the task's prompt-bound target operands when present.
Operation: evaluate `round` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `unspecified`.
Annotation witnesses: `unspecified` witnesses bound by `see_annotation_contract`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `aggregation`, `formula_evaluation`

## Annotation Contract
Annotation maps every denominator-subset category label to an `[x,y]` pixel point at the center of its chart segment.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `single` | `round(100 * value(target_category) / sum(values(denominator_subset)))` | `integer_value` | `point_map` |
