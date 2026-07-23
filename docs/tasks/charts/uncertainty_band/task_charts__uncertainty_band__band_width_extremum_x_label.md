# `task_charts__uncertainty_band__band_width_extremum_x_label`

## Public Contract

1. Domain: `charts`
2. Scene: `uncertainty_band`
3. Source file: `src/trace_tasks/tasks/charts/uncertainty_band/band_width_extremum_x_label.py`
4. Prompt assets: `src/trace_tasks/resources/prompts/charts/uncertainty_band/charts_uncertainty_band_v1.json`
5. Supported sampled `query_id`: `widest_band_x_label`, `narrowest_band_x_label`

## Program Contract

Program: `arg_extreme(x_label, width(vertical_interval(target_series_band_at_x)), direction={widest,narrowest}); output=string_label; annotation=segment(answer_band_lower_upper_span); scene=uncertainty_band; scope=band_width_extremum_x_label`

Candidate set: the visible uncertainty bands, central series marks, and x labels inside the `band_width_extremum_x_label` objective scope.
Operands: prompt-bound labels, categories, series names, thresholds, intervals, references, and encoded chart values, plus the active query id's comparator, direction, target role, or extremum focus when present.
Operation: evaluate `arg_extreme` over the candidate set using the filters, comparisons, aggregations, rankings, projections, or counterfactual edits named in the program expression; generation enforces a unique final answer.
Output binding: `answer` is the `unspecified` value bound by `string_label`.
Annotation witnesses: `unspecified` witnesses bound by `segment(answer_band_lower_upper_span)`. The Annotation Contract below defines the prompt-facing witnesses.
Query ids: `narrowest_band_x_label`, `widest_band_x_label`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Contract Notes

This task uses the current source layout. The public task file owns target-series selection, extremum direction, answer binding, annotation binding, query metadata, and prompt slots; scene-local shared code only provides uncertainty-band data structures, rendering, prompt, and projection primitives.

## Query Details

| Query id | Program signature | Answer schema | Annotation schema |
|---|---|---|---|
| `narrowest_band_x_label` | `argmin_x(width(vertical_interval(target_series_band_at_x)))` | `string_label` | `segment` |
| `widest_band_x_label` | `argmax_x(width(vertical_interval(target_series_band_at_x)))` | `string_label` | `segment` |
