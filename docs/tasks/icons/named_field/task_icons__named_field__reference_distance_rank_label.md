# `task_icons__named_field__reference_distance_rank_label`

## Identity
- domain: `icons`
- scene_id: `named_field`
- task: `named_reference_distance_rank_label`
- module: `src/trace_tasks/tasks/icons/named_field/reference_distance_rank_label.py`
- prompt bundle: `src/trace_tasks/resources/prompts/icons/named_field/icons_named_field_v1.json`

## Program Contract

Program: `selection.ranked_item(scene=named_field, scope=labeled_option_icons, metric=center_distance_to_reference, ranks=closest|second_closest|farthest, output=option_letter)`

Candidate set: the visible icon instances, icon attributes, fields, grids, paths, panels, reference items, and labeled option cards inside the `labeled_option_icons` objective scope.
Operands: visible scene state and prompt-bound operands named by `named_field`, `labeled_option_icons`, `metric`, `center_distance_to_reference`, `ranks`, `closest`, `second_closest`, `farthest`.
Operation: evaluate `selection.ranked_item` over the candidate set using the visible icon attributes, positions, relationships, transforms, counts, comparisons, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `closest_to_named_reference_label`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`

## Scene And Query
The task renders one panel with exactly one unique named
reference icon, six option icons labeled `A`..`F`, and `4..8` other icons.
The reference is uniquely identified by its
prompt-named color and procedural shape, for example `red [#E63232] star`.

Supported query ids:
- `closest_to_named_reference_label`
- `second_closest_to_named_reference_label`
- `farthest_from_named_reference_label`

The six labeled option icons are the only answer options. Other icons are not
included in the distance-rank candidate set.

## Answer Contract
- `answer_gt.type = option_letter`
- answer support is exactly `A|B|C|D|E|F`
- the answer is the label of the candidate at the requested center-to-center
  distance rank from the named reference icon

## Annotation Contract
- `annotation_gt.type = bbox_map`
- annotation contains `reference_icon` for the named reference icon bbox and
  `selected_candidate` for the selected labeled candidate icon bbox
- candidate distance ranks are separated from adjacent ranks by the configured
  `distance_rank_margin_px`

## Trace Contract
- `scene_ir.entities` contains all reference, candidate, and other icons.
- Candidate entities include visible `label`, `distance_to_reference_px`, and
  `distance_rank`.
- `execution_trace.sorted_candidate_labels_by_distance` records the verifier
  order used to derive the answer.
- `projected_annotation.bbox_map` is derived from the same rendered
  reference and selected candidate bboxes.
- `render_spec.style.text_legibility` records validated panel-header and
  candidate-label text roles for the visible `A`..`F` option labels.

## Prompt Contract
- `scene_key = single_scene_counting`
- `task_key = counting_query`
- prompts ask which labeled icon is closest, second closest, or farthest from
  the unique named reference icon
- Both output modes include contract-valid JSON
  examples
