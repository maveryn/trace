# `task_puzzles__rubiks_net__rubiks_move_result_label`

## Public Taxonomy
1. Domain: `puzzles`
2. Scene id: `rubiks_net`
3. Source scene: `rubiks_net`
4. Task id: `task_puzzles__rubiks_net__rubiks_move_result_label`

## Query Contract
1. Supported `query_id`: `direct_sequence_result_label`, `inverse_sequence_result_label`
2. Internal question formats: `direct_sequence_result_label`, `inverse_sequence_result_label`
3. Direct query asks for the candidate net after applying the shown move sequence.
4. Inverse query asks for the candidate net after undoing the shown base sequence.
5. Internal variation: direct sequence length, move tokens, option order, scene treatment, and style are generation/render metadata.

## Program Contract

Program: `select_label(candidate_net_option, rule=apply_direct_or_inverse_move_sequence_to_cube_net); scene=rubiks_net; scope=rubiks_move_result_label`

Candidate set: the visible cube-net stickers, face labels, move sequence, target face/sticker cues, and labeled options inside the `rubiks_move_result_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `candidate_net_option`, `apply_direct_or_inverse_move_sequence_to_cube_net`, `rubiks_net`, `rubiks_move_result_label` plus the active `query_id` branch.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; the capital-letter label on the correct candidate-net panel.
Annotation witnesses: `annotation` uses the `bbox` schema; one bbox around the selected candidate-net option panel.
Query ids: `direct_sequence_result_label`, `inverse_sequence_result_label`.

## Reasoning Operations

Families: `state_update`

## Answer And Annotation
1. `answer_gt.type = option_letter`
2. `answer_gt.value` is the capital-letter label on the correct candidate-net panel.
3. `annotation_gt.type = bbox`
4. Annotation schema: scalar `bbox`
5. Annotation target: one bbox around the selected candidate-net option panel.
6. `scalar_annotation_checked = true`.

## Trace Contract
1. `execution_trace.rubiks_rule_code = rubiks_sequence_result_match`.
2. `execution_trace.query_sequence`, `execution_trace.base_sequence`, and `execution_trace.option_specs` record the unique candidate states.
3. `render_map.option_panel_bboxes_px` contains the selected option panel bbox projected into `annotation_gt`.

## Prompt Contract
1. Bundle: `puzzles_rubiks_net_v1`
2. Scene key: `rubiks_net`
3. Task key: `rubiks_move_result_label_query`
4. Query keys: `direct_sequence_result_label`, `inverse_sequence_result_label`
