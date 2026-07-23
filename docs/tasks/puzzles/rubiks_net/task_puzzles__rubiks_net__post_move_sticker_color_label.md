# `task_puzzles__rubiks_net__post_move_sticker_color_label`

## Public Taxonomy
1. Domain: `puzzles`
2. Scene id: `rubiks_net`
3. Source scene: `rubiks_net`
4. Task id: `task_puzzles__rubiks_net__post_move_sticker_color_label`

## Query Contract
1. Supported `query_id`: `single`
2. Internal question format: `post_move_sticker_color_label`
3. Prompt asks for the color-swatch option matching one sticker after applying a visible move sequence.
4. Internal variation: move count, move tokens, target coordinate, palette, option order, scene treatment, and style are generation/render metadata.

## Program Contract

Program: `select_label(color_swatch_option, rule=apply_move_sequence_then_read_sticker_color); scene=rubiks_net; scope=post_move_sticker_color_label`

Candidate set: the visible cube-net stickers, face labels, move sequence, target face/sticker cues, and labeled options inside the `post_move_sticker_color_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `color_swatch_option`, `apply_move_sequence_then_read_sticker_color`, `rubiks_net`, `post_move_sticker_color_label`.
Operation: evaluate `select_label` over the candidate set using the visible states, constraints, transforms, comparisons, counts, paths, or option-selection rules encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; the capital-letter label on the correct swatch option panel.
Annotation witnesses: `annotation` uses the `bbox` schema; one bbox around the selected option panel.
Query ids: `single`.

## Reasoning Operations

Families: `state_update`

## Answer And Annotation
1. `answer_gt.type = option_letter`
2. `answer_gt.value` is the capital-letter label on the correct swatch option panel.
3. `annotation_gt.type = bbox`
4. Annotation schema: scalar `bbox`
5. Annotation target: one bbox around the selected option panel.
6. `scalar_annotation_checked = true`.

## Trace Contract
1. `execution_trace.rubiks_rule_code = post_move_sticker_color_readout`.
2. `execution_trace.query_sequence` records the applied move sequence.
3. `render_map.option_panel_bboxes_px` contains the selected option panel bbox projected into `annotation_gt`.

## Prompt Contract
1. Bundle: `puzzles_rubiks_net_v1`
2. Scene key: `rubiks_net`
3. Task key: `post_move_sticker_color_label_query`
4. Query key: `post_move_sticker_color_label`
