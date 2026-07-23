# `task_games__connect_four__column_disc_profile_label`

## Contract
1. Domain: `games`
2. Scene id: `connect_four`
3. Public task id: `task_games__connect_four__column_disc_profile_label`
4. Supported `query_id` values: `single`
5. Answer schema: `label_string`
6. Annotation schema: `bbox_set`
7. Program schema: `select(column_label, red_disc_count_in_column=target_red_count and yellow_disc_count_in_column=target_yellow_count); scene=connect_four; scope=column_disc_profile_label`

## Program Contract

Program: `select(column_label, red_disc_count_in_column=target_red_count and yellow_disc_count_in_column=target_yellow_count); scene=connect_four; scope=column_disc_profile_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `column_disc_profile_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `column_label`, `red_disc_count_in_column`, `target_red_count`, `yellow_disc_count_in_column`, `target_yellow_count`, `connect_four`, `column_disc_profile_label`.
Operation: evaluate `select` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`

## Generation Notes
1. Column labels are rendered below the board, and a unique column matches the requested red/yellow count profile.
2. Annotation contains bboxes for every occupied disc cell in the selected column, not the column label text.
3. The requested red and yellow counts are sampled as task operands and recorded in trace metadata.
