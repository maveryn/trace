# `task_games__backgammon__destination_count`

## Contract
1. Domain: `games`
2. Scene id: `backgammon`
3. Public task id: `task_games__backgammon__destination_count`
4. Supported `query_id` values: `legal_move_count`, `hit_move_count`, `blocked_destination_count`
5. Answer schema: `integer_count`
6. Annotation schema: `bbox_set`
7. Program schema: `count(filter(candidate_destinations(dice_rolls, board_state), destination_status)); scene=backgammon; scope=destination_count`

## Program Contract

Program: `count(filter(candidate_destinations(dice_rolls, board_state), destination_status)); scene=backgammon; scope=destination_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `destination_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `candidate_destinations`, `dice_rolls`, `board_state`, `destination_status`, `backgammon`, `destination_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `legal_move_count`, `hit_move_count`, `blocked_destination_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `state_update`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation is projected from numbered destination point bboxes, not individual checker bboxes.
