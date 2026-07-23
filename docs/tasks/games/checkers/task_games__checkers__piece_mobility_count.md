# `task_games__checkers__piece_mobility_count`

## Contract
1. Domain: `games`
2. Scene package: `checkers`
3. Scene id: `checkers`
4. Public task id: `task_games__checkers__piece_mobility_count`
5. Supported `query_id` values: `piece_with_legal_move_count`, `piece_with_capture_move_count`
6. Answer schema: `integer_count`
7. Annotation schema: `bbox_set`
8. Program schema: `count(unique(origins(filter(legal_moves(current_player), move_filter)))); scene=checkers; scope=piece_mobility_count`

## Program Contract

Program: `count(unique(origins(filter(legal_moves(current_player), move_filter)))); scene=checkers; scope=piece_mobility_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `piece_mobility_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `unique`, `origins`, `filter`, `legal_moves`, `current_player`, `move_filter`, `checkers`, `piece_mobility_count` plus the active `query_id` branch.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `piece_with_legal_move_count`, `piece_with_capture_move_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `state_update`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation marks the current-player source-piece boxes that have at least one qualifying move.
