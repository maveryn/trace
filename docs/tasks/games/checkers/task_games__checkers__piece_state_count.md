# `task_games__checkers__piece_state_count`

## Contract
1. Domain: `games`
2. Scene package: `checkers`
3. Scene id: `checkers`
4. Public task id: `task_games__checkers__piece_state_count`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_count`
7. Annotation schema: `bbox_set`
8. Program schema: `count(filter(checkers_pieces(board_state), piece_color, board_edge_state)); scene=checkers; scope=piece_state_count`

## Program Contract

Program: `count(filter(checkers_pieces(board_state), piece_color, board_edge_state)); scene=checkers; scope=piece_state_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `piece_state_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `checkers_pieces`, `board_state`, `piece_color`, `board_edge_state`, `checkers`, `piece_state_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Generation Notes
1. The target piece color is sampled with `target_player` (`red` or `black`).
2. The target state is sampled with `piece_state_kind` (`all` or `edge`).
3. Annotation marks the pixel-space boxes of visible pieces that satisfy the color and board-edge condition.
