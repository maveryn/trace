# `task_games__checkers__max_capture_chain_length`

## Contract
1. Domain: `games`
2. Scene package: `checkers`
3. Scene id: `checkers`
4. Public task id: `task_games__checkers__max_capture_chain_length`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_value`
7. Annotation schema: `bbox_set`
8. Program schema: `longest_path(capture_state_graph(marked_king, board_state), source=marked_king, target=terminal_no_capture_state); scene=checkers; scope=max_capture_chain_length`

## Program Contract

Program: `longest_path(capture_state_graph(marked_king, board_state), source=marked_king, target=terminal_no_capture_state); scene=checkers; scope=max_capture_chain_length`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `max_capture_chain_length` objective scope.
Operands: visible scene state and prompt-bound operands named by `capture_state_graph`, `marked_king`, `board_state`, `source`, `terminal_no_capture_state`, `checkers`, `max_capture_chain_length`.
Operation: evaluate `longest_path` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `ranking`, `topology`, `state_update`

## Generation Notes
2. Query ids are internal replay/sampling keys and do not define public task units.
3. Annotation marks the captured opponent-piece boxes along the longest chain.
