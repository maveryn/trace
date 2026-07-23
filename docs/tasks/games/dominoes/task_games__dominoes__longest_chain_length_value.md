# `task_games__dominoes__longest_chain_length_value`

## Contract
1. Domain: `games`
2. Scene: `dominoes`
3. Scene id: `dominoes`
4. Public task id: `task_games__dominoes__longest_chain_length_value`
5. Supported `query_id` values: `single`
6. Answer schema: `integer_value`
7. Annotation schema: `bbox_set`
8. Program schema: `max_chain_length(start=open_right_end(reference_tile), tiles=loose_dominoes); scene=dominoes; scope=longest_chain_length_value`

## Program Contract

Program: `max_chain_length(start=open_right_end(reference_tile), tiles=loose_dominoes); scene=dominoes; scope=longest_chain_length_value`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `longest_chain_length_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `start`, `open_right_end`, `reference_tile`, `tiles`, `loose_dominoes`, `dominoes`, `longest_chain_length_value`.
Operation: evaluate `max_chain_length` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `topology`

## Generation Notes
1. Renders the chain/tableau layout with the final chain tile marked `REF`.
2. The answer is the number of loose dominoes in the unique longest one-sided extension from `REF`, not counting `REF`.
3. Answer support is fixed to `1..5`.
4. Annotation is projected from the same generated game state used for answer verification.
