# `task_games__nine_mens_morris__pieces_in_mill_count`

## Contract
1. Domain: `games`
2. Scene id: `nine_mens_morris`
3. Public task id: `task_games__nine_mens_morris__pieces_in_mill_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_count`
6. Annotation schema: `bbox_set`

## Program Contract

Program: `count(filter(pieces, participates_in_mill(piece)=true)); scene=nine_mens_morris; scope=pieces_in_mill_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `pieces_in_mill_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `pieces`, `participates_in_mill`, `piece`, `true`, `nine_mens_morris`, `pieces_in_mill_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Generation Notes
1. Query ids are internal replay/sampling keys and do not define public task units.
2. The public query id is `single`; the prompt query key remains `all_pieces_in_mill_count`.
3. Annotation is projected from the same generated game state used for answer verification.
