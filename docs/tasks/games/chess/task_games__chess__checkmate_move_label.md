# `task_games__chess__checkmate_move_label`

## Contract
1. Domain: `games`
2. Scene id: `chess`
3. Public task id: `task_games__chess__checkmate_move_label`
4. Supported `query_id` values: `single`
5. Answer schema: `option_letter`
6. Annotation schema: `point_map`
7. Program schema: `select(option where move_checkmates(opponent_king)); scene=chess; scope=checkmate_move_label`

## Program Contract

Program: `select(option where move_checkmates(opponent_king)); scene=chess; scope=checkmate_move_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `checkmate_move_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `where`, `move_checkmates`, `opponent_king`, `chess`, `checkmate_move_label`.
Operation: evaluate `select` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `state_update`

## Generation Notes
1. The board shows standard chess coordinates on the margins and a visible panel of candidate moves.
2. The visible options are encoded as piece name plus source and destination square.
3. Exactly one displayed option is an immediate checkmate by construction; the underlying board may contain other mating moves that are not displayed.
4. Annotation is projected from the selected move's source square, destination square, and opposing king square using keys `from`, `to`, and `king`.
