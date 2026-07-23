# `task_games__tetris__active_piece_shape_label`

## Contract
1. Domain: `games`
2. Scene: `tetris`
3. Scene id: `tetris`
4. Public task id: `task_games__tetris__active_piece_shape_label`
5. Supported `query_id` values: `single`
6. Answer schema: `string_label`
7. Annotation schema: `bbox`

## Program Contract

Program: `label(option where option.shape_name = shape_label(active_falling_tetromino)); scene=tetris; scope=active_piece_shape_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `active_piece_shape_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `where`, `shape_name`, `shape_label`, `active_falling_tetromino`, `tetris`, `active_piece_shape_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `matching`

## Generation Notes
1. The renderer shows a Tetris board with one active falling piece and four text options naming tetromino shapes.
2. The answer is the selected option letter whose shape name matches the falling piece.
3. Annotation is the scalar bbox enclosing the falling-piece cells on the board.
4. Scalar annotation checked: true.
