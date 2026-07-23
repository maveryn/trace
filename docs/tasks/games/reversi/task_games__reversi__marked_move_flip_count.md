# `task_games__reversi__marked_move_flip_count`

## Contract
1. Domain: `games`
2. Scene id: `reversi`
3. Public task id: `task_games__reversi__marked_move_flip_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_count`
6. Annotation schema: `point_set`
7. Program schema: `count(flipped_discs(apply(marked_legal_move))); scene=reversi; scope=marked_move_flip_count`

## Program Contract

Program: `count(flipped_discs(apply(marked_legal_move))); scene=reversi; scope=marked_move_flip_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `marked_move_flip_count` objective scope.
Operands: visible scene state and prompt-bound target operands named by the task contract.
Operation: evaluate `count(flipped_discs(apply(marked_legal_move)))` over the candidate set using the visible Reversi board state; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point_set` schema for the centers of all discs flipped by the marked move.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `state_update`

## Generation Notes

- The marked empty square is a legal Reversi move for the current player.
- The task asks how many opponent discs would flip if that marked move is played.
- Annotation points are the centers of all discs flipped by the marked move.
- The answer and annotation are bound from the same generated marked-move flip set.
