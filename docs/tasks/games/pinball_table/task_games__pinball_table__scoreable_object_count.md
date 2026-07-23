# `task_games__pinball_table__scoreable_object_count`

## Contract
1. Domain: `games`
2. Scene package: `src/trace_tasks/tasks/games/pinball_table/`
3. Scene id: `pinball_table`
4. Public task id: `task_games__pinball_table__scoreable_object_count`
5. Supported `query_id` values: `single`
6. Answer schema: `integer`
7. Annotation schema: `bbox_set`

## Program Contract

Program: `count(object for object in pinball_objects if object.has_numeric_score_label); scene=pinball_table; scope=scoreable_object_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `scoreable_object_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `object`, `pinball_objects`, `if`, `has_numeric_score_label`, `pinball_table`, `scoreable_object_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`

## Generation Notes
1. The scene renders a tilted pinball playfield with one ball, decorative table elements, and 5 to 8 visible table objects.
2. Scoreable objects display numeric score labels. Non-scoreable distractors have no visible text.
3. The answer is the number of scoreable objects, balanced across 1 to 6 when the visible object count allows it.
4. Annotation is an unordered bbox set containing every scoreable object.
