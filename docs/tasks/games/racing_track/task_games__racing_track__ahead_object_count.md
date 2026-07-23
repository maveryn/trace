# task_games__racing_track__ahead_object_count

## Taxonomy
- Domain: games
- Scene: racing_track
- Task: ahead_object_count

## Objective
Count other cars ahead of the marked car before the finish line, following the track direction.

## Program Contract

Program: `count(filter(racing_cars, progress_after(marked_car) and before_finish)); scene=racing_track; scope=ahead_object_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `ahead_object_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `racing_cars`, `progress_after`, `marked_car`, `before_finish`, `racing_track`, `ahead_object_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`

## Query IDs
- `single`

## Answer
Integer count.

## Annotation
Annotation schema: `bbox_set`.

`bbox_set` containing every counted car. Empty annotation is valid when the answer is `0`.

## Notes
The marked car is not counted.
