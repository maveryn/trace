# task_games__racing_track__finish_distance_extremum_label

## Taxonomy
- Domain: games
- Scene: racing_track
- Task: finish_distance_extremum_label

## Objective
Select the labeled car with an extremal remaining distance to the finish line, measured along the track direction.

## Program Contract

Program: `select_extremum(racing_cars, remaining_distance_to_finish, operator=closest|farthest); scene=racing_track; scope=finish_distance_extremum_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `finish_distance_extremum_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `racing_cars`, `remaining_distance_to_finish`, `operator`, `closest`, `farthest`, `racing_track`, `finish_distance_extremum_label`.
Operation: evaluate `select_extremum` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; the value is the unique selected car label.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`

## Query IDs
- `closest_to_finish_label`
- `farthest_from_finish_label`

## Answer
String car label.

## Annotation
Annotation schema: `point`.

`point` at the selected car center.

## Notes
The scene is a single-lane loop track with a direction arrow and checkered finish line. Remaining distance is based on circular progress along the track, not straight-line image distance.
