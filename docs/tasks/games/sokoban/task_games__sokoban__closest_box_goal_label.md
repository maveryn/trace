# `task_games__sokoban__closest_box_goal_label`

## Contract
1. Domain: `games`
2. Scene id: `sokoban`
3. Public task id: `task_games__sokoban__closest_box_goal_label`
4. Supported `query_id` values: `single`
5. Answer schema: `option_letter`
6. Annotation schema: `bbox`
7. Program schema: `select_closest_box_by_matching_goal(boxes, matching_goals, distance=manhattan); scene=sokoban; scope=closest_box_goal_label`
8. Scalar annotation checked: `true`

## Program Contract

Program: `select_closest_box_by_matching_goal(boxes, matching_goals, distance=manhattan); scene=sokoban; scope=closest_box_goal_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `closest_box_goal_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `boxes`, `matching_goals`, `distance`, `manhattan`, `sokoban`, `closest_box_goal_label`.
Operation: evaluate `select_closest_box_by_matching_goal` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`, `matching`

## Generation Notes
1. The board shows lettered colored boxes and matching colored goal dots.
2. No box starts on its matching goal.
3. The task asks which labeled box is closest to its matching colored goal dot.
4. Distance is Manhattan grid distance: row steps plus column steps, ignoring walls, boxes, the player, and other board objects.
5. The closest box is unique by construction.
6. Annotation is the scalar bbox of the selected box cell.
7. Prompt wording comes from `src/trace_tasks/resources/prompts/games/sokoban/games_sokoban_v1.json`.
