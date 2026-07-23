# `task_games__sokoban__push_stand_cell_label`

## Contract
1. Domain: `games`
2. Scene id: `sokoban`
3. Public task id: `task_games__sokoban__push_stand_cell_label`
4. Supported `query_id` values: `single`
5. Answer schema: `option_letter`
6. Annotation schema: `bbox`
7. Program schema: `select_player_stand_cell_for_straight_push(box, matching_goal, candidate_stand_cells); scene=sokoban; scope=push_stand_cell_label`
8. Scalar annotation checked: `true`

## Program Contract

Program: `select_player_stand_cell_for_straight_push(box, matching_goal, candidate_stand_cells); scene=sokoban; scope=push_stand_cell_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `push_stand_cell_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `box`, `matching_goal`, `candidate_stand_cells`, `sokoban`, `push_stand_cell_label`.
Operation: evaluate `select_player_stand_cell_for_straight_push` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `spatial_relations`, `state_update`

## Generation Notes
1. The board shows a player, colored boxes, matching colored goal dots, and four labeled candidate standing cells around one target box.
2. The prompt names the target box by a canonical safe color label from `trace_tasks.tasks.shared.named_colors`.
3. The target goal is in a straight horizontal or vertical line from the target box.
4. There are no walls, boxes, or other objects between the target box and its matching goal dot.
5. The correct stand cell is the labeled cell immediately behind the target box, opposite the push direction toward the goal.
6. Annotation is the scalar bbox of the selected stand cell.
7. Prompt wording comes from `src/trace_tasks/resources/prompts/games/sokoban/games_sokoban_v1.json`.
