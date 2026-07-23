# `task_games__sokoban__box_goal_status_count`

## Contract
1. Domain: `games`
2. Scene id: `sokoban`
3. Public task id: `task_games__sokoban__box_goal_status_count`
4. Supported `query_id` values: `box_on_goal_count`, `box_off_goal_count`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`
7. Program schema: `count_boxes_by_matching_goal_status(status=on_goal|off_goal); scene=sokoban; scope=box_goal_status_count`
8. Scalar annotation checked: `true`

## Program Contract

Program: `count_boxes_by_matching_goal_status(status=on_goal|off_goal); scene=sokoban; scope=box_goal_status_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `box_goal_status_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `status`, `on_goal`, `off_goal`, `sokoban`, `box_goal_status_count` plus the active `query_id` branch.
Operation: evaluate `count_boxes_by_matching_goal_status` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `box_on_goal_count`, `box_off_goal_count`.

## Reasoning Operations

Families: `filtering`, `counting`, `matching`

## Generation Notes
1. The board shows paired colored boxes and matching colored goal dots.
2. A box on its matching goal has the goal dot drawn on top of the box so the covered goal remains visible.
3. The task asks either for boxes on their matching colored goal dots or boxes not on their matching colored goal dots.
4. The answer is an integer from `1` to `5`.
5. Each instance includes at least one opposite-status box as a distractor.
6. Annotation is the bbox set of the counted box cells; cardinality equals the answer.
7. Prompt wording comes from `src/trace_tasks/resources/prompts/games/sokoban/games_sokoban_v1.json`.
