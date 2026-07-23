# `task_games__backgammon__point_state_count`

## Contract
1. Domain: `games`
2. Scene id: `backgammon`
3. Public task id: `task_games__backgammon__point_state_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_count`
6. Annotation schema: `bbox_set`
7. Program schema: `count(filter(numbered_backgammon_points(board_state), checker_color, stack_state)); scene=backgammon; scope=point_state_count`

## Program Contract

Program: `count(filter(numbered_backgammon_points(board_state), checker_color, stack_state)); scene=backgammon; scope=point_state_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `point_state_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `numbered_backgammon_points`, `board_state`, `checker_color`, `stack_state`, `backgammon`, `point_state_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Generation Notes
1. The target checker color is sampled with `target_checker_color` (`black` or `white`).
2. The target stack state is sampled with `target_stack_state` (`single` or `two_or_more`).
3. Annotation is projected from numbered point bboxes, not individual checker bboxes.
