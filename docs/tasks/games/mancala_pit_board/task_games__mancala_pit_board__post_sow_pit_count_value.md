# task_games__mancala_pit_board__post_sow_pit_count_value

Public taxonomy: `games -> mancala_pit_board -> task_games__mancala_pit_board__post_sow_pit_count_value`.

## Contract

1. Domain: `games`
2. Scene package: `src/trace_tasks/tasks/games/mancala_pit_board/`
3. Scene id: `mancala_pit_board`
4. Public task id: `task_games__mancala_pit_board__post_sow_pit_count_value`
5. Supported `query_id` values: `single`
6. Answer schema: `integer`
7. Annotation schema: `bbox_map`
8. Program schema: `count(final_seeds(target_pit) after sow_all_seeds_from(source_pit)); scene=mancala_pit_board; scope=post_sow_pit_count_value`

## Program Contract

Program: `count(final_seeds(target_pit) after sow_all_seeds_from(source_pit)); scene=mancala_pit_board; scope=post_sow_pit_count_value`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `post_sow_pit_count_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `final_seeds`, `target_pit`, `after`, `sow_all_seeds_from`, `source_pit`, `mancala_pit_board`, `post_sow_pit_count_value`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `state_update`

## Generator

- Implementation: `src/trace_tasks/tasks/games/mancala_pit_board/post_sow_pit_count_value.py`
- Config: `src/trace_tasks/resources/configs/domains/games/mancala_pit_board.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/games/mancala_pit_board/games_mancala_pit_board_v1.json`
