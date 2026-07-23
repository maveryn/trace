# task_games__mancala_pit_board__max_post_sow_option_label

Public taxonomy: `games -> mancala_pit_board -> task_games__mancala_pit_board__max_post_sow_option_label`.

## Contract

1. Domain: `games`
2. Scene package: `src/trace_tasks/tasks/games/mancala_pit_board/`
3. Scene id: `mancala_pit_board`
4. Public task id: `task_games__mancala_pit_board__max_post_sow_option_label`
5. Supported `query_id` values: `single`
6. Answer schema: `option_letter`
7. Annotation schema: `bbox`
8. Program schema: `select(option_label where final_seeds(option_pit) = max(final_seeds(marked_option_pits)) after sow_all_seeds_from(source_pit)); scene=mancala_pit_board; scope=max_post_sow_option_label`

## Program Contract

Program: `select(option_label where final_seeds(option_pit) = max(final_seeds(marked_option_pits)) after sow_all_seeds_from(source_pit)); scene=mancala_pit_board; scope=max_post_sow_option_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `max_post_sow_option_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `option_label`, `where`, `final_seeds`, `option_pit`, `max`, `marked_option_pits`, `after`, `sow_all_seeds_from`, `source_pit`, `mancala_pit_board`, `max_post_sow_option_label`.
Operation: evaluate `select` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `state_update`

## Generator

- Implementation: `src/trace_tasks/tasks/games/mancala_pit_board/max_post_sow_option_label.py`
- Config: `src/trace_tasks/resources/configs/domains/games/mancala_pit_board.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/games/mancala_pit_board/games_mancala_pit_board_v1.json`

## Generation Notes

- The board renders four marked option pits labeled `A` through `D`.
- The selected answer option is constructed to have a strictly larger post-sow seed count than the other three marked option pits.
- Annotation is the selected winning option pit bbox, not the option-letter badge alone.
