# `task_games__rule_override_board__line_result_count`

## Program Contract

Program: `count(filter(mini_boards, anti_line_result(target_player)=target_result)); scene=rule_override_board; scope=line_result_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `line_result_count` objective scope.
Operands: visible scene state and prompt-bound target operands named by the task contract.
Operation: evaluate `count(filter(mini_boards, anti_line_result(target_player)=target_result))` over the visible mini-board set; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema for every counted mini-board panel.
Query ids: `line_override_win_count`, `line_override_loss_count`.

## Reasoning Operations

Families: `filtering`, `counting`

## Generation Notes

- The prompt states the anti-line rule: a full row, column, or diagonal is a loss for the target player.
- `line_override_win_count` counts mini-boards where the target player wins; `line_override_loss_count` counts losses.
- Annotation bboxes are the counted mini-board panels.
- The answer and annotation are bound from the same generated board state.
