# task_games__marble_chain__shot_effect_value

Public taxonomy: `games -> marble_chain -> task_games__marble_chain__shot_effect_value`.

## Contract
1. Domain: `games`
2. Scene id: `marble_chain`
3. Public task id: `task_games__marble_chain__shot_effect_value`
4. Supported `query_id` values: `single`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`
7. Program schema: `count(existing_chain_marbles_removed_by(marked_shot)); scene=marble_chain; scope=shot_effect_value`

## Program Contract

Program: `count(existing_chain_marbles_removed_by(marked_shot)); scene=marble_chain; scope=shot_effect_value`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `shot_effect_value` objective scope.
Operands: visible scene state and prompt-bound operands named by `existing_chain_marbles_removed_by`, `marked_shot`, `marble_chain`, `shot_effect_value`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `counting`, `topology`, `state_update`

## Generator

- Implementation: `src/trace_tasks/tasks/games/marble_chain/shot_effect_value.py`
- Config: `src/trace_tasks/resources/configs/domains/games/marble_chain.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/games/marble_chain/games_marble_chain_v1.json`
