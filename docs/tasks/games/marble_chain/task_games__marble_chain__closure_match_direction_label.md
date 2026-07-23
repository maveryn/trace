# task_games__marble_chain__closure_match_direction_label

Public taxonomy: `games -> marble_chain -> task_games__marble_chain__closure_match_direction_label`.

## Contract
1. Domain: `games`
2. Scene id: `marble_chain`
3. Public task id: `task_games__marble_chain__closure_match_direction_label`
4. Supported `query_id` values: `single`
5. Answer schema: `option_letter`
6. Annotation schema: `point`
7. Program schema: `select(labelled_shot_options, condition=immediate_pop_count>0 && closure_boundary_colors_match); scene=marble_chain; scope=closure_match_direction_label`

## Program Contract

Program: `select(labelled_shot_options, condition=immediate_pop_count>0 && closure_boundary_colors_match); scene=marble_chain; scope=closure_match_direction_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `closure_match_direction_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `labelled_shot_options`, `immediate_pop_count`, `closure_boundary_colors_match`, `marble_chain`, `closure_match_direction_label`.
Operation: evaluate `select` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `comparison`, `topology`, `state_update`, `matching`

## Generator

- Implementation: `src/trace_tasks/tasks/games/marble_chain/closure_match_direction_label.py`
- Config: `src/trace_tasks/resources/configs/domains/games/marble_chain.yaml`
- Prompt bundle: `src/trace_tasks/resources/prompts/games/marble_chain/games_marble_chain_v1.json`
