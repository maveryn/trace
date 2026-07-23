# `task_games__cards__exact_triple_count`

## Contract
1. Domain: `games`
2. Scene id: `cards`
3. Public task id: `task_games__cards__exact_triple_count`
4. Supported `query_id` values: `single`
5. Answer schema: `integer_count`
6. Annotation schema: `bbox_set_map`
7. Program schema: `count(filter(ranks, count(cards_of_rank(rank)) = 3)); scene=cards; scope=exact_triple_count`

## Program Contract

Program: `count(filter(ranks, count(cards_of_rank(rank)) = 3)); scene=cards; scope=exact_triple_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `exact_triple_count` objective scope.
Operands: visible scene state and prompt-bound operands named by `filter`, `ranks`, `cards_of_rank`, `rank`, `cards`, `exact_triple_count`.
Operation: evaluate `count` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set_map` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`

## Generation Notes
2. Prompt wording comes from `src/trace_tasks/resources/prompts/games/cards/games_cards_v1.json`.
3. Annotation is projected from the same generated game state used for answer verification.
