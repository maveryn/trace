# `task_games__match3__gem_count`

## Contract
1. Domain: `games`
2. Scene id: `match3`
3. Public task id: `task_games__match3__gem_count`
4. Supported `query_id` values: `grid_color_gem_count`, `row_color_gem_count`, `column_color_gem_count`
5. Answer schema: `integer`
6. Annotation schema: `bbox_set`

## Program Contract

Program: `count.scoped_attribute(candidate_set=gems, scope=grid|row|column, attribute=color_name=target_color); scene=match3; scope=gem_count`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `grid|row|column` objective scope.
Operands: visible scene state and prompt-bound operands named by `candidate_set`, `gems`, `grid`, `row`, `column`, `attribute`, `color_name`, `target_color`, `match3`, `gem_count` plus the active `query_id` branch.
Operation: evaluate `count.scoped_attribute` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `bbox_set` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `grid_color_gem_count`, `row_color_gem_count`, `column_color_gem_count`.

## Reasoning Operations

Families: `filtering`, `counting`

## Generation Notes
1. Gem colors are sampled from the repo-wide canonical named-color palette.
2. Prompt-facing color labels include the canonical hex value, for example `red [#E63232]`.
3. Annotation is the bbox set of matching gems inside the requested grid, row, or column scope.
