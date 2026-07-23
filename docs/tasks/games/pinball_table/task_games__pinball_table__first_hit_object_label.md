# `task_games__pinball_table__first_hit_object_label`

## Contract
1. Domain: `games`
2. Scene package: `src/trace_tasks/tasks/games/pinball_table/`
3. Scene id: `pinball_table`
4. Public task id: `task_games__pinball_table__first_hit_object_label`
5. Supported `query_id` values: `single`
6. Answer schema: `string_label`
7. Annotation schema: `point`

## Program Contract

Program: `label(first_collision(straight_launch_path, labeled_pinball_objects)); scene=pinball_table; scope=first_hit_object_label`

Candidate set: the visible game board, pieces, tokens, cards, tiles, marked state, legal-move cues, result panels, and labeled options inside the `first_hit_object_label` objective scope.
Operands: visible scene state and prompt-bound operands named by `first_collision`, `straight_launch_path`, `labeled_pinball_objects`, `pinball_table`, `first_hit_object_label`.
Operation: evaluate `label` over the candidate set using the visible game state, rules, legal moves, comparisons, counts, simulations, or option-selection constraints encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `string` schema; generation binds a unique final answer.
Annotation witnesses: `annotation` uses the `point` schema; the prompt/annotation contract defines the minimal visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `ranking`, `spatial_relations`, `topology`

## Generation Notes
1. The scene renders a tilted pinball playfield with one ball, one straight launch cue, flippers, slingshots, rails, bumpers, lanes, and labeled targets.
2. Labeled bumpers, drop targets, rollover lanes, and standup targets are answer candidates; flippers, rails, posts, and slingshots are decorative playfield structure.
3. The target answer is unique by construction: extending the cue intersects the selected object before any other labeled object.
4. Annotation is one pixel point at the center of the first-hit object, projected from the same generated geometry used for verification.
