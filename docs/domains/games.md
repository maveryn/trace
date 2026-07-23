# Games Domain Contract

Use this document for games-domain rules. Exact active scenes and tasks live in
`docs/ACTIVE_TASK_INVENTORY.md` and `docs/tasks/games/`.

## Scope
Games covers recognizable game artifacts where the visible state, local rules,
and rendered pieces are enough to solve the query. Prefer fully observable,
low-convention tasks over hidden-information strategy.

Scenes may include boards, cards, dice, arcade screens, score strips, option
boards, counterfactual board-style grids, or rule panels when those elements
are part of the visible game state.

## Scene Boundary
A games scene is the stable game grammar: board topology, piece vocabulary,
movement/capture/scoring convention, and answer interface. Keep one scene for a
game family even when board size, palette, player color, piece skin, or camera
style varies.

Create a new scene when the visible rules or board grammar change materially,
for example a standard grid board vs a radial board, a card tableau vs a dice
tray, or a board-only view vs a source-plus-option-board view if that scaffold
changes the task contract.

## Task And Query Boundary
Split tasks when the game reasoning program changes: choosing a legal move,
computing the resulting board, counting captures, scoring a move, selecting an
option board, or reading a status are separate objective contracts when their
answer/annotation schemas or program skeletons differ.

Valid `query_id` axes include mirrored player color, row/column axis,
direction, legal-piece type, threshold direction, or other bounded parameters
inside one stable rule program. Do not use `query_id` to hide separate public
objectives.

## Annotation Policy
Prompt-facing annotation should stay on visible game witnesses: cells, pieces,
move paths, merge pairs, captured groups, option boards, score/status readouts,
or rule-table entries. Use map annotation when source and destination, before
and after, or reference and candidate roles must be bound correctly.

For games counting tasks, prefer `bbox` / `bbox_set` / bbox maps when the
counted witnesses are selectable area-like objects: grid cells, tiles, board
squares, cards, bricks, gems, bubbles, blocks, discs, tokens, pellets, balls, or
scoreable objects. Prefer `point` / `point_set` for localization-style tasks and
compact feature witnesses where the answer is a specific center/location rather
than a collection of counted objects, such as a selected option cell, first-hit
object, impact point, endpoint, graph-like node, empty liberty, or completion
point. Use `segment` / `segment_set` for row/column spans, merge links, paths,
and shot trajectories.

Selected option bboxes are valid only when the option itself is a visual
candidate board/panel or complete candidate artifact.

## Rendering And Text
Repeated-unit game scenes should vary board/unit size and style when feasible,
while computing annotation from the final layout. Board skins, tile palettes,
piece treatments, lane skins, card/table chrome, and HUD style are scene-local
visual axes.

Do not draw generic scene titles that merely name the game. Keep in-image text
when it is part of the game grammar or answer interface, such as card ranks,
coordinates, option labels, called-number headers, movement strips, or visible
rule/status panels.

Required labels, readouts, and markers should use shared legibility helpers or
the games-domain wrappers built on them.

## Shared Code
Scene-local shared modules should contain board/rule state, legal move or
scoring helpers, rendering, layout, style, text/marker helpers, and annotation
projection. Public task files own target construction, answer binding,
annotation binding, prompt slots, and task-specific trace fields.

Promote helpers to `src/trace_tasks/tasks/games/shared/` only after multiple games scenes
reuse them.
