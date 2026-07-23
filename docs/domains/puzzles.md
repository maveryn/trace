# Puzzles Domain Contract

Use this document for puzzles-domain rules. Exact active scenes and tasks live
in `docs/ACTIVE_TASK_INVENTORY.md` and `docs/tasks/puzzles/`.

## Scope
Puzzles covers hidden-rule, hidden-variable, and visual puzzle reasoning where
the image presents a constrained puzzle state, rule surface, option set, or
missing slot. It is not a catch-all for icon grids, game boards, charts, or
document forms.

Use `puzzles` when the task is to infer or apply a puzzle rule from the visible
state. Use `games` for recognizable game mechanics and `icons` for direct icon
attribute counting or matching.

## Scene Boundary
A puzzle scene is the stable visual grammar and rule family: cell board,
nonogram, maze, pipe flow, word search, automaton, Rubik-like net, voxel cube,
jigsaw/cutout puzzle, or option grid. Style, unit
size, palette, font, clue labels, and puzzle dimensions may vary when the
verifier contract remains stable.

Create a new scene when the puzzle scaffold or rule representation changes
materially.

## Task And Query Boundary
Split tasks when the objective changes between missing value, valid move,
option selection, path tracing, rule violation, future state, count, or
counterfactual outcome. Keep bounded mirror/direction/player/axis choices in
`query_id` only when the same puzzle program and annotation contract remain
stable.

## Annotation Policy
Annotation should stay local to the unknown slot, selected option image,
counted cells, path cells, rule witnesses, changed cells, or decisive clue
regions. Use sequence annotation for ordered paths and map annotation when
source/target or before/after roles matter.

For option-image puzzles, selected option bboxes are valid when the option is
the complete candidate state or patch being matched.

## Rendering
Repeated-unit puzzle scenes should vary unit size and style while computing
annotation from the final layout. Text, numbers, clue labels, markers, and path
highlights must remain readable under palette/style changes.

Do not add generic titles that only name the puzzle. Keep in-image text only
when it is part of the rule, clue, state, option label, or answer interface.

## Shared Code
Puzzle-wide helpers belong under `src/trace_tasks/tasks/puzzles/shared/`. Scene-local
shared modules should own rule construction, board/state representation,
rendering, layout, projection, and annotation for one puzzle grammar.
