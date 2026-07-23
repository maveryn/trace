# Geometry Domain Contract

Use this document for geometry-domain rules. Exact active scenes and tasks live
in `docs/ACTIVE_TASK_INVENTORY.md` and `docs/tasks/geometry/`.

## Scope
Geometry covers synthetic geometric diagrams, coordinate diagrams, measurement
figures, construction-like scenes, and formula-grounded visual reasoning. The
image should expose the relevant points, segments, shapes, labels, dimensions,
grid cues, or option panels needed to solve the task.

Use `geometry` when the geometric construction or formula schema is the source
of truth. Use `charts` for plotted data displays and `pages` for document-like
layouts.

## Scene Boundary
A geometry scene is a stable construction grammar: the object family, diagram
layout, label conventions, and measurement scaffold. Scene variants may vary
shape subtype, orientation, graph-paper style, visual treatment, or label names
when the same construction/program remains valid.

Create a new scene when the geometric theorem family, visible construction, or
answer interface changes enough that a different solver program or witness
contract is required.

## Task And Query Boundary
Geometry tasks are commonly separated by formula/program schema, not merely by
shape name. For example, area, perimeter, angle chase, similarity ratio,
Pythagorean length, volume, and option-matching objectives are different task
contracts when they use different operands or final operators.

Valid `query_id` axes include mirrored unknowns within the same formula,
different givens/unknowns in one algebraically equivalent construction, rank
direction, and shape subtype when the same program schema and annotation
contract remain stable.

For keyed point annotation in geometry, sample-bound point labels do not by
themselves change the annotation contract. A segment task may ask for `AY` in
one instance and `XB` in another, with annotation keys matching those visible
labels, as long as the role family is still "the two endpoints of the requested
segment" and the cardinality stays fixed.

Split when a branch changes the formula schema, the answer type, the witness
roles, the visual scaffold, or the final output binding.

## Annotation Policy
Use point annotation for points, vertices, segment endpoints, angle arms, and
coordinate witnesses. Use bbox annotation for whole shapes, panels, shaded
regions, option images, or diagrams whose decisive witness is an area/object.
Use map annotation when operand roles matter.

The visible `?` marker should be annotated only when the unknown location itself
is the requested visual witness. For formula tasks, annotation should usually
mark the operands/construction used to solve the unknown, not the printed `?`.

## Prompt And Numeric Policy
Decimal answers must specify one decimal place when the answer contract is a
one-decimal value. Do not display approximation constants such as `pi=3.14` in
the image unless the task explicitly asks for readout from that text.

Prompts should name the required measurement, shape, panel, or marked object
without relying on textbook convention ambiguity. If definitions overlap, such
as isosceles vs equilateral, state the intended exclusivity in task wording or
construction metadata.

## Rendering
Geometry renderers should keep labels collision-aware and separated from points,
segments, edges, and filled regions. Graph-paper scenes should align objects,
labels, and annotations through one projection path. Non-measurement scenes
should avoid unnecessary graph-paper backgrounds unless grid cues are part of
the task contract.

Geometry uses two shared technical-diagram style profiles. Analytical diagram
scenes use the `analytical_diagram` profile: plain paper, textbook, whiteboard,
exam, print, and dark board/slide themes that do not add coordinate grids.
Coordinate and graph-paper scenes use the `graph_paper` profile: square or
lab-grid themes only, with axes, grid spacing, and lattice readability preserved.
Do not use ruled, isometric, crosshair, or decorative grid treatments for
Cartesian graph-paper tasks.

## Shared Code
Cross-scene geometry helpers belong in `src/trace_tasks/tasks/geometry/shared/`.
Scene-local `shared/` modules should own construction primitives, projection,
layout, rendering, and annotation helpers for one scene. Public task files own
formula/query selection, answer binding, prompt slots, and final output.
