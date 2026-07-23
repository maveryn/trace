# Physics Domain Contract

Use this document for physics-domain rules. Exact active scenes and tasks live
in `docs/ACTIVE_TASK_INVENTORY.md` and `docs/tasks/physics/`.

## Scope
Physics covers diagram-grounded mechanics, circuits, optics, and related
formula tasks where the operative quantities, directions, components, and
geometry are visible or explicitly implied by the diagram.

Use `physics` when physical relationships are the semantic source of truth. Use
`geometry` for purely geometric constructions and `pages` for document-like
explanatory diagrams.

## Scene Boundary
A physics scene is the stable physical system grammar: free-body diagram,
incline, pulley, circuit, ray/optics setup, fluid/pressure layout, motion path,
or measurement panel. Visual style, font, component palette, orientation, and
non-semantic distractor labels may vary inside a scene.

Create a new scene when the physical system, component grammar, or diagram
scaffold changes enough that a different formula/program or annotation contract
is needed.

## Task And Query Boundary
Split tasks by formula/program schema: force balance, torque, work/energy,
circuit equivalent value, optical path, motion time/distance, pressure, or
component selection should be distinct when operand roles or final operators
differ.

Valid `query_id` axes include mirrored directions, input/output unknown choice
inside the same formula, component target choice, or threshold/rank direction
when the program and annotation roles stay stable.

## Annotation Policy
Annotation should mark visible physical witnesses: force arrows, masses,
components, nodes, rays, target points, distances, angles, surfaces, or
measurement labels. Use map annotation for role-bound operands such as input
force and output force, resistor A and resistor B, source ray and reflected ray,
or object and support.

Do not annotate decorative apparatus parts unless they are operands or queried
witnesses.

## Prompt And Rendering
All required values must be visible or diagram-implied by explicit labels.
Avoid hidden constants or physical assumptions unless the prompt and task
contract make them part of the visible setup.

Early mechanics tasks should keep vectors axis-aligned unless vector
decomposition is the objective. Labels and arrows must be legible and
collision-aware.

Physics technical diagrams share the same technical style profiles as geometry.
Use `analytical_diagram` by default for apparatus, circuit, mechanics, optics,
and formula diagrams, with the same 20 light and 5 dark analytical themes used
by analytical geometry scenes. Use `graph_paper` only when the visible grid or
axes are part of the measurement or coordinate contract, such as motion graphs
and PV diagrams. Local panel guides, axes, tick marks, or helper grids inside an
apparatus scene must not force the global graph-paper theme profile.
Semantic colors for charges, fields, wires, rays, and components must remain
protected against sampled non-semantic theme colors.

## Shared Code
Reusable physical formulas, component sampling, diagram projection, rendering,
and annotation helpers belong under `src/trace_tasks/tasks/physics/shared/`. Scene-local
shared modules should own helpers tied to one physical system grammar.
