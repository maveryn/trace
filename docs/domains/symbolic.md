# Symbolic Domain Contract

Use this document for symbolic-domain rules. Exact active scenes and tasks live
in `docs/ACTIVE_TASK_INVENTORY.md` and `docs/tasks/symbolic/`.

## Scope
`symbolic` is a bounded domain for synthetic notation, instrument-like, and
formal readout scenes that do not yet justify a dedicated top-level domain. It
currently includes families such as abacus, clocks, automata, notation,
logic/structured symbols, dice/spinner probability, and similar controlled
symbolic renderers.

Do not add to `symbolic` when an existing renderer domain fits cleanly. Promote
a family out of `symbolic` if it becomes a coherent domain-scale surface.

## Scene Boundary
A symbolic scene is a stable notation or readout grammar: clock face, abacus,
music staff, Braille cell, chemistry/logic notation, automaton panel, dice
tray, spinner, or similar formal device. Style, font, palette, orientation, and
label choices may vary inside a scene when the same parser/verifier contract
holds.

Create a new scene when the symbolic grammar, notation system, or readout
interface changes materially.

## Task And Query Boundary
Split tasks when the objective changes between direct readout, transformed
readout, option matching, probability computation, state transition, symbol
count, or relation lookup. Valid `query_id` axes include mirrored directions,
offset unit, event type, target symbol, or bounded parameter choice inside one
stable program.

## Annotation Policy
Annotation should mark the decisive symbolic witnesses: readout hands/columns,
state cells, notation symbols, event sectors, dice faces, option cards, or rule
table entries. Use map annotation when roles such as initial state, target
state, rule row, source symbol, or selected option must be bound.

## Prompt And Rendering
Notation must prioritize legibility over decorative variation. Prompts should
name the relevant formal device, event, symbol, state, or readout scope without
assuming unstated conventions beyond the scene contract.

## Shared Code
Reusable symbolic renderers, notation parsers, probability calculators, state
transition helpers, and annotation projection utilities belong under
`src/trace_tasks/tasks/symbolic/shared/`. Keep family-specific helpers in scene-local
shared modules when they are not reused.
