# Graph Domain Contract

Use this document for graph-domain rules. Exact active scenes and tasks live in
`docs/ACTIVE_TASK_INVENTORY.md` and `docs/tasks/graph/`.

## Scope
Graph covers node-link, tree, route, automaton-like, and network diagrams where
topology or explicit edge labels are the semantic source of truth. Layout is
visual variation unless the task explicitly asks about screen position.

## Scene Boundary
A graph scene is the stable graph grammar: node/edge vocabulary, directionality,
edge labels or weights when present, tree/route/layout family, and any visible
reference panels. Create a new scene when the visual grammar changes from
ordinary node-link topology to a tree, route map, automaton table, layered
network, or another distinct scaffold.

## Task And Query Boundary
Split graph tasks when the graph algorithm or witness structure changes:
degree, adjacency, path/traversal, reachability, predecessor/successor,
connected component, cycle, cut/bridge, route distance, or option selection are
distinct objective contracts when their program schemas differ.

Valid `query_id` axes include mirrored direction such as predecessor/successor,
in/out degree, source/target role mirrors, or shortest/longest variants only
when the underlying algorithm and annotation contract remain stable under the
task policy. If the algorithm is genuinely different, split the task.

## Annotation Policy
Node witnesses usually use `point_set` or `bbox_set`. Edge witnesses use
`segment_set` or path-like point/box sequences. Ordered path, traversal, or
operation tasks must use sequence annotation. Unordered node/edge counts should
use unordered sets.

Use map annotation when distinct roles such as source, target, bridge,
reference node, or candidate node must be bound.

## Prompt And Rendering
Node labels are the canonical prompt-facing identities. Directed edges must use
clear arrowheads and prompt wording such as in-degree, out-degree, predecessor,
or successor when direction matters. Weighted tasks must render weights as
readable, semantically essential text.

Graph layout, node glyph style, labels, colors, and mild geometric transforms
may vary when they do not encode the answer. Questions should remain answerable
from topology and visible labels under layout changes.

## Shared Code
Use `src/trace_tasks/tasks/graph/shared/` for reusable graph sampling, algorithms,
layout, rendering, and annotation projection. Scene-local shared modules should
be used when a helper is tied to one graph grammar.
