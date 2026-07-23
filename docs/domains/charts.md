# Charts Domain Contract

Use this document for chart-domain rules that are not already repo-wide
contracts. Exact active scenes and tasks live in `docs/ACTIVE_TASK_INVENTORY.md`
and `docs/tasks/charts/`.

## Scope
Charts covers rendered data displays where the answer is grounded in displayed
data values, marks, scales, legends, regions, panels, or tables. This includes
standard plots, dashboards, maps used as data displays, table-like data grids,
scientific plots, and composition charts.

Use `charts` when the data model is the semantic source of truth, even if the
image looks like a report or table. Use `pages` when page layout, form fields,
process steps, document sections, or controls are the semantic source of truth.

## Scene Boundary
Create or keep a scene when the visible data-display grammar is stable: one
chart family, one panel family, one map family, one table/grid family, or a
deliberate composite dashboard grammar.

Scene variants may cover non-semantic rendering or representation axes inside
the same scene, such as vertical vs horizontal bars, dot vs line marks, map
source type, scientific paper styling, palette, font, panel treatment, or
context-text framing. Record those axes in render/query metadata; do not split
public tasks only for style.

Create a new scene when the solver must parse a different visual grammar, such
as moving from a Cartesian series to a radial progress chart, a Sankey diagram,
a real/synthetic region map, a 3D surface, or a table grid.

## Task And Query Boundary
Apply `docs/contracts/TASK_UNIT_POLICY.md` directly. In charts, common valid
`query_id` axes include:

- mirrored rank/extremum direction, such as largest vs smallest;
- threshold direction, such as above vs below, over the same support;
- source/target role mirrors when the same visual grammar and program skeleton
remain stable;
- chart-family scene variants that do not change the objective contract.

Split into a new task when a branch changes one-bound threshold counting into
two-bound interval counting, count into value aggregation, label selection into
numeric value computation, single-chart reasoning into cross-panel reasoning,
or mark-local annotation into option-panel annotation.

## Annotation Policy
Annotation should mark the chart marks, cells, regions, intervals, panels, or
flows that visually witness the answer. Do not annotate answer text or legends
unless the legend/readout text itself is the visual object being queried.

Use map annotation when roles matter, such as source vs target marks,
reference vs candidate marks, interval endpoints, source and destination nodes,
or separate operands in an arithmetic comparison. Use unordered sets for
homogeneous counted marks/regions/cells.

Point annotation is preferred for small precise marks such as scatter points,
candles, curve points, or bar top-centers. Box annotation is preferred for
regions, cells, panels, bars when the full visible object matters, and option
images.

## Prompt Policy
Prompt wording should name visible labels in quotes and avoid assuming chart
jargon when ordinary wording is clearer. Examples should match the active answer
support: multi-character labels for chart labels, one-letter labels only for
option/panel-letter tasks.

Query wording should specify the visible scope, such as chart, panel, axis,
series, legend category, region set, or interval. If the task relies on an axis
label as the answer, make that answer role explicit.

## Rendering And Labels
Chart scenes should use shared font, label, palette, context-text, and
legibility helpers unless a scene has a documented reason not to. Labels and
legends should come from large reusable pools with scene-appropriate filtering,
not tiny task-local lists.

Keep non-answer context text as non-semantic. It may appear as headers,
captions, notes, source lines, callouts, sidebars, or report/dashboard framing,
but it must not cover marks, legends, labels, annotation targets, or answer
readouts.

Chart context sampling is profile-based. Dense scenes should use
`dense_clean_minimal` (`clean: 0.7`, `minimal: 0.3`) and must not emit
`paragraph_box`. Report-capable scenes should use
`report_paragraph` (`clean: 0.3`, `minimal: 0.4`, `paragraph_box: 0.3`) and
must render a real paragraph/context box whenever `paragraph_box` is selected.
In clean mode, chart scenes should not draw decorative titles, headers,
captions, notes, or other non-answer context text. Required chart grammar such
as axes, legends, panel labels, option labels, and task-relevant readouts
should remain visible.

Geographic map variants should sample only regions whose largest projected
connected component is large enough to inspect. For `region_map`, selected
geographic regions use a `400 px^2` minimum largest-component area and render
only that largest component in the selected color; smaller disconnected
components remain neutral. Annotation points for geographic regions should use
the center of the rendered selected component.

Semantic style, color, marker shape, line pattern, or size may be queried only
when the verifier records the same predicate. Otherwise those axes must remain
non-semantic and independent of answer value, query id, construction order, and
correct option.

## Shared Code
Domain-shared chart helpers belong under `src/trace_tasks/tasks/charts/shared/` only when
they are reused across multiple scenes. Related scene families may share
domain-level subpackages for Cartesian charts, composition charts, map charts,
Sankey-style flows, panels/grids, or distribution charts when reuse is real.

Scene-local shared modules belong under
`src/trace_tasks/tasks/charts/<scene_id>/shared/` and should contain reusable sampling,
layout, rendering, projection, scale, and annotation primitives. Public task
files own objective/query logic, answer binding, annotation binding, prompt
slots, and final output construction.
