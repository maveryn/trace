# Pages Domain Contract

Use this document for pages-domain rules. Exact active scenes and tasks live in
`docs/ACTIVE_TASK_INVENTORY.md` and `docs/tasks/pages/`.

## Scope
Pages covers structured page reasoning over forms, documents, process diagrams,
timelines, schedules, calendars, schema diagrams, GUI/web screens, static maps,
and infographic-style layouts. Tasks should be OCR-light: they may use typed
labels and short fields, but should not depend on long-form prose extraction.

Use `pages` when page layout, section structure, controls, fields, routes,
steps, or document regions are the semantic source of truth. Use `charts` when
the semantic source is a data display, and `graphs` when topology is the source.

## Scene Boundary
A scene is a reusable page scaffold: calendar grid, form, schedule, GUI, map,
process flow, hierarchy, concept map, schema, infographic, or route/layout
surface. Variant styling, fonts, section titles, themes, and short text pools
may vary inside a scene.

Create a new scene when the page grammar changes enough that fields, controls,
nodes, sections, or routes no longer share a common verifier structure.

Pages scenes that render structured information must support the shared
`information_scene` treatment baseline from `src/trace_tasks/resources/configs/domains/pages/base.yaml`.
That baseline mirrors the Charts 25-treatment pool: 20 light treatments and 5
dark treatments. Scene-specific renderers may map those shared roles into their
own calendar, table, form, timeline, hierarchy, or infographic grammar, but
they should not invent parallel theme axes for the same non-semantic style
role. The Pages render-audit wrapper records the selected shared style and
applies fallback outer chrome for scenes that do not map the style directly.
GUI/control scenes that have completed first-class style mapping, including
`control_board`, `navigation_flow`, `web_action`, and `workspace`, must not
expose a separate `style_variant` sampling axis for
non-semantic palette/theme changes; they should derive their internal chrome
theme from the resolved `information_scene_style`.
Shared context text, side notes, and paragraph distractors should inherit the
selected `information_scene` role colors when style metadata is present, unless
explicit context colors are passed for a task-specific reason.
Pages uses the same context profile vocabulary as Charts:
`dense_clean_minimal` for scenes that should only receive clean/minimal
safe-margin text, and `report_paragraph` for scenes whose layout can support
paragraph-style side-note distractor boxes. The default profile should be
`dense_clean_minimal`; scenes opt into paragraph support through
`pages_context_profile: report_paragraph` in `rendering.shared`.
Timeline and hierarchy scenes use `report_paragraph` because their page panels
provide stable interior whitespace for paragraph side notes; those scenes must
mark broad page/panel boxes as background and still protect event/node witness
boxes from context overlap.

## Task And Query Boundary
Split tasks when the program changes between field lookup, section-local count,
route/path reasoning, control selection, process step reasoning, schema
relation, option matching, or calendar/schedule computation.

Valid `query_id` axes include mirrored directions, named field/section choices,
threshold direction, selected control type, or bounded target attributes inside
one stable program.

## Annotation Policy
Annotation should mark the decisive visible field, section, checkbox/control,
event block, route landmark, process step, schema row/edge, map region, or
option image. Use map annotation when multiple roles matter, such as source
field vs target field, action header vs target row, route start vs route end,
or input vs output step.

For zero-count tasks, an empty set is allowed when the countable witness set is
empty; do not widen annotation to a whole section unless the section itself is
the queried witness.

## Prompt And Text
Prompts should name the queried field, section, event, control, route, node, or
step explicitly. Avoid requiring hidden page-layout conventions.

Use controlled short text, shared label pools, and shared context text. Keep
distractor text non-answering and visibly separated from required fields,
labels, and annotation targets. Distractor text colors must remain readable on
the active page treatment, including dark treatments.

## Shared Code
Reusable page layout, text fitting, form/control rendering, route mapping,
schema drawing, and section indexing helpers belong under
`src/trace_tasks/tasks/pages/shared/`. Scene-local helpers should stay inside the scene
package when tied to one scaffold.
