# Shared Font Assets

This directory stores a curated Trace-vendored subset of permissively licensed
fonts for deterministic visual variation. The current subset comes from the
Google Fonts GitHub repository and includes family-local license files recorded
in `sources.json`.

Task renderers should sample font families through
`trace_tasks.tasks.shared.font_assets` and render them through
`trace_tasks.tasks.shared.text_rendering`. Runtime generation must not download
font files.

`readout_pool_v0.json` shortlists 100 narrow, legible families for required
text such as measurements, chart labels, table cells, graph node labels, and
compact option text. Decorative titles, context text, and other non-semantic
elements may use the broader 500-family manifest when readability does not
affect solving.

Runtime renderers use the shared role-aware font dispatcher:

- `readout` for required labels and values;
- `context` for non-answer chrome and side notes;
- `decorative` for non-semantic visual dressing.

System fonts such as DejaVu Sans and Liberation Sans are fallback/reference
fonts only, not vendored Trace assets. Updates to the font set must preserve
each family's metadata and local license file and refresh `sources.json`.
