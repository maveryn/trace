# Shared Font Assets

Trace keeps repo-wide reusable font assets under `src/trace_tasks/resources/assets/fonts/`. Use this
layer when a renderer draws visible text and has access to a deterministic
instance seed.

## Source Policy

The current Trace font subset is vendored from the Google Fonts GitHub
repository. Google Fonts stores font families under top-level license
directories, and each family directory carries the font files, metadata, and
the applicable license file. The current Trace subset uses 500 Latin-capable
OFL/Apache families with family-local license files and metadata recorded in
`src/trace_tasks/resources/assets/fonts/sources.json`.

Runtime task generation must use local files only. Do not download fonts while
generating task instances.

## Runtime Rules

1. Sample fonts deterministically from seed and namespace through
   `src/trace_tasks/tasks/shared/font_assets.py`.
2. Render fonts through `src/trace_tasks/tasks/shared/text_rendering.py`.
3. Registered task generation installs one deterministic per-instance
   `implicit_readout_font_family` before calling the task renderer. This is a
   compatibility path for existing shared renderers that call `load_font(...)`
   without an explicit `font_family`; it still samples from the `readout` pool
   through the shared dispatcher and records the family under
   `render_spec.font_assets`.
4. Chart tasks must expose `render_spec.font_assets.chart_font_family` for
   consistent trace metadata. Renderers that sample an explicit chart family should
   write that value directly; otherwise the registry records the same
   deterministic `implicit_readout_font_family` as the chart family alias.
5. New or touched renderer code should pass an explicit sampled family into
   `load_font(..., font_family=...)`, or enter a documented
   `temporary_default_font_family(...)` block for one coherent scene/panel.
6. Every task-facing font sample must declare a role:
   - `readout`: answer-bearing or read-required text such as measurements,
     chart ticks, table cells, graph labels, board coordinates, option labels,
     card ranks, and compact visible task text. This role samples from
     `src/trace_tasks/resources/assets/fonts/readout_pool_v0.json`.
   - `context`: non-answer chrome, titles, side notes, headers, body text, and
     realistic distractor text. This role may sample from the full vendored
     manifest.
   - `decorative`: non-semantic visual dressing only. This role may sample from
     the full vendored manifest.
7. Keep existing text roles internally consistent:
   - all chart axis/category/value labels in one chart or chart panel should use
     one sampled family unless the scene explicitly has separate text regions;
   - all option labels in one option set should use one sampled family;
   - each page/document section may use one sampled family for that section;
   - context boxes may use a different sampled family per box, but heading/body
     text inside one box should normally share the same family.
8. Math-symbol readouts may use the shared symbol-safe fallback for the affected
   text token when the sampled readout family does not reliably cover the
   required glyphs, such as `∠`, `θ`, `β`, `π`, `√`, or `−`. The fallback must
   be routed through `src/trace_tasks/tasks/shared/text_rendering.py` so bbox calculation
   and drawing use the same font.
9. Mixing multiple families in one image is allowed when it follows meaningful
   visual regions: title/chrome, body text, option set, chart labels, sidebar
   note, callout box, etc. Do not sample a different family per glyph, per word,
   or per answer candidate unless that variation is the task itself.
10. Record sampled font metadata in `render_spec` or scene entity metadata:
   `font_family`, `font_role`, `font_pool_id`, `font_pool_size`, and
   `font_asset_version`. Non-answer context text should record font family and
   role in each `context_text_layer.elements[]` record.
11. Font choice must be independent of answer value, correct option, query id,
   and difficulty bucket.
12. Do not use ad hoc include/exclude tag filters for semantic readout text; the
   readout pool already owns that policy. Use explicit role selection first, and
   only add a scene-local exclusion for a documented non-readout readability or
   semantics reason.
13. Do not use unlicensed system fonts as a source of variation.
   System fonts remain fallback only.

## Available Families

The current vendored target set is 500 readable Latin-capable families covering sans,
serif, monospace, condensed, rounded, accessible, technical, editorial,
slab-like, display, pixel, stencil, handwriting, and script styles. The expanded
set keeps the original neutral families and adds more visually distinct fonts
such as `bangers`, `black_ops_one`, `caveat`, `cookie`, `dancing_script`,
`lobster`, `pacifico`, `patrick_hand`, `press_start_2p`, and `righteous`.

The builder starts from a stable hand-curated core, then fills the remaining
families from the live Google Fonts catalog using conservative filters:
Latin coverage, open source, OFL/Apache source directories with local license
files, no color/symbol/emoji/icon/math families, no Noto script-expansion
families beyond the core Noto text families, and bounded catalog size. It also
uses category quotas so the pack has broad visual variety without sampling the
full Google Fonts catalog blindly.

Use `src/trace_tasks/resources/assets/fonts/sources.json` as the authoritative family list. It records
the family key, display name, local regular and bold font paths, license path,
source URL, and role-filter tags for every family.

Use `src/trace_tasks/resources/assets/fonts/readout_pool_v0.json` as the authoritative `readout` role
shortlist. It contains 100 narrow, legible, policy-safe families selected for
compact read-required text.

See `src/trace_tasks/resources/assets/fonts/sources.json` for source URL, license path, selected regular
and bold file paths, and role-filter tags.

## API

Use deterministic family sampling:

```python
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font

font_family = sample_font_family(
    role="readout",
    instance_seed=instance_seed,
    namespace=f"{TASK_ID}.chart_labels_font",
    params=params,
)
label_font = load_font(14, bold=True, font_family=font_family)
```

For shared render paths that need one consistent family across many existing
`load_font(...)` and `fit_font_to_box(...)` calls, use the temporary default
font context and record the sampled family in render metadata:

```python
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family

with temporary_default_font_family(font_family):
    rendered = render_scene(...)
```

Do not set this context from hidden randomness. The caller must sample the
family deterministically through `sample_font_family(...)` and trace the family
key before or after rendering.

If a scene needs a fixed family for debugging or reproducibility checks, pass
`font_family=<family_key>` in params for the relevant sampling call, or define a
role-specific explicit key in the scene adapter.

When updating the vendored font set, inspect
`src/trace_tasks/resources/assets/fonts/sources.json`, preserve every
family-local license file, compile the shared font modules, and generate
representative samples for renderers whose font behavior changed.

## Audit

Use the static migration checks before accepting a domain-level font change:

```bash
python scripts/audit_text_legibility.py --root . --scan-root src/trace_tasks/tasks \
  --strict-renderer-migration --strict-role-metadata --strict-font-routing
```

`--strict-font-routing` rejects missing sampler roles, suspicious
context/readout role mismatches, and direct `ImageFont.truetype/load_default`
usage outside the shared text-rendering fallback. `--strict-role-metadata`
requires every traced text draw to explicitly declare `role=` and `required=`.
