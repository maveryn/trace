# Pages Visual Asset Pool

This folder contains curated visual assets for pages-domain renderers.
They are non-answer decorative or layout-context assets by default. A small
curated semantic overlay promotes selected assets into stable answer-bearing
page icons/markers when a task contract explicitly needs that.

Runtime generation must load only local files through
`trace_tasks.tasks.pages.shared.page_visual_assets`.

## Role Counts

- `badge_spot`: 493
- `hero_anchor`: 234
- `section_illustration`: 493

## Contents

- `manifest.jsonl`: accepted asset metadata and role/category tags.
- `semantic_overlay.jsonl`: pages-owned answer-bearing icon/marker overlay
  keyed by stable semantic ids and backed by assets from `manifest.jsonl`.
- `sources.json`: source, license, count, and policy metadata.
- `raw/`: vendored source SVGs.
- `normalized/`: transparent PNGs used by renderers.
- `licenses/`: local license notes.

Assets from `manifest.jsonl` must remain decorative. Answer-bearing page
symbols must be loaded through
`trace_tasks.tasks.pages.shared.page_semantic_assets`, which records the semantic id,
display label, overlay manifest, resolved visual asset, and source metadata.
