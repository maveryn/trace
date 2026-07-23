# Shared Label Assets

Trace keeps repo-wide reusable text labels under `src/trace_tasks/resources/assets/labels/`. Use this
layer for visible node labels, chart categories, page names, legend names,
organization-style tokens, place names, and other prompt/render labels that are
not task-specific.

## Rules

1. Load labels through `trace_tasks.tasks.shared.name_assets`.
   Graph-domain tasks should use `trace_tasks.tasks.graph.shared.label_assets`
   instead, which applies the graph label caps and eligible-bucket rules on top
   of these same manifests.
2. Keep task-specific constraints at the task call site. For example, a graph
   task that needs short edge labels should filter `mixed/compact_labels.txt`
   with `min_chars`, `max_chars`, `allow_spaces`, and `allow_punctuation`
   instead of creating a graph-only manifest.
3. Do not hardcode new one-off proper-name lists inside task modules.
4. Use only permissively licensed or public-domain sources. Add the source and
   local license metadata to `src/trace_tasks/resources/assets/labels/sources.json` when a manifest is
   added or regenerated.
5. Runtime generation must read vendored manifests only. Do not download label
   data during task generation.
6. If a task needs answer strings, make sure the rendered label, answer value,
   and verifier normalization policy are decided together. Avoid silent
   case-folding unless the verifier explicitly supports it.

## Available Pools

- `people/first_names_ssa.txt`
- `people/surnames_census_2010.txt`
- `places/countries_natural_earth.txt`
- `places/cities_natural_earth.txt`
- `organizations/company_tickers_sec.txt`
- `organizations/company_terms_sec.txt`
- `categories/abstract_group_labels.txt`
- `categories/priority_labels.txt`
- `categories/product_labels.txt`
- `categories/status_labels.txt`
- `occupations/occupations_bls_oews.txt`
- `industries/industries_bls_qcew.txt`
- `mixed/proper_labels.txt`
- `mixed/compact_labels.txt`
- Generated chart-only entity bucket: `temporal`

For chart-domain tasks, prefer `trace_tasks.tasks.charts.shared.label_assets` over
calling `load_label_manifest(...)` directly. That helper exposes reusable
entity-label and category-label bucket selection while still recording the
manifest/filter metadata needed for trace diagnostics. The chart `temporal`
entity bucket is generated at runtime rather than loaded from a manifest because
it must return ordered sequences. It participates in the same bucket-weight
maps as manifest-backed entity buckets and may internally sample consecutive
years, fiscal years, quarters, months, or month-day labels while preserving
chronological order.

## API

```python
from trace_tasks.tasks.shared.name_assets import load_label_manifest

labels = load_label_manifest(
    "mixed/compact_labels.txt",
    min_chars=5,
    max_chars=10,
    allow_spaces=False,
    allow_punctuation=False,
    compact_length=False,
)
```

For short alphabetic token pools such as Braille/Morse word options, use the
global helper built on this same manifest layer:

```python
from trace_tasks.tasks.shared.word_assets import load_short_word_bank_by_length

words_by_length = load_short_word_bank_by_length(min_length=3, max_length=5)
```

The helper defaults to a common leading slice of `people/first_names_ssa.txt`,
normalizes tokens to lowercase, groups by exact length, and keeps up to 1000
candidates per length bucket. Override the manifest, length support, or pool cap
only when a task has a concrete rendering or semantics reason.

Do not create scene-local word lists when a filtered shared label manifest is
adequate.

Use `load_label_sources()` when runtime tooling or documentation needs source
and license metadata.

When updating a vendored manifest, normalize ASCII labels, deduplicate them
case-insensitively, and update `sources.json` and local license metadata in the
same change.
