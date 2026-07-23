# Shared Label Assets

This directory stores reusable prompt and render labels. Task modules load
these manifests through `trace_tasks.tasks.shared.name_assets` and apply
task-local filters for length, spacing, punctuation, and answer support.

## Manifests

- `people/first_names_ssa.txt`: SSA first names sorted by aggregate frequency.
- `people/surnames_census_2010.txt`: 2010 U.S. Census surnames.
- `places/countries_natural_earth.txt`: Natural Earth country/region labels.
- `places/cities_natural_earth.txt`: Natural Earth populated-place labels.
- `organizations/company_tickers_sec.txt`: alphabetic SEC ticker labels.
- `organizations/company_terms_sec.txt`: generic organization terms.
- `categories/`: synthetic category and status labels.
- `panel_titles/technical_topics.txt`: synthetic scientific topic labels.
- `occupations/occupations_bls_oews.txt`: BLS OEWS occupation titles.
- `industries/industries_bls_qcew.txt`: BLS QCEW industry titles.
- `mixed/proper_labels.txt`: broad proper-label pool.
- `mixed/compact_labels.txt`: compact alphabetic labels for tight layouts.

## Source Metadata

`sources.json` records source URLs, metadata URLs, licenses, local license
files, and row counts. Runtime generation uses only the vendored manifests.

## Chart Label Roles

Chart tasks use `trace_tasks.tasks.charts.shared.label_assets` as their adapter.
Dense categorical axes and repeated marks should use the compact ID resolver
provided by `sample_chart_labels()`. Semantic legends, series names, panel
names, table headers, and map categories use manifest-backed resolvers with
task-local length and spacing filters.

Updates must preserve deterministic ordering, normalize ASCII labels,
deduplicate case-insensitively, and refresh `sources.json` and local license
metadata.
