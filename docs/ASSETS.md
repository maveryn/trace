# Assets

Runtime assets are packaged under `src/trace_tasks/resources/assets/` so
generation works deterministically without downloading mutable resources at
task execution time.

- `fonts/`: curated font families and per-family licenses
- `icons/`: icon manifests, SVGs, source metadata, and attribution
- `pages/`: visual assets used by page and interface tasks
- `charts/`: map geometry and chart label resources
- `context_text/`: deterministic context text pools
- `labels/`: names and label vocabularies

Repository branding is documentation-only and lives under
`docs/assets/brand/`.

Task code should access resources through shared loaders rather than embedding
machine-specific paths. New third-party resources must include provenance,
license text, and deterministic selection metadata.

See
[THIRD_PARTY_NOTICES.md](https://github.com/maveryn/trace/blob/main/THIRD_PARTY_NOTICES.md)
for the top-level attribution summary.
