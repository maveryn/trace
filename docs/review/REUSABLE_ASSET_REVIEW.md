# Reusable Asset Review

Use the review application's resource surfaces when a public illustration
object or 3D object profile changes. Review shared fonts and icons through
their downstream task images. Resource review supplements the licensing and
provenance rules in `docs/ASSETS.md` and `docs/resources/README.md`.

## Procedure

1. Open the catalog metadata in the review application and identify every
   changed entry and its downstream scenes.
2. Materialize representative downstream tasks and inspect each entry at the
   smallest scale used by a task.
3. If a contribution has its own deterministic preview producer, write one
   manifest-and-PNG pair using the contract below. The public review CLI does
   not synthesize asset-only scenes, and loose image files are not indexed.
4. Record `approve`, `remove`, or `improve` with a concise reason.
5. Export decisions as a local report when needed; do not commit preview
   images or the feedback database.

Check recognizability, transparency, clipping, palette contrast, label
agreement, style consistency, deterministic selection, source attribution, and
license compatibility. For 3D assets, also inspect camera framing, depth,
occlusion, floor/wall contact, and silhouette. For fonts, inspect every glyph
used by answer-bearing text rather than only a sample word.

The catalog remains usable for metadata review when no preview has been
materialized; do not treat a missing preview as visual approval.

## Optional preview manifest contract

Store each pair directly under one canonical kind directory. The manifest and
PNG use the same filesystem-safe stem. For example, the repo-relative paths for
an illustration preview may be:

```text
review/task-reviews/assets/illustrations/apple-preview.json
review/task-reviews/assets/illustrations/apple-preview.png
```

The manifest is exactly `trace-review-asset-preview-v1`; extra or missing keys
are rejected. Its `image.path` is relative to the configured
`review/task-reviews` review root, not the manifest file:

```json
{
  "schema": "trace-review-asset-preview-v1",
  "kind": "illustrations",
  "asset_id": "apple",
  "image": {
    "path": "assets/illustrations/apple-preview.png",
    "format": "png",
    "mode": "RGB",
    "width": 320,
    "height": 240,
    "sha256": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
  }
}
```

Replace the illustrative digest with the SHA-256 of the exact PNG bytes.
`kind` is exactly `illustrations` or `three_d`; `asset_id` is the corresponding
public catalog id. The path must be the lower-case `.png` sibling of the
manifest under the matching kind directory. `mode`, `width`, and `height` must
match Pillow's decoded image. The index rejects symlinks, path escapes, hash
mismatches, non-PNG bytes (including SVG renamed to `.png`), duplicate manifests
for one catalog entry, and previews for unknown catalog ids.
