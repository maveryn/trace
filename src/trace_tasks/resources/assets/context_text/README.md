# Shared Context Text Assets

This directory stores reusable non-answer text pools for visual context,
chrome, captions, callouts, sidebars, source notes, and distractor text.
These assets are intended for chart, graph, and page context layers.

Task renderers should treat these strings as non-semantic unless a task
explicitly scopes them into the verifier contract. Context-layer metadata
must record role, bbox, source manifest, and exclusion status for every
drawn text element.

## Source Policy

Assets are normalized from CC0/public-domain compatible sources and
project-local Trace synthetic templates. `sources.json` records source
URLs, local license files, and per-manifest counts.

When updating these manifests, preserve deterministic ordering and refresh the
source counts and local license references in `sources.json`.
