# Shared Context Text Assets

Trace keeps repo-wide reusable non-answer context text under
`src/trace_tasks/resources/assets/context_text/`. Use this layer when a renderer needs visual chrome,
captions, source notes, sidebar notes, callouts, decorative metric snippets,
legend-like explanatory blocks, or other distractor text that is not part of
the task answer.

These assets are different from `src/trace_tasks/resources/assets/labels/`: label assets are for
answer-bearing names, categories, nodes, regions, options, and visible fields.
Context text assets are for non-answer page/report/dashboard framing unless a
task explicitly scopes a specific text element into its verifier contract.

## Rules

1. Runtime generation must read vendored manifests only. Do not download text
   data during task generation.
2. Treat context strings as non-answer-bearing by default. If a task asks about
   one of these strings, move that text into the task-specific scene/query
   contract and record it as normal answer/annotation-bearing content.
3. Every drawn context text element must be trace-backed and bbox-backed. Record
   at least role, text, bbox, manifest path, source ids, and whether it is
   excluded from the answer contract.
4. Keep context text independent of answer value, correct option, query id, and
   difficulty bucket. It may vary by seed, scene style, or non-semantic layout
   mode.
5. Do not make distractor text a fixed single box. Scene adapters should
   randomize safe non-semantic choices such as placement, count, box size,
   heading/body/note manifests, and optional title presence when the scene has
   reserved space for that variation.
6. Context text should use the shared font assets documented in
   `docs/resources/SHARED_FONT_ASSETS.md`. Each context box should normally use
   one sampled family for its heading/body/note block, while surrounding chrome
   may use a separate sampled family.
7. Avoid graph-like, chart-like, or table-like distractors that can be mistaken
   for semantic marks in that scene. For example, graph scenes should not add
   extra node/edge-looking callouts, and chart scenes should not add unrelated
   mark-like symbols inside the plot area.
8. Place context text only after the scene layout mode is resolved, and compute
   public annotation after final layout. If content-frame mode translates the main
   scene, translate annotation bboxes/points with the same transform.
9. Keep task prompts concise. Do not mention context text unless the task
   explicitly needs the model to ignore or inspect that context.
10. Use only manifests with source/license metadata in `sources.json`. If a new
   manifest is added or regenerated, update source ids, counts, and local
   license references in the same change.

## Available Pools

- `phrases/headlines.txt` — short neutral titles/headlines.
- `phrases/captions.txt` — non-answer captions.
- `phrases/source_notes.txt` — source-note/footer-style strings.
- `phrases/footers.txt` — report/page footer strings.
- `phrases/sidebar_notes.txt` — sidebar and margin-note strings.
- `phrases/callout_phrases.txt` — short callout labels.
- `phrases/metric_snippets.txt` — decorative snippets with numbers.
- `phrases/legend_notes.txt` — legend-like explanatory blocks.
- `sentences/context_template_sentences.txt` — neutral one-sentence filler.
- `paragraphs/context_template_blocks.txt` — neutral two-sentence filler blocks.
- `paragraphs/context_long_blocks.txt` — neutral longer filler blocks for larger
  callout/sidebar text areas.
- `domains/industries_corpora.txt` and `words/*.txt` — source word/category
  pools used to build the template-derived manifests.

## Runtime Loading

Use the shared loader at
`src/trace_tasks/tasks/shared/context_text_assets.py`. Domain adapters should
use that loader instead of parsing `sources.json` independently.

The shared loader should expose:

- manifest loading by relative path;
- source/license lookup for a loaded manifest;
- deterministic sampling from an explicit `random.Random` or seed-derived
  sampler supplied by the caller;
- trace-friendly metadata for selected strings: manifest path, source ids, and
  row/index where practical.

## Source Metadata

Source and license metadata live in `src/trace_tasks/resources/assets/context_text/sources.json`.
Current sources are:

- Darius Kazemi Corpora Project word/category lists, recorded as `CC0-1.0`.
- Trace-authored synthetic context templates, recorded under the local
  `trace-synthetic.txt` note.

Local license files live under `src/trace_tasks/resources/assets/context_text/licenses/`.

## Intended Domain Use

Use these assets in domains whose renderers look like structured information
artifacts:

- charts: figure titles, captions, source notes, callout boxes, dashboard
  sidebars, non-answer metric cards.
- pages: headers, footers, section notes, sidebar blocks, form/document chrome,
  app-window/report framing.
- graph: captions, source notes, report framing, explanatory sidebars that are
  visually separate from nodes and edges.

Other domains may use the assets when they add document-like framing, but the
same non-answer and bbox-recording rules still apply.

For pages, the default adapter is intentionally conservative: it draws only
short safe-margin context text after the page artifact is rendered and skips any
candidate that would overlap traced scene entities. This keeps structured page
fields, controls, nodes, rows, routes, and annotation boxes as the only
answer-bearing text. Page adapters may also use controlled density variants for
one-sided or rare two-sided side-note blocks, but these blocks should remain
outside the structured artifact, use non-answer manifests, and stay balanced
against clean/light cases so the scene is not usually crowded.

When updating the vendored manifests, preserve deterministic ordering, refresh
`sources.json` and local license notes, and inspect representative strings.
Context text must remain neutral, concise, and visually plausible as page or
chart chrome.
