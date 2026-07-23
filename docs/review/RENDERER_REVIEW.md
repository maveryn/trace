# Renderer Review

Review materialized images scene-by-scene and across the recipe's seed, query,
theme, font, layout, and context variation. Rendering review checks whether the
semantic contract is visually answerable; pixels are never verifier ground
truth.

## Checks

- Answer-bearing marks, labels, legends, controls, panels, and annotations are
  visible, unclipped, and legible.
- Text and semantic marks do not overlap; dense layouts retain useful spacing.
- Light and dark themes preserve contrast and semantic colors remain
  distinguishable.
- The prompt's names and option markers match visible labels exactly.
- Annotation overlays identify their intended witnesses without hiding the
  evidence being reviewed.
- Shared fonts, icons, illustration objects, and 3D object profiles remain
  recognizable at their actual task scale.
- Canvas size, padding, line width, and minimum object size follow the public
  rendering contracts.
- Fixed recipe inputs reproduce the same semantic state even when host-native
  encoding differs.

Use `scripts/audit_text_legibility.py` and `trace-review audit` for automated
signals, then inspect representative images manually. Automated measurements
do not waive a visible overlap, ambiguity, or fidelity defect.

Raw-pixel and PNG hashes are environment-sensitive. If semantic hashes agree,
non-strict verification reports rendering drift rather than changing the task
answer. Reproduce the manifest's native libraries and use
`trace-review verify --recipe <recipe-root> --strict-rendering` before making
byte-level claims.
