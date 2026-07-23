# Annotation Review

Review annotations against
`docs/contracts/ANNOTATION_AND_REWARD_CONTRACTS.md`. The annotation must be a
minimal visual witness for the answer and must be projected from the same
execution trace as the typed answer.

## Checks

- Task documentation, prompt hint, emitted `annotation_gt`, projected
  annotation, and verifier schema agree.
- A single witness uses a scalar point, box, or segment; unordered homogeneous
  witnesses use a set; ordered witnesses use a sequence; distinct semantic
  roles use a map.
- Boxes identify area-like objects, points identify compact locations, and
  segments identify edges, paths, spans, or intervals.
- Boxes satisfy the public minimum-side policy and remain unambiguous after
  padding.
- The payload marks answer-bearing visual evidence rather than answer labels,
  decorative context, or an unnecessarily large scene region.
- Similar witnesses in a domain use a consistent annotation family unless the
  domain contract documents a semantic exception.
- Coordinate order, bounds, cardinality, role keys, and endpoint symmetry match
  the verifier contract.

Report each task as `good`, `bad`, or `borderline`. Treat schema disagreement,
wrong cardinality, unstable geometry, and answer/annotation trace divergence as
clear failures. Use `borderline` when two geometries remain defensible and
recommend the choice most consistent with neighboring public tasks.
