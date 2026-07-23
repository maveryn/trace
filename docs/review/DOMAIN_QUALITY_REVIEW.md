# Domain Quality Review

Use this audit to decide whether a domain is coherent and maintainable as a
public task surface. It does not replace source-layout, taxonomy, prompt,
annotation, renderer, or verifier contracts.

## Audit

1. Compare the live registry, active inventory, task docs, source packages,
   configs, prompts, and recipe coverage.
2. Confirm each scene has a distinct visible grammar rather than a cosmetic or
   implementation-history distinction.
3. Compare task signatures within each scene: program, answer schema,
   annotation schema, candidate set, prompt scaffold, and witness roles.
4. Confirm query ids are narrow semantic branches and not hidden styles,
   layouts, sampled values, or separate programs.
5. Review prompt, annotation, renderer, verifier/trace, distribution, and
   shared-helper consistency across the domain.
6. Report findings as `blocker`, `fix_before_review`, `cleanup`, `follow_up`, or
   `accepted`, with affected domain/scene/task/query and required validation.

Audit without changing implementation unless the user requests fixes. Keep
reports free of generated images, feedback databases, machine paths, and
model-campaign acceptance state.
