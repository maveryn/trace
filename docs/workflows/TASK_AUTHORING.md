# Task Authoring

1. Choose the public `domain`, visible `scene_id`, and objective contract.
2. Add the implementation at
   `src/trace_tasks/tasks/<domain>/<scene_id>/<objective_contract>.py`.
3. Add scene and task prompt templates under
   `src/trace_tasks/resources/prompts/<domain>/...`.
4. Add domain or scene defaults under
   `src/trace_tasks/resources/configs/domains/<domain>/`.
5. Construct a deterministic scene, query, render specification, execution
   trace, typed answer, and projected annotation.
6. Reject ambiguous or non-unique instances instead of weakening constraints.
7. Add focused tests and task documentation.

Use shared helpers only when they preserve the same semantic contract across
their callers. Domain-specific refinements are documented under
`docs/domains/`.
