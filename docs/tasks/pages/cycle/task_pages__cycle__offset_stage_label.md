# `task_pages__cycle__offset_stage_label`

## Identity
1. Domain: `pages`
2. Scene id: `cycle`
3. Source scene: `cycle`
4. Task id: `task_pages__cycle__offset_stage_label`

## Contract
1. Objective: return the exact visible stage label reached by walking a requested number of steps before or after a named stage in one directed cycle.
2. Public task contract: `offset_stage_label`
3. Supported `query_id` values: `after_stage_offset_label`, `before_stage_offset_label`
4. Answer type: `string`
5. Annotation schema: `bbox`
6. Annotation witness: the full box of the correct destination stage.
7. Query argument axes: before/after relationship, queried stage label, step count, cycle direction, and scene variant.

## Program Contract
- `stage_at_offset(cycle_order, start_stage=query_stage, relation=before_or_after, steps=k, arrow_direction=cycle_direction); output=stage_label; annotation=bbox(destination_stage); scene=cycle; scope=one directed cycle diagram`

## Reasoning Operations

Families: `ranking`, `topology`

## Prompt + Trace
1. Prompt bundle: `pages_cycle_v1`
2. Scene key: `cycle_diagram`
3. Task key: `offset_stage_query`
4. Prompt query key: `offset_stage_label`; the public `query_id` selects the before/after relationship used in the prompt slots.
5. Trace records stage order, arrow direction, queried stage, step count, before/after relationship, destination stage id, final stage bboxes, and sampled visual metadata.
6. Generation is deterministic from `instance_seed`; answer and annotation come from the finalized rendered target stage.
