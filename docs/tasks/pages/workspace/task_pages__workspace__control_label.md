# `task_pages__workspace__control_label`

## Identity
1. Domain: `pages`
2. Scene id: `workspace`
3. Source scene: `workspace`
4. Task id: `task_pages__workspace__control_label`

## Contract
1. Objective: identify the labeled workspace control matching the visible instruction, guide cue, context row, and coded header.
2. Public task contract: `control_label`
3. Supported `query_id` values: `single`
4. Answer type: `option_letter`
5. Annotation schema: `bbox`
6. Annotation witness: the target labeled control box.
7. Query argument axes: target context row, cue/header column, target candidate label, workspace variant, and style variant.

## Program Contract
- `workspace_control_label(instruction, guide_cue, context_row, coded_header); output=option_letter; annotation=bbox(target_control); scene=workspace; scope=one professional application workspace`

## Reasoning Operations

Families: `matching`

## Prompt + Trace
1. Prompt bundle: `pages_workspace_v1`
2. Scene key: `workspace`
3. Task key: `workspace_control_query`
4. Prompt query key: `target_control_lookup`
5. Runtime `query_id` is `single`; workspace flavor is recorded as `workspace_variant`.
6. Trace records guide cards, context rows, coded headers, target control metadata, sampled workspace/app/style metadata, and prompt metadata.
