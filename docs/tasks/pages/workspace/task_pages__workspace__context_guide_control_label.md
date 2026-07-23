# `task_pages__workspace__context_guide_control_label`

## Identity
1. Domain: `pages`
2. Scene id: `workspace`
3. Source scene: `workspace`
4. Task id: `task_pages__workspace__context_guide_control_label`

## Contract
1. Objective: identify the labeled workspace control matching a context cue and a directly stated Key column.
2. Public task contract: `context_guide_control_label`
3. Supported `query_id` values: `single`
4. Answer type: `option_letter`
5. Annotation schema: `bbox`
6. Annotation witness: the selected target control box.
7. Query argument axes: target context cue among three context rows, target Key column, target candidate label, workspace variant, and style variant.

## Program Contract
- `workspace_context_guide_control_label(context_cue, key_column); output=option_letter; annotation=bbox(target_control); scene=workspace; scope=one professional application workspace`

## Reasoning Operations

Families: `matching`

## Prompt + Trace
1. Prompt bundle: `pages_workspace_v1`
2. Scene key: `workspace`
3. Task key: `workspace_control_query`
4. Prompt query key: `context_guide_control_lookup`
5. Runtime `query_id` is `single`; workspace flavor is recorded as `workspace_variant`.
6. Trace records context cue cards, context rows, coded headers, target control metadata, sampled workspace/app/style metadata, and prompt metadata.
