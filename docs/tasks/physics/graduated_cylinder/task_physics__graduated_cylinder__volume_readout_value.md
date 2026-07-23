# `task_physics__graduated_cylinder__volume_readout_value`

## Summary
- Domain: `physics`
- Scene id: `graduated_cylinder`
- Implementation scene: `graduated_cylinder`
- Implementation source: `src/trace_tasks/tasks/physics/graduated_cylinder/volume_readout_value.py`

## Task Contract
Reads the liquid volume from one visible graduated cylinder using the meniscus and scale.

## Program Contract

Program: `integer(read_scale_value(meniscus, graduated_scale, unit=mL)); scene=graduated_cylinder; scope=volume_readout_value`

Candidate set: the visible cylinder readouts, liquid levels, tick marks, numeric labels, and unit labels inside the `volume_readout_value` objective scope.
Operands: `meniscus` (semantic_role, allowed `visible_liquid_level`, source `program_schema_concrete`); `graduated_scale` (semantic_role, allowed `visible_tick_scale_with_mL_units`, source `program_schema_concrete`).
Operation: evaluate `integer` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `integer_value` schema; The answer value is the exact integer mL value shown by the scale.
Annotation witnesses: `bbox` witnesses from the finalized render. Annotation is one box around the graduated-cylinder readout, including the cylinder body, liquid level, tick marks, numeric scale labels, and mL unit. Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
Query ids: `single`.

## Reasoning Operations

Families: `direct_retrieval`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `read_scale_value(meniscus, graduated_scale, unit=mL); scene=graduated_cylinder; scope=volume_readout_value; query_branch=single_cylinder_volume_readout` |

## Program Metadata
- Program signatures: `physics.graduated_cylinder_volume_readout`
- Base program contract: `read_scale_value(meniscus, graduated_scale, unit=mL); scene=graduated_cylinder; scope=volume_readout_value`
- Parameter axes: `fixed_query`
- Arguments:
  - `meniscus`: semantic_role; allowed `visible_liquid_level`; source `program_schema_concrete`
  - `graduated_scale`: semantic_role; allowed `visible_tick_scale_with_mL_units`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`

## Answer Contract
- Answer schema: `integer_value`
- Generator `answer_gt.type`: `integer`
- The answer value is the exact integer mL value shown by the scale.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation is one box around the graduated-cylinder readout, including the cylinder body, liquid level, tick marks, numeric scale labels, and mL unit.
- Annotation must mark minimal visual witnesses from the final rendered diagram, not answer labels, option choices, decorative chrome, or derived numeric annotations unless those are the queried visual witnesses.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.
- Scalar annotation checked: `true`

## Prompt And Trace Requirements
- Prompt text must come from the physics graduated-cylinder v1 prompt bundle, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, query operands, formula quantities, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep all quantities required for the physics computation visible or explicitly stated by the task prompt contract.
