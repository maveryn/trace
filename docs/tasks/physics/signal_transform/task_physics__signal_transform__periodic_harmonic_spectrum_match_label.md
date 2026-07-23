# `task_physics__signal_transform__periodic_harmonic_spectrum_match_label`

## Summary
- Domain: `physics`
- Scene id: `signal_transform`
- Implementation scene: `signal_transform`
- Implementation source: `src/trace_tasks/tasks/physics/signal_transform/periodic_harmonic_spectrum_match_label.py`

## Program Contract

Program: `option_letter(select(spectrum_options, harmonic_pattern=periodic_wave_harmonics(input_waveform))); scene=signal_transform; scope=periodic_harmonic_spectrum_match_label`

Candidate set: the visible input waveform panel, spectrum option panels, frequency markers, and labels inside the `periodic_harmonic_spectrum_match_label` objective scope.
Operands: `input_waveform` (semantic_role, allowed `visible_square_triangle_or_sawtooth_waveform`, source `program_schema_concrete`); `spectrum_options` (semantic_role, allowed `visible_labeled_one_sided_magnitude_spectra`, source `program_schema_concrete`); `harmonic_pattern` (query_operand, allowed `odd_harmonics_slow_decay|odd_harmonics_fast_decay|all_harmonics_slow_decay`, source `program_schema_concrete`).
Operation: evaluate `option_letter` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; The answer value is the selected visible spectrum option letter from `A` through `D`.
Annotation witnesses: `bbox_map` witnesses from the finalized render. Annotation keys are `input_waveform` and `selected_spectrum`. Annotation must mark the input time-domain waveform panel and the selected matching spectrum option panel. It must not mark every option, decorative axes/grid lines, title text, or hidden semantic labels.
Query ids: `single`.

## Reasoning Operations

Families: `transformation`, `matching`

## Query Branches

| Query id | Program schema |
| --- | --- |
| `single` | `option_letter(select(spectrum_options, harmonic_pattern=periodic_wave_harmonics(input_waveform))); scene=signal_transform; scope=periodic_harmonic_spectrum_match_label` |

## Program Metadata
- Program signatures: `physics.periodic_harmonic_spectrum_match`
- Base program contract: `option_letter(select(spectrum_options, harmonic_pattern=periodic_wave_harmonics(input_waveform))); scene=signal_transform; scope=periodic_harmonic_spectrum_match_label`
- Parameter axes: `scene_variant`, `waveform_family`, `correct_option_letter`
- Arguments:
  - `input_waveform`: semantic_role; allowed `visible_square_triangle_or_sawtooth_waveform`; source `program_schema_concrete`
  - `spectrum_options`: semantic_role; allowed `visible_labeled_one_sided_magnitude_spectra`; source `program_schema_concrete`
  - `harmonic_pattern`: query_operand; allowed `odd_harmonics_slow_decay|odd_harmonics_fast_decay|all_harmonics_slow_decay`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `single`
- Internal trace query branch: `periodic_wave_harmonic_spectrum`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is the selected visible spectrum option letter from `A` through `D`.

## Annotation Contract
- Annotation schema: `bbox_map`
- Generator `annotation_gt.type`: `bbox_map`
- Annotation keys are `input_waveform` and `selected_spectrum`.
- Annotation must mark the input time-domain waveform panel and the selected matching spectrum option panel. It must not mark every option, decorative axes/grid lines, title text, or hidden semantic labels.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/signal_transform/physics_signal_transform_v1.json`, with scene and task/query layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, waveform family, four-option order, selected option letter, spectrum signatures, and verifier payloads must be explicit in the instance trace.
- Diagrams use one-sided magnitude spectra in the initial public version. Equation choices, phase spectra, dense tick labels, and hidden waveform-family text are intentionally excluded.
