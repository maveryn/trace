# `task_physics__waveform_panel__wave_property_extremum_label`

## Summary
- Domain: `physics`
- Scene id: `waveform_panel`
- Implementation scene: `waveform_panel`
- Implementation source: `src/trace_tasks/tasks/physics/waveform_panel/wave_property_extremum_label.py`

## Program Contract

Program: `option_letter(arg_extreme(waveform_panels, property=amplitude_or_frequency_or_wavelength, direction=highest_or_lowest)); scene=waveform_panel; scope=wave_property_extremum_label`

Candidate set: the visible waveform option panels, amplitudes, periods, wavelengths, and panel labels inside the `wave_property_extremum_label` objective scope.
Operands: `waveform_panels` (semantic role, allowed `visible_labeled_sinusoid_panels_on_shared_scale`, source `program_schema_concrete`); `amplitude` (query operand, allowed `vertical_displacement_from_midline`, source `program_schema_concrete`); `frequency` (query operand, allowed `cycle_count_over_shared_horizontal_span`, source `program_schema_concrete`); `wavelength` (query operand, allowed `crest_spacing_over_shared_horizontal_span`, source `program_schema_concrete`); active `query_id` branch when present.
Operation: evaluate `option_letter` over the candidate set using the visible quantities, relations, branch semantics, and formulas encoded in the program expression; generation enforces a unique final answer.
Output binding: `answer` uses the `option_letter` schema; The answer value is the selected visible panel letter.
Annotation witnesses: `bbox` witnesses from the finalized render. Annotation is one bounding box around the selected waveform panel. Annotation must mark the minimal selected panel witness from the final rendered diagram. It must not mark every panel, background grid lines, title text, or derived property annotations.
Query ids: `highest_amplitude_label`, `lowest_amplitude_label`, `highest_frequency_label`, `lowest_frequency_label`, `longest_wavelength_label`, `shortest_wavelength_label`.

## Reasoning Operations

Families: `ranking`, `formula_evaluation`

## Task Contract
Selects the labeled waveform panel with the requested highest or lowest wave property.

## Query Branches

| Query id | Program schema |
| --- | --- |
| `highest_amplitude_label` | `option_letter(arg_extreme(waveform_panels, amplitude, direction=highest)); scene=waveform_panel; scope=wave_property_extremum_label` |
| `lowest_amplitude_label` | `option_letter(arg_extreme(waveform_panels, amplitude, direction=lowest)); scene=waveform_panel; scope=wave_property_extremum_label` |
| `highest_frequency_label` | `option_letter(arg_extreme(waveform_panels, frequency, direction=highest)); scene=waveform_panel; scope=wave_property_extremum_label` |
| `lowest_frequency_label` | `option_letter(arg_extreme(waveform_panels, frequency, direction=lowest)); scene=waveform_panel; scope=wave_property_extremum_label` |
| `longest_wavelength_label` | `option_letter(arg_extreme(waveform_panels, wavelength, direction=longest)); scene=waveform_panel; scope=wave_property_extremum_label` |
| `shortest_wavelength_label` | `option_letter(arg_extreme(waveform_panels, wavelength, direction=shortest)); scene=waveform_panel; scope=wave_property_extremum_label` |

## Program Metadata
- Program signatures: `physics.waveform_property_extremum`
- Base program contract: `option_letter(arg_extreme(waveform_panels, property=amplitude_or_frequency_or_wavelength, direction=highest_or_lowest)); scene=waveform_panel; scope=wave_property_extremum_label`
- Parameter axes: `query_id`, `scene_variant`, `panel_count`, `correct_option_letter`
- Arguments:
  - `waveform_panels`: semantic role; allowed `visible_labeled_sinusoid_panels_on_shared_scale`; source `program_schema_concrete`
  - `amplitude`: query operand; allowed `vertical_displacement_from_midline`; source `program_schema_concrete`
  - `frequency`: query operand; allowed `cycle_count_over_shared_horizontal_span`; source `program_schema_concrete`
  - `wavelength`: query operand; allowed `crest_spacing_over_shared_horizontal_span`; source `program_schema_concrete`
- Argument metadata status: `curated`
- Supported query ids: `highest_amplitude_label`, `lowest_amplitude_label`, `highest_frequency_label`, `lowest_frequency_label`, `longest_wavelength_label`, `shortest_wavelength_label`

## Answer Contract
- Answer schema: `option_letter`
- Generator `answer_gt.type`: `option_letter`
- The answer value is the selected visible panel letter.

## Annotation Contract
- Annotation schema: `bbox`
- Generator `annotation_gt.type`: `bbox`
- Annotation is one bounding box around the selected waveform panel.
- Annotation must mark the minimal selected panel witness from the final rendered diagram. It must not mark every panel, background grid lines, title text, or derived property annotations.
- Annotation and answer must be projected from the same generated execution trace, not inferred from pixels or prompt text.

## Prompt And Trace Requirements
- Prompt text must come from `src/trace_tasks/resources/prompts/physics/waveform_panel/physics_waveform_panel_v1.json`, with scene, task, query, and output layers selected deterministically and recorded in metadata.
- Render randomness, sampled fonts/styles, panel count, waveform amplitudes, cycle counts, selected panel label, and verifier payloads must be explicit in the instance trace.
- Diagrams must keep all panels on a shared horizontal scale, with labels and midlines readable.
