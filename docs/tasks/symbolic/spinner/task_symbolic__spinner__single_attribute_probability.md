# `task_symbolic__spinner__single_attribute_probability`

## Public Taxonomy
1. Domain: `symbolic`
2. Scene id: `spinner`
3. Task id: `task_symbolic__spinner__single_attribute_probability`
4. Objective: single-spinner attribute probability.

## Program Contract
Program: `spinner.single_attribute_probability(scene=spinner, scope=one_equal_sector_spinner, attribute=color|shape, sample_space=visible_sectors, output=probability_option_letter)`

Candidate set: all equal-area visible sectors on the spinner.
Operands: each sector's target attribute value and the resolved color or shape predicate.
Operation: count sectors matching the requested single-attribute predicate and reduce `matching_sectors / total_sectors`.
Output binding: `answer` is the letter of the visible A-F option whose reduced fraction is the requested probability.
Annotation schema: `bbox`.
Annotation witnesses: the scalar bbox of the full spinner panel.
Query ids: `single_color_probability`, `single_shape_probability`.

## Reasoning Operations

Families: `filtering`, `counting`, `formula_evaluation`

## Query Contract
- `single_color_probability`: target predicate is one resolved sector color.
- `single_shape_probability`: target predicate is one resolved sector shape marker.

## Generation And Trace
1. The task samples one semantic predicate branch and asks for the probability that a uniformly selected sector satisfies it.
2. The execution trace records sector specs, favorable outcome counts, total outcome counts, the exact reduced fraction, visible option fractions, and the selected answer label.
3. Scene variants are `spinner_clean`, `spinner_card`, and `spinner_notebook`.
4. Annotation is projected from finalized spinner-panel geometry, not from pixels.
5. Scalar annotation checked: true.

## Implementation
1. Registered class: `trace_tasks.tasks.symbolic.spinner.single_attribute_probability.SymbolicSpinnerSingleAttributeProbabilityTask`
2. Config: `src/trace_tasks/resources/configs/domains/symbolic/spinner.yaml`
3. Prompt asset: `src/trace_tasks/resources/prompts/symbolic/spinner/symbolic_spinner_v1.json`
4. Focused tests: `tests/test_symbolic_probability_spinner_tasks.py`
