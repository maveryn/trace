# `task_symbolic__spinner__multi_attribute_or_probability`

## Public Taxonomy
1. Domain: `symbolic`
2. Scene id: `spinner`
3. Task id: `task_symbolic__spinner__multi_attribute_or_probability`
4. Objective: color-or-shape spinner probability.

## Program Contract
Program: `spinner.color_shape_or_probability(scene=spinner, scope=one_equal_sector_spinner, predicate=color_or_shape, sample_space=visible_sectors, output=probability_option_letter)`

Candidate set: all equal-area visible sectors on the spinner.
Operands: each sector's color and shape marker, plus the resolved target color and target shape.
Operation: count sectors satisfying either target attribute, counting overlap once, and reduce `matching_sectors / total_sectors`.
Output binding: `answer` is the letter of the visible A-F option whose reduced fraction is the requested probability.
Annotation schema: `bbox`.
Annotation witnesses: the scalar bbox of the full spinner panel.
Query ids: `single`.

## Reasoning Operations

Families: `filtering`, `counting`, `logical_composition`, `formula_evaluation`

## Generation And Trace
1. The task samples one color-shape disjunction with a nontrivial favorable count.
2. The execution trace records sector specs, favorable outcome counts, total outcome counts, the exact reduced fraction, visible option fractions, and the selected answer label.
3. Scene variants are `spinner_clean`, `spinner_card`, and `spinner_notebook`.
4. Annotation is projected from finalized spinner-panel geometry, not from pixels.
5. Scalar annotation checked: true.

## Implementation
1. Registered class: `trace_tasks.tasks.symbolic.spinner.multi_attribute_or_probability.SymbolicSpinnerMultiAttributeOrProbabilityTask`
2. Config: `src/trace_tasks/resources/configs/domains/symbolic/spinner.yaml`
3. Prompt asset: `src/trace_tasks/resources/prompts/symbolic/spinner/symbolic_spinner_v1.json`
4. Focused tests: `tests/test_symbolic_probability_spinner_tasks.py`
