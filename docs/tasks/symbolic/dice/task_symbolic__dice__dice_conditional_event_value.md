# `task_symbolic__dice__dice_conditional_event_value`

## Public Taxonomy
1. Domain: `symbolic`
2. Scene id: `dice`
3. Task id: `task_symbolic__dice__dice_conditional_event_value`
4. Objective: single-tray conditional probability.

## Program Contract
Program: `dice.conditional_event_probability(scene=dice, scope=single_visible_tray, condition=color|value_property|value_set, event=value_property|color, output=probability_option_letter)`

Candidate set: the visible dice in the single tray after filtering by the query condition.
Operands: each die's color and top-face value, the resolved condition predicate, and the resolved event predicate.
Operation: restrict the sample space to dice satisfying the condition, count conditioned dice also satisfying the event, and reduce `favorable / conditioned_total`.
Output binding: `answer` is the letter of the visible A-F option whose reduced fraction is the requested probability.
Annotation schema: `bbox`.
Annotation witnesses: the scalar bbox of the single visible dice tray.
Query ids: `conditional_value_property_given_color_probability`, `conditional_color_given_value_property_probability`, `conditional_color_given_value_set_probability`.

## Reasoning Operations

Families: `filtering`, `counting`, `formula_evaluation`

## Generation And Trace
1. The task asks for a conditional probability over one uniformly selected die from the shown tray.
2. The execution trace records tray specs, die colors, die values, favorable outcome counts, total outcome counts, the exact reduced fraction, visible option fractions, and the selected answer label.
3. Scene variants are `dice_tray_clean`, `dice_tray_felt`, and `dice_tray_notebook`; dice visual style is non-semantic rendering metadata.
4. Annotation is projected from finalized tray geometry, not from pixels.
5. Scalar annotation checked: true.

## Implementation
1. Registered class: `trace_tasks.tasks.symbolic.dice.dice_conditional_event_value.SymbolicProbabilityDiceConditionalEventValueTask`
2. Config: `src/trace_tasks/resources/configs/domains/symbolic/dice.yaml`
3. Prompt asset: `src/trace_tasks/resources/prompts/symbolic/dice/symbolic_dice_probability_v1.json`
4. Focused tests: `tests/test_symbolic_probability_dice_tasks.py`
