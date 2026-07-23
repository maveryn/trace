# Program Schema Catalog

This catalog defines reusable program-schema names for Trace task contracts.
It is guidance for writing and reviewing objective contracts; it is not a task
inventory and does not merge public tasks by itself.

Use this document with `docs/contracts/TASK_UNIT_POLICY.md`.

## Core Rule

A program schema describes the concrete reasoning skeleton:

```text
candidate set -> filters / operands -> intermediate computation -> final operator -> output binding
```

Two tasks may share a program-schema name and still remain separate public tasks
when their scene contract, answer schema, annotation schema, prompt scaffold, or
visual witness roles differ.

Do not use a generic placeholder such as `compute(value)` or
`count(predicate)`. Pick the narrowest reusable schema that names the actual
operation.

## Count Schemas

| Schema | Use For | Do Not Merge With |
| --- | --- | --- |
| `count.direct_cardinality` | Count all visible units selected by one role or narrow predicate. | Attribute binding, scoped counts, relation counts, path/search counts, counterfactual counts. |
| `count.single_attribute_membership` | Count units matching one attribute axis or a set of values on one axis. | Multi-attribute Boolean predicates, arithmetic combinations, scoped predicates. |
| `count.multi_attribute_and` | Count units satisfying multiple attribute predicates conjunctively. | Single-axis membership, OR/XOR/exclusion/complement variants. |
| `count.multi_attribute_or` | Count units satisfying inclusive OR across attributes or predicates. | Single-axis set membership or arithmetic sums of separate counts. |
| `count.multi_attribute_xor` | Count units satisfying exactly one of multiple predicates. | Inclusive OR, exclusion, complement. |
| `count.multi_attribute_exclusion` | Count units satisfying one predicate while excluding another. | AND, OR, XOR, complement. |
| `count.multi_attribute_complement` | Count units satisfying none of a fixed predicate set. | Exclusion or XOR counts. |
| `count.scoped_attribute` | Select a spatial, structural, or named scope, then count units by attribute. | Unscoped attribute counts or relation counts. |
| `count.relation_attribute` | Count units selected by visible relation to another unit, region, path, or support object. | Plain scoped counts or metric-reference comparisons. |
| `count.adjacency_relation` | Count units adjacent to, neighboring, or bordering a reference. | General relation counts where adjacency is not the operation. |
| `count.reference_metric_relation` | Count units whose metric compares to a reference unit metric. | Exact attribute-match reference counts. |
| `count.one_bound_threshold` | Count units satisfying one threshold bound, such as above or below. | Two-bound interval predicates or multi-condition predicates. |
| `count.interval_predicate` | Count units satisfying a two-bound interval predicate. | One-bound threshold counts. |
| `count.group_predicate` | Count groups whose aggregate/member predicate satisfies the query. | Counting entities inside one group. |
| `count.pairwise_comparison` | Count aligned items where one side/series/profile wins against another. | Threshold counts against a fixed reference value. |
| `count.intersection_or_crossing` | Count intersections, crossings, overlaps, or comparable primitive contacts. | Threshold/category counts. |
| `count.sequence_or_line_pattern` | Count runs, lines, streaks, or sequence-pattern instances. | Unordered object counts. |
| `count.counterfactual` | Apply an explicit edit, then count in the edited state. | Static counts over the original scene. |

## Selection Schemas

| Schema | Use For | Do Not Merge With |
| --- | --- | --- |
| `selection.direct_label` | Return a label selected by a direct visible rule. | Numeric values or counts. |
| `selection.extreme_metric_label` | Select the label of an item with maximum/minimum metric. | Ranked non-extreme selection or threshold counts. |
| `selection.ranked_item` | Select an item by rank under a metric or order. | Direct lookup or unordered counts. |
| `selection.nearest_label` | Select the candidate nearest to a reference under a scene metric. | Generic extremum when the metric is not distance-to-reference. |
| `selection.option_match` | Choose a visible option panel/label matching a visual rule or transformation. | Free-form label answers. |
| `selection.option_value_match` | Compute a scene value, then select the option label whose value matches it. | Free-form numeric answers. |
| `selection.rule_violation` | Select the visible item/cell/index violating a displayed or implicit rule. | Completion or valid-option selection. |
| `selection.adjacency_relation_label` | Select the single unit satisfying an adjacency/predecessor/successor relation. | Adjacency counts or path-derived labels. |

## Numeric Schemas

| Schema | Use For | Do Not Merge With |
| --- | --- | --- |
| `numeric.direct_or_derived_value` | Return a numeric value computed from selected visible support. | Count or label-selection tasks. |
| `numeric.aggregate_sum` | Sum or total values over a selected support set. | Difference, ratio, rate, or counterfactual programs. |
| `numeric.summary_statistic` | Apply a summary statistic such as sum, mean, median, or average over selected values. | Tasks adding a different selection/filter/nested aggregation stage. |
| `numeric.difference_or_change` | Compute numeric difference/change between two selected supports. | Aggregate totals or ratio/rate programs. |
| `numeric.ranked_difference` | Rank items by a metric, then compute a difference between ranked items. | Direct extremum-label selection. |
| `numeric.ranked_value` | Select by rank and return the selected numeric value. | Label-returning ranked selection. |
| `numeric.extreme_metric_value` | Select an extreme item and return its numeric metric. | Label-returning extremum tasks. |
| `numeric.derived_metric` | Compute ratio, rate, percent/share conversion, angle conversion, or similar derived metric. | Direct readout or simple sum. |
| `numeric.count_arithmetic` | Combine two selector counts with arithmetic. | Boolean OR, direct single-selector counts, or counterfactual counts. |

## Path, Graph, Formula, And Simulation Schemas

| Schema | Use For | Do Not Merge With |
| --- | --- | --- |
| `path.shortest_path_value` | Find an optimal shortest path or its length. | Longest path, reachability, or route-following lookup. |
| `path.longest_path_value` | Find an optimal longest path under scene constraints. | Shortest path. |
| `topology.reachable_set` | Construct reachable nodes/cells/regions under movement or graph rules. | Shortest/longest path values. |
| `graph.component_count_or_size` | Construct components before returning count or size. | Local degree, path, or cut-structure programs. |
| `graph.minimum_spanning_tree` | Construct or score a minimum spanning tree. | Shortest path or flow/cut programs. |
| `graph.cut_structure_count` | Count structures whose removal/cut property changes connectivity or flow. | Component counting or degree filters. |
| `graph.path_distance_filter_count` | Count units at exact graph distance from a reference. | Simple route membership or shortest path values. |
| `graph.traversal_order` | Follow a specified traversal order. | Unordered reachability or path-length tasks. |
| `formula.solve_unknown` | Solve a numeric unknown from a domain formula/rule schema. | Different formula schemas needing different intermediate quantities. |
| `counterfactual.transform_then_answer` | Apply a specified hypothetical edit before computing the answer. | Direct readout/count tasks without scene transform. |
| `simulation.discrete_state_update` | Simulate a discrete state-update system under visible rules. | Static lookup or single-step counterfactual edits. |
| `probability.event_fraction` | Compute a reduced fraction from a visible finite sample space. | Different sample-space product or conditioning schemas. |

## Analysis-Level Reasoning Operations

Every active public task class declares one or more reasoning-operation
families in a literal `reasoning_operations` tuple. That executable declaration
is authoritative. Each task doc mirrors it in a machine-readable
`## Reasoning Operations` section. These families summarize the meaningful
operations that determine the answer. They support aggregate coverage analysis;
they are not public taxonomy nodes, task ids, query ids, or sampling units.

Use only these keys, in this order:

| Key | Include When |
| --- | --- |
| `direct_retrieval` | The answer is obtained by a terminal localized lookup or readout, with no symbolic decoding, transformation, or other meaningful operation. This is an exclusive fallback and cannot be combined with another family. |
| `filtering` | The program constructs an answer-determining subset using a predicate, relation, or named scope. Do not add it for ordinary candidate access before ranking, matching, or arithmetic. |
| `counting` | The program returns or materially uses the cardinality of a set, sequence, path, component, or group. |
| `comparison` | The program evaluates equality, scalar order, a threshold, an interval, or another answer-determining comparison relation. |
| `ranking` | The program orders candidates by a metric or sequence position, or selects an extremum, rank, nearest, farthest, earliest, or latest candidate. A field or card named "rank" and generic MCQ choice do not qualify. |
| `aggregation` | The program reduces a peer collection using sum, mean, median, cumulative total, share, mass, or an equivalent aggregate. Fixed-arity theorem arithmetic belongs to `formula_evaluation`, not aggregation. |
| `logical_composition` | The program combines predicates or sets using AND, OR, XOR, NOT, exclusion, union, intersection, or difference. |
| `spatial_relations` | The program infers or tests a geometric, positional, directional, overlap, containment, occlusion, or metric-spatial relation. Merely reading labeled dimensions is not spatial reasoning; path/link connectivity belongs to `topology`, and projection/folding belongs to `transformation`. |
| `topology` | Connectivity, paths, components, cycles, trees, graph traversal, reachability, or another topological relation materially determines the answer. |
| `transformation` | The program applies or infers a geometric, symbolic-decoding, representational, folding, projection, overlay, or reconstruction transform. |
| `state_update` | The program simulates, applies, or reasons over a move, action, counterfactual edit, or discrete state transition. |
| `formula_evaluation` | Arithmetic, algebra, a domain formula, a derived metric, or a numeric difference/ratio/rate materially determines the answer. |
| `matching` | The program tests substantive equivalence, correspondence, consistency, completion, rule satisfaction, or candidate-to-reference matching. Merely binding a computed result to an MCQ letter is not matching. |

Assignments are multi-label when the answer genuinely requires multiple
operations. For example, a thresholded count uses `filtering`, `counting`, and
`comparison`; a conjunctive attribute count additionally uses
`logical_composition`; selecting the category with the largest sum uses
`ranking` and `aggregation`. Do not label primitive visual access, answer
serialization, annotation construction, or MCQ letter binding as reasoning
operations.

The canonical source and task-doc forms are:

```python
reasoning_operations = ("filtering", "counting", "comparison")
```

```markdown
## Reasoning Operations

Families: `filtering`, `counting`, `comparison`
```

The order must follow the table above. `direct_retrieval` must appear alone.
Assignments must be checked against the complete Program Contract, including
all query branches; filename keywords are not sufficient. Source declarations
and task-document mirrors must stay synchronized.

## Assignment Guidance

When evaluating a task contract:

1. Start with the concrete task behavior, not the old task name.
2. Pick the most specific reusable schema above.
3. Add a new schema only when none of the existing schemas names the concrete
   reasoning skeleton.
4. Reuse of a schema across domains normalizes terminology only; it is never
   enough to merge public task ids.
5. Record the union of meaningful families used by a task's supported query
   branches. Different families do not by themselves force a split when the
   defined outer objective, candidate type, answer binding, annotation roles,
   and visible scaffold remain stable.
