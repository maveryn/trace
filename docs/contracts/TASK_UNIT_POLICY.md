# Trace Task-Unit Policy

This document defines what counts as one public Trace task.

Use it when proposing tasks, deciding whether a branch is a `query_id` or a new
task id, and auditing split/merge decisions under uniform task-level sampling.

## 1) Core Rule
A public task must have one stable:

```text
scene contract + objective contract
```

The objective contract has three required parts:

1. **Answer schema** — prompt-facing answer type and shape, such as integer
   count, numeric value, option letter, string label, reduced fraction, or list.
2. **Annotation schema** — prompt-facing annotation type and semantic witness
   structure, such as unordered points, ordered path points, object bboxes, or
   keyed role-bound witnesses.
3. **Program schema** — the concrete reasoning skeleton over the scene. It must
   name the candidate set, operand roles, intermediate computation, final
   operator, output binding, and annotation witness roles.

Use `docs/contracts/PROGRAM_SCHEMA_CATALOG.md` for reusable program-schema names and
do-not-merge boundaries. The catalog normalizes terminology; it does not merge
public tasks by itself.

Same scene, answer type, or annotation type is not enough to merge tasks. Merge
only when all three objective-contract fields and the query-facing visual
scaffold remain stable.

## 2) Program Schema Requirement
Do not approve vague program descriptions such as:

```text
count(objects where predicate)
select_by_rank(items, metric, rank)
compute(value)
filter(items, condition)
```

Those are useful draft labels, but final task decisions need concrete schemas,
for example:

- count legal destination cells for one marked piece under the shown move rule;
- select the option board equal to applying the shown move to the source board;
- count visible objects in a named row that satisfy one color predicate;
- compute a formula-derived unknown from marked operand labels.

If the hidden intermediate objects differ, split the task unless the difference
is only a bounded parameter of the same concrete schema.

## 3) Query IDs
Use **query id** in prose and `query_id` as the metadata field.

A query id is allowed only for narrow parameter branches inside one objective
contract:

1. mirrored directions such as left/right, above/below, X/O, red/blue;
2. bounded rank or threshold parameters over the same candidate set;
3. target attributes over the same visible witness family;
4. operand variations that keep the same answer schema, annotation schema,
   program schema, and query-facing scaffold;
5. a finite predicate or rule-family argument inside an explicitly defined
   higher-order objective whose candidate type, outer computation, output
   binding, witness roles, and visible scaffold remain fixed.

A query id must not choose between public objectives. If changing `query_id`
changes the answer schema, annotation schema, concrete program schema, semantic
witness roles, or visible task scaffold, split the public task.

Operation-family metadata is descriptive and multi-label; it does not define
task boundaries by itself. A retained task may therefore cover several
predicate families when its public objective is explicitly the evaluation of a
prompt-bound predicate under one stable outer program. The task-level operation
declaration records the union across supported branches.

For attribute-heavy scenes, distinguish literal operands from visual reasoning
channels. Changing the literal target within one channel, such as red to blue
or circle to square, is usually a task parameter. Switching channels or arity,
such as type match vs color match, color-pattern violation vs size-pattern
violation, shape-only lookup vs color+shape lookup, or one-attribute comparison
vs multi-attribute binding, is a public task split when it changes the visual
scan/reasoning operation. Operators over the same operand roles may remain
queries when the program scaffold and witness roles stay fixed, such as
`total_count` vs `difference_count` over the same two count operands.

For map annotation, the schema is defined by annotation type, semantic role
family, and cardinality/order requirements, not by fixed literal key names.
Different samples or query branches may bind different visible labels as keys
when those keys fill the same witness roles. For example, a segment-length task
that always annotates the two requested segment endpoints can use keys `A,Y` in
one sample and `X,B` in another without becoming a new annotation schema.

Every active public task should declare `supported_query_ids`. Tasks with
no semantic query branches use the single repo-wide sentinel `("single",)`.
Do not use `default`, the task id, or the objective-contract name as a
single-query placeholder; those values blur public task identity with internal
branch metadata.

## 4) Public Task Selection
`task_id` is the public selector. Callers should request:

```text
task_games__2048__merge_count
```

not a broad task plus `params["query_id"]` to select the objective.

Public task files resolve and validate `query_id`, then pass semantic arguments
into shared helpers. Scene shared code must not branch on public task ids or
query ids.

## 5) When To Split
Create a new task when a branch changes any of these:

1. answer shape or type;
2. annotation type or semantic witness roles;
3. reasoning program or intermediate object set;
4. unit of visual attention, such as nodes vs edges, rows vs cells, or pieces vs
   destinations;
5. query-facing scaffold, such as board-only vs source-plus-option-panels;
6. final output binding, such as selecting an object vs counting all qualifying
   objects.

Common split signals:

- unordered subset counting vs ordered sequence/path reasoning;
- one-bound predicate count vs two-bound interval count when the predicate
   itself defines the objective rather than parameterizing one defined outer
  program;
- choosing a move vs computing the resulting state;
- reading a value vs comparing/ranking multiple values;
- selecting an answer option image instead of selecting a source-scene object.

## 6) What May Vary Inside One Task
The following do not by themselves require a new task:

1. non-semantic style, palette, font, noise, and layout variation;
2. object counts, labels, names, colors, values, and placements;
3. mirrored directions or players over the same rule;
4. local arithmetic difficulty over the same operand roles;
5. answer range, board size, or item count when the program schema stays fixed;
6. model difficulty.

Difficulty is not taxonomy. A hard and easy branch can stay one task if the
scene and objective contracts are genuinely the same.

## 7) Random Sampling Semantics
Every semantic random choice must be represented as an explicit finite support
or bounded range plus optional weights, then sampled with a seeded RNG draw.
Uniform sampling is the default and means every support item has weight `1`.
Do not implement semantic sampling by selecting `seed % n`,
`hash(seed, namespace) % n`, a cursor modulo, or any other index-cycling scheme.

This applies to task-internal semantic axes such as:

1. target answer class;
2. query branch;
3. construction family;
4. target object/attribute/role;
5. answer support value when it is meant to be randomly sampled.

Use shared sampling primitives rather than inventing task-local conventions.
For weighted axes, build the support/weight map and use
`weighted_support_choice`. For unweighted finite supports, use `uniform_choice`
or `uniform_choice_with_probabilities`. For contiguous integer supports, use
`integer_range_choice`. All of these take the same shared RNG-based path and
produce explicit probability metadata where applicable.

Index modulo is allowed only for non-random roles where deterministic
enumeration is explicitly intended, such as assigning cyclic visual labels,
laying out repeated colors, or choosing a fixed validation sweep. Such use must not
be described as random, uniform, equal-weight, or representative sampling.

If a build needs exact stratification, put that stratification in the build
sampler and record it as a sampling policy. Do not hide stratification inside
an individual task generator.

Global build allocation remains task-level. Build presets must not derive task
weights from the number of internal `query_id` branches.

## 8) Annotation Boundary
Annotation should mark minimal visual answer-verification witnesses for the
task family. It is not the complete reasoning proof. Direct selection/counting
tasks usually annotate the selected or counted answer objects. Derived
chart/graph/measurement/arithmetic tasks annotate the minimal visible operands
needed to verify the computed answer. Geometry and physics diagram tasks
annotate canonical visual primitives such as points, segments, rays, intervals,
axes, or marked regions. Do not include reference objects, scope panels,
distractors, legends, intermediate values, or derivation scaffolding unless
those elements are part of the task's answer-verification witness.

Use trace metadata for the full derivation path: prompt operands, reference
objects, scope regions, candidate lists, ranked candidates, formula inputs,
debug maps, and other solver/audit context.

Prefer map annotation types when roles matter or an unordered set would be
ambiguous. Use unordered sets for homogeneous counting witnesses where
cardinality is the answer or role identity does not matter.

For scoped selection tasks, the scope is usually a prompt operand, not an
annotation witness. For example, if the prompt asks which card in a named
section has a requested rank, annotate the selected card; keep the section name
and section bbox in trace metadata unless the section itself is needed to
verify the answer contract.

For `point_map` and `bbox_map`, the required key literals may be
instance-bound visible labels as long as the role family remains stable. Split
only when the semantic witness role changes, such as requested endpoints vs all
construction points, point witnesses vs region bboxes, or two required points vs
a variable-size witness set.

Answer and annotation must come from the same execution trace.

## 9) Current Surface Policy
Retired public ids must be deleted, not kept as compatibility tasks, disabled
registry entries, alias modules, redirect docs, config stubs, prompt branches,
absence-only tests, or stale artifacts. After a task id is renamed, split,
merged, or retired, active code, docs, configs, prompts, and tests should
describe only the current task surface.

Active scenes should not use source-routing metadata, config query weights,
retired scalar difficulty gates, or operational artifacts as source contracts.
Use `docs/contracts/SOURCE_LAYOUT.md` for source ownership rules.

## 10) Task-Boundary Decisions
When evaluating task boundaries, assign one concrete outcome:

- `Keep`: current task already matches one stable scene/objective contract.
- `Broaden`: task is valid but needs more in-contract scene/query variety.
- `Merge`: tasks share the same scene contract, answer schema, annotation
  schema, concrete program schema, and query-facing scaffold.
- `Split`: one task contains branches with different answer schemas,
  annotation schemas, program schemas, witness roles, or visible scaffolds.
- `Retire`: task is redundant, degenerate, not naturally grounded, or not worth
  maintaining as a Trace sampling unit.
- `Blocked Needs Inspection`: code, prompts, or docs disagree
  enough that the contract cannot be classified.

Do not keep a vague "related task" bucket for merely related tasks. If a
required merge condition fails, the decision is `Keep` unless the task itself
has a concrete split, broaden, retire, or blocked reason.

## 11) Versioning And Refinement
The current public inventory is a stable versioned surface. A boundary change
requires an explicit taxonomy version update with synchronized code, docs,
tests, and release notes.
