# Certificate adapter interface (v1)

This is the contract between the shared certificate core (`types.py`, `parsing.py`,
`geometry.py`, `identity.py`, `checks.py`, `registry.py`, `scoring.py`) and the four task
adapters in `adapters/`. The design document (DESIGN sections 2-5) is authoritative for the
semantics of each adapter; this file says how an adapter plugs in.

## 1. Module convention

- One module per adapter: `adapters/selected_set.py`, `adapters/filtered_mean.py`,
  `adapters/weighted_sum.py`, `adapters/path.py`.
- Each module defines one subclass of `adapters.base.Adapter` and a factory:

  ```python
  def create_adapter() -> Adapter: ...
  ```

- `registry.py` imports the module lazily, calls `create_adapter()` once, runs
  `type(adapter).validate_definition()` and caches the instance. `registry.TASK_ADAPTERS`
  maps task ids to module names; the adapter itself decides query ids and instance versions
  through `supported` and `instance_versions`.
- `adapter_id` must start with the module name and a dot, for example
  `weighted_sum.lever_side_torque`. `registry.adapter_by_id` relies on this
  (`validate_definition` enforces it; test-only adapters set `_allow_foreign_module = True`
  and are registered with `registry.register_adapter`).
- Adapters must not import `scoring.py`, `export.py`, `prompts.py`, `evaluate.py` or
  `reward.py`. The core never imports an adapter except through the registry.

## 2. Class attributes (all required unless a default is shown)

| attribute | type | meaning |
|---|---|---|
| `adapter_id` | `str` | e.g. `"filtered_mean.table_filtered_column_mean"` (DESIGN 3.x headings) |
| `kind` | `str` | certificate `kind` value, e.g. `"filtered_mean"` |
| `version` | `str` | adapter version (enters `certificate_extension_id`), e.g. `"1.0.0"` |
| `schema_id` | `str` | e.g. `"cert.filtered_mean.v1"` (enters `certificate_extension_id`) |
| `verifier_version` | `str` | bump when check semantics change |
| `supported` | `Mapping[str, tuple[str, ...]]` | task id -> supported public query ids |
| `instance_versions` | `tuple[str, ...]` | default `("v0",)` |
| `annotation_type` | `str` | the task's legacy annotation type (`bbox_set`, `bbox_set_map`, `bbox_map`, ...) |
| `annotation_roles` | `tuple[str, ...]` | roles that `annotation_index` binds into; default `("",)` (plain list). Table: `("filter_cells", "target_cells")`. Path: `()` (no index binding) |
| `list_key` | `str \| None` | `"members"`, `"rows"`, `"terms"`, `"steps"` |
| `list_cap` | `int` | 64 / 16 / 8 / 32 (DESIGN 2.1) |
| `cert_fields` | `Mapping[str, FieldSpec]` | certificate-level fields other than `kind` and `list_key` |
| `element_fields` | `Mapping[str, FieldSpec]` | exact keys of each list element |
| `m_families` | `tuple[str, ...]` | the **fixed** list of truth families averaged into M, e.g. `("ws.terms.truth", "ws.weight.truth", "ws.distance.truth", "ws.product.truth")` |
| `family_stages` | `Mapping[str, str]` | each M family -> `"grounding"`, `"operand"` or `"computation"` (error classes) |
| `local_check_ids` | `tuple[str, ...]` | the local check ids the adapter emits (documentation and summaries) |
| `train_cells` | `tuple[str, ...]` | DESIGN 6 train/E1 cells (values of `split_cell`); default `()` (scoring-only adapters) |
| `heldout_cells` | `Mapping[str, str]` | DESIGN 6 E2 cells -> `"render"`, `"program"` or `"both"`; disjoint from `train_cells`; default `{}` |

`FieldSpec` values come from `parsing.py`: `INT` (JSON integer; bool rejected; integral
float accepted by value but F=0), `INDEX` (an `INT` range-checked to `[0, n_annotation)`;
never negative indexing), `POINT` (`[x, y]`, two finite non-bool numbers, any precision),
`BOOL` (`true`/`false` only), `enum("white", "black")` (exact, case-sensitive), and
`const(value)` (exactly this JSON value; `const(True)` does not accept `1`).

Stage assignment used by the core error classes (DESIGN 5 `error_class`):
selection/binding families are `grounding` (`ss.set`, `fm.rows`, `ws.terms`,
`pa.endpoints`, `pa.legal`), operand reads are `operand` (`ss.attr`, `fm.filter_value`,
`fm.target_value`, `ws.weight`, `ws.distance`, `pa.cost`), gated aggregates are
`computation` (`ss.count`, `fm.count`, `fm.sum`, `fm.mean`, `ws.product`, `pa.total`,
`pa.optimal`).

## 3. Methods

All signatures are exact. `Mapping`/`Dict` values must be JSON-safe (str keys, finite
numbers, no tuples where a list round-trip would change equality the adapter relies on).

### 3.1 Identity (concrete in the base class)

```python
def spec(self) -> AdapterSpec
def identity(self) -> dict[str, str]          # {"id", "version", "schema_id", "verifier_version"}
def supports(self, task: str, query_id: str, instance_version: str) -> bool
def public_contract(self, image_size: Sequence[int], coord_frame: str) -> PublicContract
@classmethod
def validate_definition(cls) -> None            # raises TypeError on an inconsistent class
```

### 3.2 Export side (abstract; runs on generated records)

```python
def build_oracle(self, record: TraceRecord) -> dict[str, Any]
```
Build the private oracle of DESIGN 3.x from `record.train` (TrainInstance dict) and
`record.trace` (TraceInstance sidecar dict: `scene_ir`, `render_map`, `execution_trace`,
...). `record.source_size()` gives `(W, H)`. Geometry is source pixels, unrounded.
Check every export invariant listed in DESIGN 3.x and raise
`QuarantineError(reason, message)` on the first failure (`reason` is a short stable code
such as `"sum_not_divisible"`, `"weight_specs_misaligned"`). Never return a partial oracle,
never read `answer_gt` to *derive* oracle values (only to cross-check them).

```python
def split_cell(self, record: TraceRecord) -> str
```
Held-out cell label for the split builder (DESIGN 6 table), deterministic, e.g.
`"above|zebra"`, `"offset_fulcrum"`, `"landscape"`. Every value must be declared in
`train_cells` or `heldout_cells`; the concrete `holdout_type(cell) -> str` returns `""` for a
train/E1 cell and the holdout type for an E2 cell, and raises
`QuarantineError("unknown_cell")` otherwise. The exporter reads held-out cells only through
these attributes (D13), never by hard-coding DESIGN 6.

```python
def semantic_key(self, record: TraceRecord) -> str
```
Cross-split dedupe key (DESIGN 6 "Group and dedupe rules"); return a stable string (hash it
with `types.canonical_digest` if it is long).

```python
def sft_certificate(self, oracle: Mapping[str, Any], rng_seed: int) -> tuple[Any, dict, Any]
```
Return `(annotation, certificate, answer)` of the canonical SFT target: annotation in the
legacy annotation type, unrounded source pixels (the exporter rounds boxes half-up with
`geometry.round_box_half_up`); certificate dict with `"kind"` as its **first** key and the
list in the canonical order of DESIGN 3.x; answer = the oracle answer (an `int`). Any choice
among valid witnesses (path) uses `random.Random(rng_seed)` only; the exporter passes
`instance_seed`. Scoring this object with the unrounded annotation must give A=G=M=F=1.

### 3.3 Schema (concrete, override only if needed)

```python
def schema(self, cert: Any) -> CertificateSchema
```
Schema of the variant `cert` claims (`cert` is any JSON value: guard with
`isinstance(cert, dict)`). The default builds one schema from the class attributes.
Override for several variants, e.g. path returns a `{"reachable": const(False)}`
schema with `list_key=None` when `cert.get("reachable") is False`, and the step schema
(`{"reachable": const(True), "total": INT}`, `steps` with `{"at": POINT, "cost": INT}`)
otherwise. Use `variant=` on `CertificateSchema` to name it.

```python
def annotation_size(self, annotation: AnnotationView) -> int
```
Upper bound for `annotation_index`. Default: the smallest list length over
`annotation_roles` (0 when the roles are empty). Override only with a DESIGN reason.

```python
def validate_certificate(self, cert: Any, n_annotation: int) -> CertificateView
def element_schema_ok(self, cert: Any, n_annotation: int) -> list[bool]
def answer_schema_ok(self, answer: Any) -> bool       # default: type(answer) is int
```

What the core already enforces before `checks` is called (adapters must not repeat it):
- **Envelope** (M=0, every family `valid_input=False`, `checks` is *not* called): no scored
  object; strict-parse failure (NaN/Infinity, Python literals, > 16 KiB, nesting > 32);
  duplicate key at the top level, in the certificate object or anywhere in the annotation;
  certificate missing or not an object; wrong `kind`; `list_key` missing or not a list; list
  longer than `list_cap`, or an index-bound annotation role (`annotation_roles`) longer than
  `list_cap` (`error_code="budget"`). When a final object exists it is the scored object: a
  final object that fails strict parsing or lacks `certificate` is an envelope failure, never
  replaced by an earlier draft.
- **Numbers**: a JSON number that does not fit a float (e.g. a 400-digit integer) is never a
  coordinate; `read_point`, `frame.box` and `frame.point` return None for it, so the element or
  box is invalid. Adapters must never call `float()` on raw model values themselves.
- **Elements**: each `ElementView` has `valid`, `error_code` (`not_object`, `duplicate_key`,
  `missing_key`, `extra_key`, `bad_type`, `bad_value`, `index_out_of_range`) and `values`
  (typed values of the fields that parsed). An invalid element stays in `n` and must earn
  no credit; `annotation_index` is None unless the element is valid.
- **F** is computed by the core (`parsing.format_check`): adapters only declare the schema,
  plus `annotation_keys()` for map-type annotations (path: `player_cell`, `target_cell`) and
  `annotation_cap()` (default: `list_cap` when the certificate binds by index), which the core
  uses for the strict annotation structure check.

### 3.4 Verification

```python
def candidates(self, oracle: Mapping[str, Any]) -> Mapping[str, CandidateSet]
```
Per annotation role (`""` for list annotations; `"filter_cells"`, `"target_cells"`,
`"player_cell"`, `"target_cell"` for maps): **all** scene objects of the adapter's class as
source-pixel boxes, `selected` = the oracle witness positions, optional `keys`
(e.g. `(row, col)` or oid). Used by `resolve_role`, the identity log and `G_id`. Path
returns the tile grid for both endpoint roles with `selected=(start,)` / `(target,)`.

```python
def family_applicable(self, oracle: Mapping[str, Any]) -> Mapping[str, bool]
```
Oracle-only applicability (default: all true). selected_set: `ss.attr.truth` applies iff
k > 0. At least one family must stay applicable.

```python
def checks(
    self,
    cert: CertificateView,
    annotation: AnnotationView,
    oracle: Mapping[str, Any],
    ctx: PublicContext,
) -> list[CheckResult]
```
Called only for envelope-valid certificates. Must:
1. Return exactly one family-level **truth** result per `m_families` entry, with
   `check_id == family` and `applicable == family_applicable(oracle)[family]`
   (`checks.truth_check`, `checks.counted_truth`, `checks.gated_check`).
2. Return the local checks (`checks.local_check`, kind `"local"`), never counted in M.
3. Optionally return per-element detail results with a distinct `check_id`
   (e.g. `"ws.product.truth[2]"`) and `family` set to their family.
4. Resolve every annotation role with `self.resolve_role(annotation, role, oracle, ctx)`
   (boxes: D4 rule with `ctx.identity` thresholds and one-to-one dedupe; points: strict
   point-in-tile). Never bind by value or label.
5. Convert every coordinate that appears **in the certificate** (path `steps[i].at`) with
   `ctx.frame.point(raw)` / `ctx.frame.box(raw)`. Annotation geometry in `annotation` is
   already in source pixels. (The scorer re-runs `checks` with the Qwen2.5-VL shown frame
   for `M_shown`, so hard-coding px_source is a bug.)
6. Use `max(n_claimed, n_oracle)` denominators (`checks.partial_credit`/`counted_truth`);
   missing, extra or duplicate claims must never raise credit. For the selection family,
   n_claimed counts the annotation boxes too (`max(len(elements), annotation.size(role))`):
   a box no element references is still a claim (DESIGN 2.4 review note).
7. Gate dependent families with `checks.gated_check(check_id, gates, value_ok)`: 1 iff every
   gate is exactly 1 and the computation on the **model's own** values is right.
8. Be deterministic and never raise on envelope-valid input (any element may be invalid,
   any typed value may be absent: use `element.get(key)` and `cert.fields.get(key)`).

`cert.answer` is the response's top-level answer read as an int (or None); use it only for
`*.answer.local`. Local checks of the model's arithmetic use the model's values.

Binding helpers in `checks.py`:
- `bind_elements(cert.elements, resolutions) -> list[Binding]`: element -> resolved
  candidate through its `annotation_index` (status `bound`, `invalid_element`,
  `unresolved`, `ambiguous`, `duplicate`, `invalid`; `repeated_index` flags a reused index).
  For two-role binding (table) call it once per role and combine.
- `bound_exactly_once(bindings, targets) -> {target: element position}`: targets claimed by
  exactly one bound element ("resolved by exactly one valid member").
- `claims_by_candidate`, `indices_bijective(elements, n)` (the `*.ref.local` check),
  `partial_credit(num, n_claimed, n_oracle)`, `read_int`, `read_enum`, `read_point`.

```python
def diagnostics(self, cert, annotation, oracle, ctx, checks) -> dict[str, Any]
```
Optional JSON-safe extras (e.g. `ss.attr_read_acc`, operand reads on wrong-side blocks,
fraction of legal moves). Logged under `diagnostics["adapter"]`.

```python
def error_class(self, families, *, envelope_failure, semantics, answer) -> str
```
Default: first failing stage in `family_stages` order, then `answer_inconsistent`
(M == 1, A == 0), else `correct`. Override only with a DESIGN reason.

Concrete helpers: `resolve_role(annotation, role, oracle, ctx, candidates=None)` and
`identity_report(annotation, oracle, ctx) -> (IdentityReport, G_id)`. `resolve_role` caches its
result on the `AnnotationView` (keyed by role, thresholds and candidate boxes), so checks,
diagnostics and the identity report resolve each box once per frame; an adapter with a faster
equivalent resolution overrides `_resolve_role(annotation, role, candidate_set, ctx)`, never
`resolve_role`.

### 3.5 Counterexamples

```python
def corruptions(self, oracle: Mapping[str, Any], sft_obj: Mapping[str, Any], rng_seed: int) -> list[Corruption]
```
`sft_obj` is `{"annotation", "certificate", "answer"}` (unrounded source px, from
`sft_certificate`). Return deterministic (seeded) corruptions from DESIGN 9 for this
adapter. `Corruption(name, response_obj, expected, description="")`; `expected` maps a
check id or a signal (`"A"`, `"G"`, `"M"`, `"F"`) to `"pass"` (== 1), `"fail"` (< 1),
`"unchanged"` (equal within 0.01 to the uncorrupted SFT object's score; use for `"G"` in
operand-only corruptions) or a float (exact, abs tol 1e-9). Include at least: wrong value
with consistent arithmetic, wrong arithmetic with right values, omitted claim, duplicate
claim, negative index, index out of range, boxes shifted off target, plus the
adapter-specific cases listed in DESIGN 9.

### 3.6 Prompt pieces (single source: `cert_prompts_v1.json`)

```python
def prompt_block(self) -> Mapping      # concrete: kinds.<kind> of the bundle, contract-checked
def certificate_line(self) -> str     # concrete: the bundle's certificate_line (DESIGN 4)
def example_object(self) -> dict      # concrete: the bundle's fixed self-consistent example
def example_json(self) -> str         # concrete: compact JSON, keys annotation, certificate, answer
```
All three read the kind's block of `resources/prompts/experimental/certificates/cert_prompts_v1.json`
(the file `prompts.py` renders from), so the adapter and the rendered prompt can never diverge.
`prompt_block` raises `CertificateVersionError` when the block's `schema_id`, `annotation_type`,
`list_key`, `list_cap` or task set differ from the adapter. Instance-independent: no oracle
value, set size, id or witness may appear. Example coordinates must be <= 1000 so one literal
serves both frames.

## 4. Data types (where they live)

| type | module | notes |
|---|---|---|
| `TraceRecord(train, trace, image_size, image_path)` | `types` | `.task`, `.query_id`, `.instance_seed`, `.scene_ir`, `.render_map`, `.execution_trace`, `.render_spec`, `.answer_gt`, `.annotation_gt`, `.source_size()`; `read_trace_records(dataset_dir)` loads a built dataset |
| `CheckResult(check_id, family, kind, score, applicable, valid_input, error_code, detail)` | `types` | score validated finite in [0, 1] |
| `Corruption`, `AdapterSpec`, `PublicContract`, `CertificateExtension`, `ScoreResult` | `types` | |
| `QuarantineError(reason, message)`, `CertificateVersionError`, `ExtensionMismatchError` | `types` | data problems raise; never model zeros |
| `CertificateSchema`, `CertificateView`, `ElementView`, `FieldSpec`, `INT`, `INDEX`, `POINT`, `BOOL`, `enum`, `const` | `parsing` | |
| `AnnotationView(annotation_type, geometry, roles, structure_ok, raw)` | `geometry` | `items(role)` index-aligned source boxes/points, `None` for malformed entries |
| `Frame(name, scale_x, scale_y)` | `geometry` | `.box(raw)`, `.point(raw)` -> source px or None |
| `CandidateSet(boxes, selected, keys)`, `Resolution`, `IdentityParams` | `identity` | statuses `resolved`, `unresolved`, `ambiguous`, `duplicate`, `invalid` |
| `PublicContext(task, query_id, image_size, coord_frame, frame, identity)` | `adapters.base` | |

## 5. How the scorer uses an adapter

`score_certificate_response(response, base, extension, *, coord_frame, ...)`:
1. Verifies the extension (id hash, `base.instance_id`, optional `cert_extension_id`) and
   that `extension.adapter == adapter.identity()`; mismatches raise.
2. Locates the final object (legacy rule, F only) and the scored object (M and G).
3. A: legacy exact_json + bool/duplicate-answer guards. G: legacy annotation score on the
   scored object's annotation in source px (legacy tolerant extraction if none).
4. M: envelope checks -> `annotation_view` -> `adapter.validate_certificate(cert,
   adapter.annotation_size(view))` -> `adapter.checks(...)` -> mean of applicable
   `m_families`.
5. F: `parsing.format_check` (final position, strict JSON, exact top-level keys, legacy
   annotation parse, schema, caps, integers, `adapter.answer_schema_ok`).
6. Reward arms, `joint = A and F and M == 1`, `adapter.error_class(...)`, diagnostics
   (`A_legacy`, `G_id`, identity log with raw IoU/containment, `G_shown`/`M_shown`,
   `verifier_ms` logged only).

## 6. Rules

- Never `eval`/`exec` descriptive strings (`program_code`) or model output.
- Oracle data never reaches a prompt; prompt pieces are instance-independent.
- Deterministic: no wall-clock dependence, `random.Random(seed)` only.
- No absolute machine paths in any file under the repository.
- Keep `version`/`verifier_version` bumps explicit: the extension id hashes `version` and
  `schema_id`, and the scorer refuses an extension whose adapter block differs.
