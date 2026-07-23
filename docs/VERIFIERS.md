# Verifiers And Rewards

Trace verifiers score a model response against the typed ground truth and
versioned reward contract stored on each generated instance. The verifier
payload is produced from the same execution trace as the answer and annotation.

The public entry point is:

```python
from trace_tasks import score_trace_response
```

Pass `answer_gt`, `annotation_gt`, and `reward_contract` directly from the
instance being scored. Do not infer a contract from the response or hand-author
task-specific scoring branches. The examples below inline those objects only to
remain self-contained.

## Response Contract

Trace prompts request one final JSON object. The required keys depend on the
selected output mode:

| Output mode | Final object |
| --- | --- |
| `answer` (alias `answer_only`) | `{"answer": <typed value>}` |
| `answer_and_annotation` | `{"answer": <typed value>, "annotation": <typed value>}` |

For a format score of `1.0`, the response must end with that object (a final
JSON code block is also accepted), and the object must contain exactly the keys
shown above. Additional prose may precede the final object. Extra keys, missing
keys, a non-object final value, or text after the object produce a format score
of `0.0`.

Scoring and format validation are intentionally separate. The scorer can
recover an answer object from a tagged block, code block, or balanced object
elsewhere in a response. In answer mode it can also compare an atomic response
such as `7`. Recovered content can receive task reward while still receiving a
format score of zero. `format_weight` defaults to `0.0`, so format is a logged
diagnostic unless a caller explicitly includes it in `overall`.

Emit valid JSON for portable use. The current compatibility parser can also
read a Python literal mapping, but prompts, exports, and new integrations should
not rely on that behavior.

## Answer Scoring

This example is runnable from an installed checkout:

```bash
python - <<'PY'
from trace_tasks import score_trace_response

result = score_trace_response(
    response='{"answer":7}',
    answer_gt={"type": "integer", "value": 7},
    annotation_gt={"type": "bbox", "value": [10, 20, 30, 40]},
    reward_contract={
        "reward_contract_version": "v0",
        "answer": {"id": "answer_exact_match_v0", "type": "integer"},
        "annotation": {"id": "bbox_soft_iou_v0", "type": "bbox"},
    },
    trace_reward_mode="answer",
)

print({
    key: result[key]
    for key in ("overall", "answer_reward", "format", "format_schema_ok")
})
PY
```

Output:

```text
{'overall': 1.0, 'answer_reward': 1.0, 'format': 1.0, 'format_schema_ok': 1.0}
```

The annotation payload remains part of the instance contract, but answer mode
does not use it to calculate task reward. Answer scoring is exact after
canonical JSON normalization. In particular:

- strings are stripped and compared case-insensitively;
- finite integral floats and integers normalize to the same value;
- mapping keys are normalized and mapping order is ignored;
- set order is ignored;
- sequence order remains significant.

## Answer-And-Annotation Scoring

The following response has the correct answer and a predicted box whose IoU
with the target box is `1/3`:

```bash
python - <<'PY'
from trace_tasks import score_trace_response

result = score_trace_response(
    response='{"answer":2,"annotation":[5,0,15,10]}',
    answer_gt={"type": "integer", "value": 2},
    annotation_gt={"type": "bbox", "value": [0, 0, 10, 10]},
    reward_contract={
        "reward_contract_version": "v0",
        "answer": {"id": "answer_exact_match_v0", "type": "integer"},
        "annotation": {"id": "bbox_soft_iou_v0", "type": "bbox"},
    },
    trace_reward_mode="answer_and_annotation",
)

print({
    key: result[key]
    for key in (
        "overall",
        "answer_reward",
        "annotation_reward",
        "annotation_parse_ok",
    )
})
PY
```

Output:

```text
{'overall': 0.6666666666666666, 'answer_reward': 1.0, 'annotation_reward': 0.3333333333333333, 'annotation_parse_ok': 1.0}
```

Annotation parsing follows the declared `annotation_gt.type`; it does not guess
a type from the returned value. Scalar contracts reject set-shaped values,
map contracts preserve role keys, sequences preserve order, and sets use
unordered assignment. Missing or extra set items and map keys reduce the soft
score. A reward-contract id or type that does not match `annotation_gt` yields
zero annotation reward and `annotation_parse_ok == 0.0`.

## Coordinate Rules

Annotation coordinates use the pixel coordinate frame of the provided or
exported image:

- points are `[x, y]`;
- boxes are `[x0, y0, x1, y1]`;
- segments are `[[x0, y0], [x1, y1]]`;
- set, sequence, and map types compose those scalar shapes.

Do not return normalized coordinates, scene-local coordinates, grid indices,
or coordinates from a model-internal resized tensor. The scorer does not
rescale predictions. Box endpoints are reordered to left/top/right/bottom for
comparison, but zero-area boxes, booleans, non-numeric coordinates, and
non-finite coordinates fail parsing. Valid numeric coordinates are not clipped
to image bounds.

Box rewards use raw IoU. Point and segment rewards use soft pixel distance:

```text
similarity = exp(-ln(2) * (distance / half_life_px)^2)
```

Pass the source `image_size=[width, height]` when scoring point or segment
annotations. Unless `point_half_life_px` is supplied explicitly, the half-life
is `clamp(0.035 * image_diagonal, 20, 80)` pixels; missing size metadata falls
back to 32 pixels. Image size selects the distance scale and never transforms
coordinates.

See [Annotation and reward contracts](contracts/ANNOTATION_AND_REWARD_CONTRACTS.md)
for all public annotation types, matching rules, and reward ids.

## Reward Composition

`answer_weight` and `annotation_weight` default to `0.5` and are normalized by
their sum. Use non-negative weights whose sum is positive.

In `answer` mode:

```text
task_reward = answer_reward
```

In `answer_and_annotation` mode, the default `gated` formula is:

```text
task_reward = answer_reward * (answer_weight + annotation_weight * annotation_reward)
```

Here the weights mean their normalized values. A wrong answer therefore gets
zero task reward even if its annotation overlaps the target. Callers that need
independent components can select
`trace_annotation_reward_formula="additive"`:

```text
task_reward = answer_weight * answer_reward + annotation_weight * annotation_reward
```

Finally, `format_weight` blends the format score into the returned `overall`:

```text
overall = (1 - format_weight) * task_reward + format_weight * format
```

`format_weight` must be in `[0, 1]`. Invalid mode names, a non-positive total
task weight, an out-of-range format weight, or a non-positive explicit point
half-life raises `ValueError`.

## Return Value

`score_trace_response` returns a flat `dict[str, float]` suitable for training
and experiment logging. The primary fields are:

| Field | Meaning |
| --- | --- |
| `overall` | Final reward after the optional format blend. |
| `accuracy`, `answer_reward` | Exact normalized answer score, either `0.0` or `1.0`. |
| `annotation_reward` | Soft geometric annotation score in `[0, 1]`. |
| `task_reward_raw` | Answer/annotation reward before the format blend. |
| `format` | Whether the final object obeys the selected output schema. |
| `format_structure_ok`, `format_json_ok`, `format_schema_ok` | Format diagnostics. |
| `answer_parse_ok`, `annotation_parse_ok`, `json_found` | Extraction and typed-value diagnostics. |
| `zero_reward` | `1.0` when task reward is zero, independent of any format reward. |

The result also includes normalized weight values, mode indicators, a one-hot
annotation-type field, and numeric type-specific diagnostics such as assigned
item count, mean IoU, mean point similarity, and missing or extra map-key count.
Content that cannot be recovered or parsed is represented by zero rewards and
diagnostic flags. The invalid scorer options listed under reward composition
raise instead of being converted into model failures.

## Contract Source Of Truth

The builder resolves `reward_contract` in
`src/trace_tasks/core/reward_contracts.py`. Generic scoring lives in
`src/trace_tasks/core/reward_scoring.py`. The normative schema and resolver
table are documented in
[Annotation and reward contracts](contracts/ANNOTATION_AND_REWARD_CONTRACTS.md).
