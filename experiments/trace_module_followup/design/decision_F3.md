# F3 decision record: one norm-matched VERO checkpoint

Written 2026-11-09 22:55 MST, before the checkpoint was built or evaluated, under design/followup_v1.md §7. F3 runs
"only if update magnitude remains an explanatory question after scoring normalization and F1". This record says why
it does, and fixes the construction and evaluation before any V-NM output exists. Exploratory.

## Why update magnitude is still an open question

* The reviewed study measured global Frobenius update norms of ‖Δ_T‖ = 1.994 and ‖Δ_V‖ = 8.800 (rounded), so VERO's
  update is about 4.4 times larger.
* F1 (provisional, seeds 42 and 44, all P1 runs it used complete) places both parents' transferable gains in the
  language component, with no material interaction:
  * TRACE g_L = +3.6;
  * VERO g_L = +1.2 official, and +3.0 under the frozen format-normalized-v2 scale.
* After normalization, therefore, VERO's language-side gain is about as large as TRACE's, yet its update moves about
  4.4 times farther. Scoring normalization and F1 together leave the magnitude question open: is most of VERO's
  displacement needed for its measured result along its own update direction?
* F4 is not run. Its motivating question was whether a frontend effect changes rendering sensitivity, and the
  frontend transplants are near zero (TF, VF within about 0.5 points of B), so no concrete question for F4 arises.

## Locked construction

* `V-NM = θ_B + λ·(θ_V − θ_B)`, built as the convex interpolation `(1 − λ)·B + λ·V`.
* λ = ‖Δ_T‖_F / ‖Δ_V‖_F, recomputed in float64 while streaming every stored tensor of the pinned parents. All 729
  tensors are included; the reviewed `parent_update_statistics` routine is used. λ is not hard-coded (the published
  ≈ 0.227 is rounded).
* The builder is the reviewed streaming interpolator (`trace_merge.merge.build_merge`): float32 accumulation from the
  BF16 parents, one BF16 cast with round-to-nearest-even, B's shard layout, atomic finalize and parent hashes checked.
  It runs under a new recipe id (`VNM`) in the follow-up work root. The reviewed study's configuration and lock are
  not reused.
* After serialization, the following are recomputed and reported, and nothing is retuned after seeing accuracy:
  * the realized global and per-group update norms;
  * the residual against the target λ·‖Δ_V‖;
  * the fraction of stored values bitwise unchanged from B;
  * finite values.
* The stored tensors are BF16, so this is the geometry of the released checkpoints, not of unavailable
  optimizer/master weights.

## Locked evaluation

* Same stack, interface and scorer versions as F1, with no new token budget or prompt tuning:
  * P1: smoke → development screen (seed 42) → confirmation seeds 42, 43, 44;
  * P2: smoke and development screen (seed 42), as a diagnostic beside F1's template matrix.
* References are the F1 runs of B, T and V, which share the stack, run period and view files.
* Reported:
  * S(V-NM) − S(V), and how much of VERO's gain V-NM keeps, (S(V-NM) − S(B)) / (S(V) − S(B));
  * S(V-NM) − S(B);
  * S(T) − S(V-NM), with both checkpoints at matched realized norm.
  * All on official and format-normalized-v2 scales, with the F1 paired cluster bootstrap (10,000 replicates,
    seed 314159).
* Descriptive only; there is no confirmatory family and no coefficient search. One checkpoint is evaluated, and
  amplifying TRACE by 4.4 times is not tested.
* Interpretation limits:
  * Equal parameter distance is not equal functional change, and the two parents have different training
    histories. This is a geometric diagnostic, not "quality per parameter moved".
  * A V-NM that retains VERO's score says that much of VERO's displacement is unnecessary for this inference result
    along this interpolation path. It says nothing about the training corpus or the minimum movement needed to train
    the model.
