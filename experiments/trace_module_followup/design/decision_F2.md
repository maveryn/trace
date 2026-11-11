# F2 decision record: one conditional extension

Written 2026-11-09, before any F2 checkpoint was built or evaluated, under design/followup_v1.md §7 (at most one F2
extension, chosen on development evidence and the F1 pattern; recipes locked before evaluation; validated on separate
data). Exploratory.

## Development evidence used

Development screen (2,910 development rows), P1, seed 42, all seven F1 cells complete. For this record only, the
score campaigns' own per-benchmark aggregates were averaged over 23 benchmarks (WeMath left out because its campaign
value uses the fixed-525 formula the study replaced; reviewed DESIGN.md A2). These are not the study's reported values.

| | B | T | V | TF | TL | VF | VL |
|---|---|---|---|---|---|---|---|
| 23-benchmark macro | 48.62 | 51.73 | 49.55 | 48.58 | 51.16 | 48.45 | 48.91 |

| Parent | S(P) − S(B) | g_F | g_L | I |
|---|---|---|---|---|
| TRACE | +3.12 | −0.04 | +2.54 | +0.62 |
| VERO (P1) | +0.93 | −0.17 | +0.30 | +0.81 |

Supporting development facts: F0 (frozen format-normalized-v2) leaves TRACE's gain over B on the development partition
essentially unchanged (+4.19 official, +4.12 normalized; B/T/V archives, 3 seeds), so TRACE's language-side gain is not
an answer-format artifact. VERO's P1 effects are small on one seed and strongly interface-dependent (on the same rows
V scores 3.0 points higher under its own P2 interface than under P1; F0 attributes most of VERO's P1 interface loss to
answer formatting).

The confirmation-partition F1 runs were already partly complete when this record was written. They were not used for
this choice. A rough campaign-aggregate look at them (the runner's summaries) was taken while checking run health, and
it shows the same direction. That look is disclosed here; nothing on the confirmation partition selects anything below.

## Choice against the triggers

* **F2a (encoder vs merger)** — trigger: a frontend transplant is helpful or unexpectedly harmful. TF and VF are within
  0.2 points of B on development. **Not triggered.**
* **F2b (cross-parent swaps)** — trigger: complementary, transferable components. Both parents' transferable part is
  on the language side and neither frontend transfers a gain, so there is no complementary pair. **Not triggered.**
* **F2c (coarse decoder reversion)** — trigger: language-only transfers dominate, and a specific decoder-level
  difference remains after formatting controls. For TRACE, TL recovers about 80% of T's development gain while TF
  recovers none, and the gain survives format normalization. **Triggered for TRACE.**
* **Stopping rule** for VERO: its effects are tiny relative to one-seed screen uncertainty and depend on the
  interface, so VERO gets no extension.

**Selected: F2c on TRACE.** It is the guide's lower-priority option, chosen because neither F2a nor F2b is triggered.

## Locked recipes (exact copies, no averaging)

Three checkpoints. Each one is full TRACE with one fixed decoder block group restored to B, and nothing else changed.
The groups are the guide's illustrative fixed partition, chosen before any F2 result exists. Each block group carries
its attention, MLP and in-block norms. The visual frontend, token embeddings, final norm and output head stay TRACE's.

| ID | Restored to B | Kept from T | Tensors restored |
|---|---|---|---|
| TR0 | decoder layers 0–8 (9 blocks) | everything else | 108 |
| TR1 | decoder layers 9–18 (10 blocks) | everything else | 120 |
| TR2 | decoder layers 19–27 (9 blocks) | everything else | 108 |

The groups differ in size (9/10/9 blocks), and that is recorded rather than corrected. Ownership comes from
`fine_parts` / `fine_partitions` in configs/followup_v1.yaml (layer ranges over the audited `language_layers` group,
with no fallback owner). The F1 partitions and recipe hashes are unchanged; the build verifies this.

## Evaluation (same stack, interface and rows as F1)

P1 only. The order is smoke → development screen (seed 42) → confirmation seeds 42, 43, 44, which is the separate
data the F2 effects are reported on. The B and T reference runs are the F1 runs: same stack and run period, same view
files. Reported for each k: the reversion effect `R_k = S(T) − S(TR_k)` (the share of TRACE's gain that needs block
group k), plus `S(TR_k) − S(B)`. All of these use official scores and format-normalized-v2, with the same paired
cluster bootstrap as F1 (10,000 replicates, seed 314159, B/T/TR0–TR2 and all seeds attached to each unit).
Exploratory: there is no confirmatory family, and Holm is applied within the three R_k only as a descriptive
safeguard. The group effects are not additive by construction, so no decomposition identity is claimed. A large R_k
does not localize a "skill"; it only says that this block group is needed for the measured gain under this
interface and scorer.

Stop after F2c whatever it shows. F3 and F4 have separate justifications (design §7).
