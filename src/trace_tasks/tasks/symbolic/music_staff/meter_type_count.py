"""Count measures whose visible signature has a requested meter type."""

from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task

from ._lifecycle import DOMAIN, MusicStaffRuntime, load_music_staff_defaults, run_music_staff_runtime
from .shared.components import MusicSceneSpec, Pitch, StaffBarline, StaffNote, StaffRange, StaffSystem, StaffTimeSignature
from .shared.rules import COMPOUND_METER_SIGNATURES, METER_TYPE_SIGNATURE_UNITS, SIMPLE_METER_SIGNATURES, duration_partition, title_for
from .shared.sampling import resolve_count_target
from .shared.state import MusicStaffDataset


TASK_ID = "task_symbolic__music_staff__meter_type_count"
PROMPT_QUERY_KEY = "meter_type_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_KEY = "music_meter_type_count"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_music_staff_defaults(TASK_ID)


def _opposite_meter_type(target_meter_type):
    return "compound" if str(target_meter_type) == "simple" else "simple"


def _build_dataset(
    *,
    branch_key,
    instance_seed,
    scene_variant,
    params,
    gen_defaults,
):
    """Construct four measures with an exact requested meter-type count."""

    if str(branch_key) != "meter_type_count":
        raise ValueError(f"unsupported music-staff branch for meter-type task: {branch_key}")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    measure_total = 4
    target_meter_type = str(params.get("target_meter_type", rng.choice(("simple", "compound"))))
    if target_meter_type not in {"simple", "compound"}:
        raise ValueError("target_meter_type must be 'simple' or 'compound'")
    requested_meter_count, answer_support, answer_probabilities = resolve_count_target(
        params,
        gen_defaults,
        instance_seed=int(instance_seed),
        namespace=TASK_ID,
        minimum=0,
        maximum=4,
        context="meter type count",
    )
    if int(requested_meter_count) > int(measure_total):
        raise ValueError("meter type target count cannot exceed shown measure count")
    matching_measure_indices = set(rng.sample(range(measure_total), int(requested_meter_count)))
    measure_notes: list[StaffNote] = []
    time_signatures: list[StaffTimeSignature] = []
    barlines = [StaffBarline("barline_0", 0, 0.65)]
    measure_ranges: list[StaffRange] = []
    matching_range_ids: list[str] = []
    meter_records: list[dict[str, Any]] = []
    measure_slot = 0.95
    for measure_offset in range(measure_total):
        matches_requested_type = measure_offset in matching_measure_indices
        meter_type = target_meter_type if matches_requested_type else _opposite_meter_type(target_meter_type)
        signature_pool = SIMPLE_METER_SIGNATURES if meter_type == "simple" else COMPOUND_METER_SIGNATURES
        sig = rng.choice(signature_pool)
        start_slot = float(measure_slot - 0.40)
        time_signatures.append(StaffTimeSignature(f"measure_{measure_offset + 1}_time_signature", 0, float(measure_slot), str(sig)))
        measure_slot += 0.82
        for part_index, part in enumerate(duration_partition(METER_TYPE_SIGNATURE_UNITS[sig], rng), start=1):
            measure_notes.append(
                StaffNote(
                    f"measure_{measure_offset + 1}_note_{part_index}",
                    0,
                    float(measure_slot),
                    Pitch("B", 4),
                    duration_units=int(part),
                )
            )
            measure_slot += 0.46
        range_item_id = f"measure_{measure_offset + 1}"
        measure_ranges.append(StaffRange(range_item_id, 0, float(start_slot), float(measure_slot - 0.18), str(measure_offset + 1)))
        if matches_requested_type:
            matching_range_ids.append(range_item_id)
        barlines.append(StaffBarline(f"barline_{measure_offset + 1}", 0, float(measure_slot)))
        meter_records.append(
            {
                "measure_index_1based": int(measure_offset + 1),
                "time_signature": str(sig),
                "meter_type": str(meter_type),
                "matches_requested_type": bool(matches_requested_type),
            }
        )
        measure_slot += 0.30
    spec = MusicSceneSpec(
        title=title_for(scene_variant, "meter count"),
        systems=(
            StaffSystem(
                clef="treble",
                slot_count=max(9, int(measure_slot) + 1),
                notes=tuple(measure_notes),
                time_signatures=tuple(time_signatures),
                barlines=tuple(barlines),
                ranges=tuple(measure_ranges),
            ),
        ),
    )
    return MusicStaffDataset(
        branch_key=str(branch_key),
        answer_type="integer",
        answer_value=int(requested_meter_count),
        annotation_item_ids=tuple(matching_range_ids),
        spec=spec,
        scene_variant=str(scene_variant),
        prompt_slots={"target_meter_type": str(target_meter_type)},
        metadata={
            "measure_total": int(measure_total),
            "target_meter_type": str(target_meter_type),
            "target_measure_indices_1based": [int(index) + 1 for index in sorted(matching_measure_indices)],
            "target_answer_probabilities": dict(answer_probabilities),
            "measures": list(meter_records),
        },
        target_answer_support=tuple(int(value) for value in answer_support),
    )


_RUNTIME = MusicStaffRuntime(TASK_ID, SUPPORTED_QUERY_IDS, PROMPT_KEY, "bbox_set", _build_dataset, _GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS, PROMPT_QUERY_KEY, PROMPT_QUERY_KEY)

@register_task
class SymbolicMeterTypeCountTask:
    """Count measures whose visible time signature has a requested meter type."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_music_staff_runtime(
            _RUNTIME,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )
