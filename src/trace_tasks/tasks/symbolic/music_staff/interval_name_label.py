"""Read the interval name between two marked staff notes."""

from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task

from ._lifecycle import DOMAIN, MusicStaffRuntime, load_music_staff_defaults, run_music_staff_runtime
from .shared.components import MusicSceneSpec, Pitch, StaffNote, StaffRange, StaffSystem, build_interval_pitch, staff_step_for_pitch
from .shared.output import build_text_option_dataset
from .shared.rules import INTERVAL_NAME_OPTION_SUPPORT, interval_name, random_staff_pitch, title_for


TASK_ID = "task_symbolic__music_staff__interval_name_label"
PROMPT_QUERY_KEY = "interval_name_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_KEY = "music_interval_name_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_music_staff_defaults(TASK_ID)


def _sample_interval_pair(rng) -> tuple[Pitch, Pitch, str]:
    """Sample a readable staff interval that fits in the treble excerpt."""

    qualities = ("major", "minor", "perfect", "augmented", "diminished")
    for _attempt in range(100):
        lower = random_staff_pitch(rng, low_step=-1, high_step=5, allow_accidental=False)
        number = rng.choice((2, 3, 4, 5, 6, 7, 8))
        quality = rng.choice(qualities)
        upper = build_interval_pitch(lower, int(number), str(quality))
        if upper is None:
            continue
        if -2 <= staff_step_for_pitch(upper, "treble") <= 10:
            return lower, upper, interval_name(lower, upper)
    lower = Pitch("C", 4)
    upper = Pitch("E", 4)
    return lower, upper, interval_name(lower, upper)


def _build_dataset(
    *,
    branch_key,
    instance_seed,
    scene_variant,
    params,
    gen_defaults,
):
    """Build two marked notes whose interval is answered via visible options."""

    if str(branch_key) != "interval_name_label":
        raise ValueError(f"unsupported music-staff branch for interval task: {branch_key}")
    _ = params, gen_defaults
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    target_pair_index = rng.randrange(4)
    notes = []
    ranges = []
    answer = ""
    pair_slots = (0.8, 2.75, 4.7, 6.65)
    for pair_index, start_slot in enumerate(pair_slots):
        lower, upper, pair_answer = _sample_interval_pair(rng)
        first_id = f"interval_{pair_index + 1}_note_1"
        second_id = f"interval_{pair_index + 1}_note_2"
        if pair_index == target_pair_index:
            first_id = "target_interval_note_1"
            second_id = "target_interval_note_2"
            range_id = "target_interval_range"
            answer = pair_answer
        else:
            range_id = f"interval_{pair_index + 1}_range"
        notes.extend(
            (
                StaffNote(first_id, 0, float(start_slot), lower, accidental_visible=True),
                StaffNote(second_id, 0, float(start_slot + 0.58), upper, accidental_visible=True),
            )
        )
        ranges.append(
            StaffRange(
                range_id,
                0,
                float(start_slot - 0.25),
                float(start_slot + 0.85),
                label=str(pair_index + 1),
            )
        )
    spec = MusicSceneSpec(
        title=title_for(scene_variant, "marked interval"),
        systems=(StaffSystem(clef="treble", slot_count=9, notes=tuple(notes), ranges=tuple(ranges)),),
    )
    return build_text_option_dataset(
        rng,
        branch_key=str(branch_key),
        correct_text=str(answer),
        candidate_texts=INTERVAL_NAME_OPTION_SUPPORT,
        annotation_item_ids=("target_interval_range",),
        spec=spec,
        scene_variant=str(scene_variant),
        prompt_slots={"target_marker": str(target_pair_index + 1)},
        metadata={"interval_answer": str(answer), "target_marker": str(target_pair_index + 1)},
    )


_RUNTIME = MusicStaffRuntime(TASK_ID, SUPPORTED_QUERY_IDS, PROMPT_KEY, "bbox", _build_dataset, _GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS, PROMPT_QUERY_KEY, PROMPT_QUERY_KEY)

@register_task
class SymbolicIntervalNameLabelTask:
    """Read the interval name between two staff notes."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation', 'matching')
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
