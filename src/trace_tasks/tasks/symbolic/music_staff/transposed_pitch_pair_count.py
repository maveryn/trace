"""Count marked note pairs matching a requested upward transposition."""

from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task

from ._lifecycle import DOMAIN, MusicStaffRuntime, load_music_staff_defaults, run_music_staff_runtime
from .shared.components import MusicSceneSpec, Pitch, StaffNote, StaffRange, StaffSystem, build_interval_pitch
from .shared.rules import format_pitch, interval_name, pitch_midi, random_staff_pitch, title_for
from .shared.sampling import resolve_count_target
from .shared.state import MusicStaffDataset


TASK_ID = "task_symbolic__music_staff__transposed_pitch_pair_count"
PROMPT_QUERY_KEY = "transposed_pitch_pair_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_KEY = "music_transposed_pitch_pair_count"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_music_staff_defaults(TASK_ID)
_PAIR_NOTE_DELTA_SLOT = 1.45
_PAIR_RANGE_PAD_START_SLOT = 0.35
_PAIR_RANGE_PAD_END_SLOT = 1.85
_PAIR_STEP_SLOT = 2.85
_PAIR_BRACKET_Y_OFFSET_PX = -18
_PAIR_LABEL_Y_OFFSET_PX = -10


def _build_dataset(
    *,
    branch_key,
    instance_seed,
    scene_variant,
    params,
    gen_defaults,
):
    """Construct four marked note pairs with an exact transposition-match count."""

    if str(branch_key) != "transposed_pitch_pair_count":
        raise ValueError(f"unsupported music-staff branch for transposition task: {branch_key}")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    exercise_count = 4
    match_count, answer_support, answer_probabilities = resolve_count_target(
        params,
        gen_defaults,
        instance_seed=int(instance_seed),
        namespace=TASK_ID,
        minimum=0,
        maximum=4,
        context="transposed-pitch pair count",
    )
    if int(match_count) > int(exercise_count):
        raise ValueError("transposed-pitch target count cannot exceed shown pair count")
    interval_number = int(rng.choice((3, 4, 5)))
    interval_quality = "perfect" if interval_number in (4, 5) else "major"
    matching_pair_indices = set(rng.sample(range(exercise_count), int(match_count)))
    transposition_notes: list[StaffNote] = []
    transposition_ranges: list[StaffRange] = []
    matched_range_ids: list[str] = []
    transposition_records: list[dict[str, Any]] = []
    cursor_slot = 0.9
    interval_label = ""
    for example_index in range(exercise_count):
        source = random_staff_pitch(rng, low_step=0, high_step=4, allow_accidental=False)
        expected = build_interval_pitch(source, interval_number, interval_quality) or Pitch("E", 4)
        interval_label = interval_name(source, expected)
        matches_interval = example_index in matching_pair_indices
        if matches_interval:
            shown = expected
        else:
            shown = Pitch(
                str(expected.letter),
                int(expected.octave),
                int(expected.accidental) + (1 if int(expected.accidental) <= 0 else -1),
            )
            if pitch_midi(shown) == pitch_midi(expected):
                shown = Pitch(str(expected.letter), int(expected.octave), int(expected.accidental) - 1)
        range_start = float(cursor_slot - _PAIR_RANGE_PAD_START_SLOT)
        transposition_notes.extend(
            [
                StaffNote(f"pair_{example_index + 1}_source", 0, float(cursor_slot), source, accidental_visible=True),
                StaffNote(f"pair_{example_index + 1}_shown", 0, float(cursor_slot + _PAIR_NOTE_DELTA_SLOT), shown, accidental_visible=True),
            ]
        )
        range_item_id = f"pair_{example_index + 1}"
        transposition_ranges.append(
            StaffRange(
                range_item_id,
                0,
                float(range_start),
                float(cursor_slot + _PAIR_RANGE_PAD_END_SLOT),
                str(example_index + 1),
                bracket_y_offset_px=_PAIR_BRACKET_Y_OFFSET_PX,
                label_y_offset_px=_PAIR_LABEL_Y_OFFSET_PX,
            )
        )
        if matches_interval:
            matched_range_ids.append(range_item_id)
        transposition_records.append(
            {
                "pair_index_1based": int(example_index + 1),
                "source_pitch": format_pitch(source, include_octave=True),
                "expected_pitch": format_pitch(expected, include_octave=True),
                "shown_pitch": format_pitch(shown, include_octave=True),
                "matches_target_interval": bool(pitch_midi(shown) == pitch_midi(expected)),
            }
        )
        cursor_slot += _PAIR_STEP_SLOT
    spec = MusicSceneSpec(
        title=title_for(scene_variant, "transposition count"),
        systems=(StaffSystem(clef="treble", slot_count=max(8, int(cursor_slot) + 1), notes=tuple(transposition_notes), ranges=tuple(transposition_ranges)),),
    )
    return MusicStaffDataset(
        branch_key=str(branch_key),
        answer_type="integer",
        answer_value=int(match_count),
        annotation_item_ids=tuple(matched_range_ids),
        spec=spec,
        scene_variant=str(scene_variant),
        prompt_slots={"interval_name": str(interval_label)},
        metadata={
            "exercise_count": int(exercise_count),
            "target_pair_indices_1based": [int(index) + 1 for index in sorted(matching_pair_indices)],
            "target_answer_probabilities": dict(answer_probabilities),
            "pairs": list(transposition_records),
        },
        target_answer_support=tuple(int(value) for value in answer_support),
    )


_RUNTIME = MusicStaffRuntime(TASK_ID, SUPPORTED_QUERY_IDS, PROMPT_KEY, "bbox_set", _build_dataset, _GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS, PROMPT_QUERY_KEY, PROMPT_QUERY_KEY)

@register_task
class SymbolicTransposedPitchPairCountTask:
    """Count marked note pairs that match the requested transposition."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'transformation')
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
