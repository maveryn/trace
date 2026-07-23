"""Count scale fragments that correctly fit the requested key."""

from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task

from ._lifecycle import DOMAIN, MusicStaffRuntime, load_music_staff_defaults, run_music_staff_runtime
from .shared.components import MusicSceneSpec, Pitch, StaffNote, StaffRange, StaffSystem
from .shared.rules import KEY_SIGNATURES, display_accidental_for_key, format_pitch, key_signature_text, major_scale_pitch, title_for
from .shared.sampling import resolve_count_target
from .shared.state import MusicStaffDataset


TASK_ID = "task_symbolic__music_staff__scale_validation_count"
PROMPT_QUERY_KEY = "scale_validation_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_KEY = "music_scale_validation_count"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_music_staff_defaults(TASK_ID)
_FRAGMENT_NOTE_STEP_SLOT = 0.52
_FRAGMENT_GAP_SLOT = 0.64


def _scale_fragment_degrees():
    return tuple(range(1, 6))


def _scale_fragment_title(scene_variant):
    return title_for(scene_variant, "scale validation count")


def _build_dataset(
    *,
    branch_key,
    instance_seed,
    scene_variant,
    params,
    gen_defaults,
):
    """Construct four scale fragments with an exact count of valid fragments."""

    if str(branch_key) != "scale_validation_count":
        raise ValueError(f"unsupported music-staff branch for scale-validation task: {branch_key}")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    visible_key_labels = tuple(key for key in KEY_SIGNATURES if key != "C major")
    key_label = rng.choice(visible_key_labels)
    root = KEY_SIGNATURES[key_label][0]
    scale_fragment_total = 4
    valid_fragment_count, answer_support, answer_probabilities = resolve_count_target(
        params,
        gen_defaults,
        instance_seed=int(instance_seed),
        namespace=TASK_ID,
        minimum=0,
        maximum=4,
        context="scale validation count",
    )
    if int(valid_fragment_count) > int(scale_fragment_total):
        raise ValueError("scale-validation target count cannot exceed shown fragment count")
    valid_fragment_indices = set(rng.sample(range(scale_fragment_total), int(valid_fragment_count)))
    fragment_notes: list[StaffNote] = []
    fragment_ranges: list[StaffRange] = []
    valid_range_ids: list[str] = []
    scale_fragment_records: list[dict[str, Any]] = []
    staff_slot = 0.85
    for fragment_offset in range(scale_fragment_total):
        fits_key = fragment_offset in valid_fragment_indices
        range_start = float(staff_slot - 0.22)
        degrees_in_fragment = _scale_fragment_degrees()
        altered_scale_degree = rng.choice(degrees_in_fragment)
        note_item_ids = []
        note_pitches = []
        display_accidentals = []
        for degree in degrees_in_fragment:
            pitch = major_scale_pitch(root, int(degree))
            if (not fits_key) and int(degree) == int(altered_scale_degree):
                pitch = Pitch(str(pitch.letter), int(pitch.octave), int(pitch.accidental) + (1 if pitch.accidental <= 0 else -1))
            note_id = f"fragment_{fragment_offset + 1}_note_{degree}"
            display_accidental = display_accidental_for_key(str(key_label), pitch)
            note_item_ids.append(note_id)
            note_pitches.append(format_pitch(pitch, include_octave=True))
            display_accidentals.append(str(display_accidental))
            fragment_notes.append(
                StaffNote(
                    note_id,
                    0,
                    float(staff_slot),
                    pitch,
                    display_accidental=str(display_accidental),
                )
            )
            staff_slot += _FRAGMENT_NOTE_STEP_SLOT
        range_item_id = f"fragment_{fragment_offset + 1}"
        fragment_ranges.append(StaffRange(range_item_id, 0, float(range_start), float(staff_slot - 0.12), str(fragment_offset + 1)))
        if fits_key:
            valid_range_ids.append(range_item_id)
        scale_fragment_records.append(
            {
                "fragment_index_1based": int(fragment_offset + 1),
                "is_correct_scale_fragment": bool(fits_key),
                "degrees": [int(value) for value in degrees_in_fragment],
                "altered_scale_degree": None if fits_key else int(altered_scale_degree),
                "note_ids": list(note_item_ids),
                "pitches": list(note_pitches),
                "display_accidentals": list(display_accidentals),
            }
        )
        staff_slot += _FRAGMENT_GAP_SLOT
    spec = MusicSceneSpec(
        title=_scale_fragment_title(scene_variant),
        systems=(
            StaffSystem(
                clef="treble",
                slot_count=max(8, int(staff_slot) + 1),
                key_signature=key_signature_text(key_label),
                key_signature_id="key_signature",
                notes=tuple(fragment_notes),
                ranges=tuple(fragment_ranges),
            ),
        ),
    )
    return MusicStaffDataset(
        branch_key=str(branch_key),
        answer_type="integer",
        answer_value=int(valid_fragment_count),
        annotation_item_ids=tuple(valid_range_ids),
        spec=spec,
        scene_variant=str(scene_variant),
        prompt_slots={"target_key": str(key_label)},
        metadata={
            "scale_fragment_total": int(scale_fragment_total),
            "target_fragment_indices_1based": [int(index) + 1 for index in sorted(valid_fragment_indices)],
            "target_answer_probabilities": dict(answer_probabilities),
            "fragments": list(scale_fragment_records),
        },
        target_answer_support=tuple(int(value) for value in answer_support),
    )


_RUNTIME = MusicStaffRuntime(TASK_ID, SUPPORTED_QUERY_IDS, PROMPT_KEY, "bbox_set", _build_dataset, _GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS, PROMPT_QUERY_KEY, PROMPT_QUERY_KEY)

@register_task
class SymbolicScaleValidationCountTask:
    """Count scale fragments that correctly fit the requested key."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'matching')
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
