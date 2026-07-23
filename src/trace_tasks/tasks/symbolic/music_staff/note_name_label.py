"""Read the name of one marked note on a music staff."""

from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task

from ._lifecycle import DOMAIN, MusicStaffRuntime, load_music_staff_defaults, run_music_staff_runtime
from .shared.components import MusicSceneSpec, StaffSystem, make_staff_note_sequence
from .shared.output import build_text_option_dataset
from .shared.rules import NOTE_NAME_OPTION_SUPPORT, format_pitch, random_staff_pitch, title_for


TASK_ID = "task_symbolic__music_staff__note_name_label"
PROMPT_QUERY_KEY = "note_name_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_KEY = "music_note_name_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_music_staff_defaults(TASK_ID)


def _sample_note_name_excerpt(rng):
    """Return a five-note excerpt and the target pitch bound to the answer."""

    target_index = rng.randrange(5)
    pitches = tuple(random_staff_pitch(rng, allow_accidental=True) for _ in range(5))
    notes = make_staff_note_sequence(
        pitches,
        item_ids=tuple(
            "target_note" if index == target_index else f"distractor_note_{index + 1}"
            for index in range(5)
        ),
        markers=tuple(str(index + 1) for index in range(5)),
        accidental_visible=True,
    )
    return target_index, pitches[target_index], notes


def _build_dataset(
    *,
    branch_key,
    instance_seed,
    scene_variant,
    params,
    gen_defaults,
):
    """Bind one sampled target note in a five-note excerpt to a text option."""

    if str(branch_key) != "note_name_label":
        raise ValueError(f"unsupported music-staff branch for note-name task: {branch_key}")

    _ = params, gen_defaults
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    target_index, target_pitch, notes = _sample_note_name_excerpt(rng)
    answer = format_pitch(target_pitch, include_octave=False)
    spec = MusicSceneSpec(
        title=title_for(scene_variant, "marked note"),
        systems=(StaffSystem(clef="treble", slot_count=7, notes=tuple(notes)),),
    )
    return build_text_option_dataset(
        rng,
        branch_key=str(branch_key),
        correct_text=str(answer),
        candidate_texts=NOTE_NAME_OPTION_SUPPORT,
        annotation_item_ids=("target_note",),
        spec=spec,
        scene_variant=str(scene_variant),
        prompt_slots={"target_marker": str(target_index + 1)},
        metadata={"target_pitch": format_pitch(target_pitch, include_octave=True), "target_marker": str(target_index + 1)},
    )


_RUNTIME = MusicStaffRuntime(TASK_ID, SUPPORTED_QUERY_IDS, PROMPT_KEY, "bbox", _build_dataset, _GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS, PROMPT_QUERY_KEY, PROMPT_QUERY_KEY)

@register_task
class SymbolicNoteNameLabelTask:
    """Read the note name of a marked staff note."""

    task_id = TASK_ID
    reasoning_operations = ('transformation',)
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
