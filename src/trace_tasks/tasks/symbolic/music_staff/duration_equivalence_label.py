"""Read the duration name of a marked note."""

from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task

from ._lifecycle import DOMAIN, MusicStaffRuntime, load_music_staff_defaults, run_music_staff_runtime
from .shared.components import MusicSceneSpec, Pitch, StaffSystem, make_staff_note_sequence
from .shared.output import build_text_option_dataset
from .shared.rules import DURATION_UNITS, duration_name, title_for


TASK_ID = "task_symbolic__music_staff__duration_equivalence_label"
PROMPT_QUERY_KEY = "duration_equivalence_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_KEY = "music_duration_equivalence_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_music_staff_defaults(TASK_ID)


def _build_dataset(
    *,
    branch_key,
    instance_seed,
    scene_variant,
    params,
    gen_defaults,
):
    """Construct one target duration and visible text duration-name options."""

    if str(branch_key) != "duration_equivalence_label":
        raise ValueError(f"unsupported music-staff branch for duration-equivalence task: {branch_key}")
    _ = params, gen_defaults
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    duration_support = (1, 2, 3, 4, 6, 8)
    target_index = rng.randrange(5)
    target_units = rng.choice(duration_support)
    target_duration_name = duration_name(target_units)
    pitch_support = (Pitch("A", 4), Pitch("B", 4), Pitch("G", 4), Pitch("C", 5), Pitch("F", 4))
    note_units = tuple(target_units if index == target_index else rng.choice(duration_support) for index in range(5))
    notes = make_staff_note_sequence(
        pitch_support,
        item_ids=tuple("target_duration" if index == target_index else f"distractor_duration_{index + 1}" for index in range(5)),
        markers=tuple(str(index + 1) for index in range(5)),
        duration_units=note_units,
        infer_duration_shape=True,
    )
    spec = MusicSceneSpec(
        title=title_for(scene_variant, "duration reading"),
        systems=(StaffSystem(clef="treble", slot_count=7, notes=tuple(notes)),),
    )
    return build_text_option_dataset(
        rng,
        branch_key=str(branch_key),
        correct_text=str(target_duration_name),
        candidate_texts=tuple(DURATION_UNITS.keys()),
        annotation_item_ids=("target_duration",),
        spec=spec,
        scene_variant=str(scene_variant),
        prompt_slots={"target_marker": str(target_index + 1)},
        metadata={
            "target_duration_units": int(target_units),
            "target_duration_name": str(target_duration_name),
            "target_marker": str(target_index + 1),
        },
    )


_RUNTIME = MusicStaffRuntime(TASK_ID, SUPPORTED_QUERY_IDS, PROMPT_KEY, "bbox", _build_dataset, _GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS, PROMPT_QUERY_KEY, PROMPT_QUERY_KEY)

@register_task
class SymbolicDurationEquivalenceLabelTask:
    """Read the duration name of a marked note."""

    task_id = TASK_ID
    reasoning_operations = ('matching',)
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
