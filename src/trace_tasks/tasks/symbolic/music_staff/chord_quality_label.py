from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task

from ._lifecycle import DOMAIN, MusicStaffRuntime, load_music_staff_defaults, run_music_staff_runtime
from .shared.components import MusicSceneSpec, StaffSystem
from .shared.output import build_text_option_dataset
from .shared.rules import CHORD_QUALITY_INTERVALS, build_numbered_quality_chords, title_for


TASK_ID = "task_symbolic__music_staff__chord_quality_label"
PROMPT_QUERY_KEY = "chord_quality_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_KEY = "music_chord_quality_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_music_staff_defaults(TASK_ID)


def _build_dataset(
    *,
    branch_key,
    instance_seed,
    scene_variant,
    params,
    gen_defaults,
):
    if str(branch_key) != PROMPT_QUERY_KEY:
        raise ValueError(f"unsupported music-staff branch for chord-quality task: {branch_key}")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    target_index = rng.randrange(4)
    target_quality = rng.choice(tuple(CHORD_QUALITY_INTERVALS.keys()))
    return build_text_option_dataset(
        rng,
        branch_key=PROMPT_QUERY_KEY,
        correct_text=str(target_quality),
        candidate_texts=CHORD_QUALITY_INTERVALS.keys(),
        annotation_item_ids=("target_chord",),
        spec=MusicSceneSpec(
            title=title_for(scene_variant, "chord quality"),
            systems=(StaffSystem(clef="treble", slot_count=7, chords=build_numbered_quality_chords(rng, target_index=target_index, target_quality=str(target_quality))),),
        ),
        scene_variant=str(scene_variant),
        prompt_slots={"target_marker": str(target_index + 1)},
        metadata={"quality": target_quality, "target_marker": str(target_index + 1)},
    )


_RUNTIME = MusicStaffRuntime(TASK_ID, SUPPORTED_QUERY_IDS, PROMPT_KEY, "bbox", _build_dataset, _GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS, PROMPT_QUERY_KEY, PROMPT_QUERY_KEY)


@register_task
class SymbolicChordQualityLabelTask:
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
