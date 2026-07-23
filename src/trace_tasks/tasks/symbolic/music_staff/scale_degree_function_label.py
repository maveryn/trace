"""Read the scale-degree function of a marked note in a shown key."""

from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task

from ._lifecycle import DOMAIN, MusicStaffRuntime, load_music_staff_defaults, run_music_staff_runtime
from .shared.components import DEGREE_NAMES, MusicSceneSpec, StaffSystem, make_staff_note_sequence
from .shared.output import build_text_option_dataset
from .shared.rules import KEY_SIGNATURES, display_accidental_for_key, format_pitch, key_signature_text, major_scale_pitch, title_for


TASK_ID = "task_symbolic__music_staff__scale_degree_function_label"
PROMPT_QUERY_KEY = "scale_degree_function_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_KEY = "music_scale_degree_function_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_music_staff_defaults(TASK_ID)


def _sample_scale_degree_excerpt(rng, key_label: str, root, target_index: int, target_degree: int):
    """Return notes whose target binds the sampled scale-degree function."""

    pitches = []
    for note_index in range(5):
        degree = int(target_degree) if note_index == int(target_index) else rng.randrange(1, 8)
        pitches.append(major_scale_pitch(root, degree))
    notes = make_staff_note_sequence(
        tuple(pitches),
        item_ids=tuple(
            "degree_note" if index == int(target_index) else f"distractor_degree_note_{index + 1}"
            for index in range(5)
        ),
        markers=tuple(str(index + 1) for index in range(5)),
        display_accidentals=tuple(display_accidental_for_key(str(key_label), pitch) for pitch in pitches),
    )
    return pitches[int(target_index)], notes


def _scale_degree_roles():
    return ("key_signature", "degree_note")


def _scale_degree_prompt_slots(key_label, target_marker):
    return {"target_key": str(key_label), "target_marker": str(target_marker)}


def _build_dataset(
    *,
    branch_key,
    instance_seed,
    scene_variant,
    params,
    gen_defaults,
):
    """Build a key signature and marked degree note for option-letter readout."""

    if str(branch_key) != "scale_degree_function_label":
        raise ValueError(f"unsupported music-staff branch for scale-degree task: {branch_key}")
    _ = params, gen_defaults
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    visible_key_labels = tuple(key for key in KEY_SIGNATURES if key != "C major")
    key_label = rng.choice(visible_key_labels)
    root = KEY_SIGNATURES[key_label][0]
    target_index = rng.randrange(5)
    target_degree = rng.randrange(1, 8)
    target_pitch, notes = _sample_scale_degree_excerpt(rng, key_label, root, target_index, target_degree)
    spec = MusicSceneSpec(
        title=title_for(scene_variant, "scale degree"),
        systems=(
            StaffSystem(
                clef="treble",
                slot_count=7,
                key_signature=key_signature_text(key_label),
                key_signature_id="key_signature",
                notes=tuple(notes),
            ),
        ),
    )
    answer = DEGREE_NAMES[int(target_degree) - 1]
    prompt_slots = _scale_degree_prompt_slots(key_label, str(target_index + 1))
    scale_metadata = {
        "degree_1based": int(target_degree),
        "note": format_pitch(target_pitch, include_octave=True),
        "target_marker": str(target_index + 1),
    }
    return build_text_option_dataset(
        rng,
        branch_key=str(branch_key),
        correct_text=str(answer),
        candidate_texts=DEGREE_NAMES,
        annotation_item_ids=_scale_degree_roles(),
        spec=spec,
        scene_variant=str(scene_variant),
        prompt_slots=prompt_slots,
        metadata=scale_metadata,
    )


_RUNTIME = MusicStaffRuntime(TASK_ID, SUPPORTED_QUERY_IDS, PROMPT_KEY, "bbox_map", _build_dataset, _GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS, PROMPT_QUERY_KEY, PROMPT_QUERY_KEY, ("key_signature", "target_note"))

@register_task
class SymbolicScaleDegreeFunctionLabelTask:
    """Read the scale-degree function of a marked note."""

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
