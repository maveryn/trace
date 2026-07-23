"""Read roman-numeral harmony labels from staff notation."""

from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ...registry import register_task

from ._lifecycle import DOMAIN, MusicStaffRuntime, load_music_staff_defaults, run_music_staff_runtime
from .shared.components import MusicSceneSpec, StaffSystem
from .shared.output import build_text_option_dataset
from .shared.rules import (
    CHORD_SLOTS,
    KEY_SIGNATURES,
    QUALITY_BY_MAJOR_DEGREE,
    ROMAN_BY_DEGREE,
    key_signature_text,
    major_scale_pitch,
    marked_chord,
    normalize_chord_for_staff,
    title_for,
)


TASK_ID = "task_symbolic__music_staff__roman_numeral_label"
PROMPT_QUERY_KEY = "roman_numeral_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_KEY = "music_roman_numeral_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_music_staff_defaults(TASK_ID)


def _build_dataset(
    *,
    branch_key,
    instance_seed,
    scene_variant,
    params,
    gen_defaults,
):
    """Build a key-context chord scene asking for one roman numeral."""

    if str(branch_key) != PROMPT_QUERY_KEY:
        raise ValueError(f"unsupported music-staff branch for roman-numeral task: {branch_key}")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    target_index = rng.randrange(4)
    key_label = rng.choice(("G major", "F major"))
    key_root = KEY_SIGNATURES[key_label][0]
    target_degree = rng.randrange(1, 8)
    chords = []
    for chord_index, slot in enumerate(CHORD_SLOTS):
        degree = target_degree if chord_index == target_index else rng.randrange(1, 8)
        chord_root = major_scale_pitch(key_root, degree)
        quality = QUALITY_BY_MAJOR_DEGREE[degree - 1]
        item_id = "target_chord" if chord_index == target_index else f"distractor_chord_{chord_index + 1}"
        chords.append(marked_chord(item_id, str(chord_index + 1), float(slot), normalize_chord_for_staff(chord_root, quality)))
    spec = MusicSceneSpec(
        title=title_for(scene_variant, "roman numeral"),
        systems=(
            StaffSystem(
                clef="treble",
                slot_count=7,
                key_signature=key_signature_text(key_label),
                key_signature_id="key_signature",
                chords=tuple(chords),
            ),
        ),
    )
    answer = ROMAN_BY_DEGREE[target_degree - 1]
    return build_text_option_dataset(
        rng,
        branch_key=PROMPT_QUERY_KEY,
        correct_text=str(answer),
        candidate_texts=ROMAN_BY_DEGREE,
        annotation_item_ids=("key_signature", "target_chord"),
        spec=spec,
        scene_variant=str(scene_variant),
        prompt_slots={"target_key": str(key_label), "target_marker": str(target_index + 1)},
        metadata={"degree_1based": int(target_degree), "target_marker": str(target_index + 1)},
    )


_RUNTIME = MusicStaffRuntime(
    TASK_ID,
    SUPPORTED_QUERY_IDS,
    PROMPT_KEY,
    "bbox_map",
    _build_dataset,
    _GEN_DEFAULTS,
    _RENDER_DEFAULTS,
    _PROMPT_DEFAULTS,
    PROMPT_QUERY_KEY,
    PROMPT_QUERY_KEY,
    ("key_signature", "target_chord"),
)


@register_task
class SymbolicRomanNumeralLabelTask:
    """Read one numbered chord's roman numeral in a key."""

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
