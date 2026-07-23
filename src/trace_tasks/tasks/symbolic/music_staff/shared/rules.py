"""Music-theory helpers for symbolic music-staff tasks."""

from __future__ import annotations

from typing import Dict, Tuple

from .components import (
    DEGREE_NAMES,
    LETTERS,
    Pitch,
    StaffChord,
    build_interval_pitch,
    format_key_signature,
    format_pitch,
    interval_name,
    major_scale_pitch,
    pitch_from_staff_step,
    pitch_midi,
    staff_step_for_pitch,
)


KEY_SIGNATURES: Dict[str, Tuple[Pitch, Tuple[str, ...]]] = {
    "C major": (Pitch("C", 4, 0), ()),
    "G major": (Pitch("G", 4, 0), ("#",)),
    "D major": (Pitch("D", 4, 0), ("#", "#")),
    "F major": (Pitch("F", 4, 0), ("b",)),
    "Bb major": (Pitch("B", 3, -1), ("b", "b")),
    "Eb major": (Pitch("E", 4, -1), ("b", "b", "b")),
}
_SHARP_KEY_ORDER: Tuple[str, ...] = ("F", "C", "G", "D", "A", "E", "B")
_FLAT_KEY_ORDER: Tuple[str, ...] = ("B", "E", "A", "D", "G", "C", "F")

DURATION_UNITS: Dict[str, int] = {
    "eighth note": 1,
    "quarter note": 2,
    "dotted quarter note": 3,
    "half note": 4,
    "dotted half note": 6,
    "whole note": 8,
}
TIME_SIGNATURE_UNITS: Dict[str, int] = {
    "2/4": 4,
    "3/4": 6,
    "4/4": 8,
}
SIMPLE_METER_SIGNATURES: Tuple[str, ...] = ("2/4", "3/4", "4/4")
COMPOUND_METER_SIGNATURES: Tuple[str, ...] = ("6/8", "9/8")
METER_TYPE_SIGNATURE_UNITS: Dict[str, int] = {
    **TIME_SIGNATURE_UNITS,
    "6/8": 6,
    "9/8": 9,
}

CHORD_QUALITY_INTERVALS: Dict[str, Tuple[int, ...]] = {
    "major triad": (0, 4, 7),
    "minor triad": (0, 3, 7),
    "diminished triad": (0, 3, 6),
    "augmented triad": (0, 4, 8),
    "dominant seventh": (0, 4, 7, 10),
    "major seventh": (0, 4, 7, 11),
    "minor seventh": (0, 3, 7, 10),
    "half-diminished seventh": (0, 3, 6, 10),
}
CHORD_QUALITY_DEGREES: Dict[str, Tuple[int, ...]] = {
    "major triad": (1, 3, 5),
    "minor triad": (1, 3, 5),
    "diminished triad": (1, 3, 5),
    "augmented triad": (1, 3, 5),
    "dominant seventh": (1, 3, 5, 7),
    "major seventh": (1, 3, 5, 7),
    "minor seventh": (1, 3, 5, 7),
    "half-diminished seventh": (1, 3, 5, 7),
}
ROMAN_BY_DEGREE: Tuple[str, ...] = ("I", "ii", "iii", "IV", "V", "vi", "vii diminished")
QUALITY_BY_MAJOR_DEGREE: Tuple[str, ...] = (
    "major triad",
    "minor triad",
    "minor triad",
    "major triad",
    "major triad",
    "minor triad",
    "diminished triad",
)
INVERSION_NAMES: Tuple[str, ...] = ("root position", "first inversion", "second inversion", "third inversion")
ARTICULATION_SYMBOLS: Tuple[str, ...] = ("staccato", "tenuto", "accent", "fermata")
CHORD_SLOTS: Tuple[float, ...] = (0.8, 3.0, 5.2, 7.4)
ROOT_SUPPORT: Tuple[Pitch, ...] = (
    Pitch("C", 4),
    Pitch("D", 4),
    Pitch("E", 4),
    Pitch("F", 4),
    Pitch("G", 4),
    Pitch("A", 4),
)
NOTE_NAME_OPTION_SUPPORT: Tuple[str, ...] = (
    "C",
    "D",
    "E",
    "F",
    "G",
    "A",
    "B",
    "C#",
    "D#",
    "F#",
    "G#",
    "A#",
    "Db",
    "Eb",
    "Gb",
    "Ab",
    "Bb",
)
INTERVAL_NAME_OPTION_SUPPORT: Tuple[str, ...] = (
    "minor 2nd",
    "major 2nd",
    "augmented 2nd",
    "diminished 3rd",
    "minor 3rd",
    "major 3rd",
    "augmented 3rd",
    "diminished 4th",
    "perfect 4th",
    "augmented 4th",
    "diminished 5th",
    "perfect 5th",
    "augmented 5th",
    "diminished 6th",
    "minor 6th",
    "major 6th",
    "augmented 6th",
    "diminished 7th",
    "minor 7th",
    "major 7th",
    "diminished octave",
    "perfect octave",
    "augmented octave",
)


def title_for(scene_variant: str, text: str) -> str:
    prefix = {
        "engraved_sheet": "Staff Notation",
        "exam_scan": "Music Theory Item",
        "notebook_staff": "Notation Notes",
    }.get(str(scene_variant), "Staff Notation")
    return f"{prefix}: {text}"


def key_signature_text(key_label: str) -> str:
    return format_key_signature(KEY_SIGNATURES[str(key_label)][1]) or "natural"


def marked_chord(item_id: str, marker: str, slot: float, pitches) -> StaffChord:
    """Create one numbered chord with accidentals visible when needed."""

    return StaffChord(str(item_id), 0, float(slot), tuple(pitches), marker=str(marker), accidental_visible=True)


def inversion_quality_support(inversion_index: int) -> Tuple[str, ...]:
    """Return chord qualities that can render the requested inversion clearly."""

    if int(inversion_index) == len(INVERSION_NAMES) - 1:
        return ("dominant seventh", "major seventh")
    return ("major triad", "minor triad", "dominant seventh", "major seventh")


def build_numbered_quality_chords(rng, *, target_index: int, target_quality: str) -> Tuple[StaffChord, ...]:
    """Build four numbered chords with one chord fixed to the requested quality."""

    qualities = tuple(CHORD_QUALITY_INTERVALS.keys())
    return tuple(
        marked_chord(
            "target_chord" if chord_index == int(target_index) else f"distractor_chord_{chord_index + 1}",
            str(chord_index + 1),
            float(slot),
            normalize_chord_for_staff(
                rng.choice(ROOT_SUPPORT),
                str(target_quality) if chord_index == int(target_index) else rng.choice(qualities),
            ),
        )
        for chord_index, slot in enumerate(CHORD_SLOTS)
    )


def build_numbered_inversion_chords(
    rng,
    *,
    target_index: int,
    target_quality: str,
    target_inversion: int,
) -> Tuple[StaffChord, ...]:
    """Build four numbered chords with one chord fixed to the requested inversion."""

    chords = []
    for chord_index, slot in enumerate(CHORD_SLOTS):
        if chord_index == int(target_index):
            item_id = "target_chord"
            chord_quality = str(target_quality)
            chord_inversion = int(target_inversion)
        else:
            item_id = f"distractor_chord_{chord_index + 1}"
            chord_inversion = rng.randrange(len(INVERSION_NAMES))
            chord_quality = rng.choice(inversion_quality_support(chord_inversion))
        chords.append(
            marked_chord(
                item_id,
                str(chord_index + 1),
                float(slot),
                normalize_inverted_chord_for_staff(rng.choice(ROOT_SUPPORT), chord_quality, chord_inversion),
            )
        )
    return tuple(chords)


def key_signature_accidentals_by_letter(key_label: str) -> Dict[str, int]:
    tokens = tuple(str(token) for token in KEY_SIGNATURES[str(key_label)][1])
    if not tokens:
        return {}
    if all(token == "#" for token in tokens):
        return {str(letter): 1 for letter in _SHARP_KEY_ORDER[: len(tokens)]}
    if all(token == "b" for token in tokens):
        return {str(letter): -1 for letter in _FLAT_KEY_ORDER[: len(tokens)]}
    return {}


def display_accidental_for_key(key_label: str, pitch: Pitch) -> str:
    key_accidental = int(key_signature_accidentals_by_letter(str(key_label)).get(str(pitch.letter), 0))
    pitch_accidental = int(pitch.accidental)
    if pitch_accidental == key_accidental:
        return ""
    if pitch_accidental == 0:
        return "natural"
    return "sharp" if pitch_accidental > 0 else "flat"


def random_staff_pitch(
    rng,
    *,
    clef: str = "treble",
    low_step: int = -1,
    high_step: int = 9,
    allow_accidental: bool = True,
) -> Pitch:
    step = rng.randrange(int(low_step), int(high_step) + 1)
    accidental = rng.choice([-1, 0, 1]) if allow_accidental and rng.random() < 0.28 else 0
    return pitch_from_staff_step(str(clef), int(step), accidental=int(accidental))


def build_chord(root: Pitch, quality: str, *, octave_shift: int = 0) -> Tuple[Pitch, ...]:
    intervals = CHORD_QUALITY_INTERVALS[str(quality)]
    degrees = CHORD_QUALITY_DEGREES[str(quality)]
    pitches = []
    for semitone, degree in zip(intervals, degrees):
        target_index = int(root.octave + int(octave_shift)) * 7 + LETTERS.index(str(root.letter)) + int(degree) - 1
        natural = Pitch(str(LETTERS[target_index % 7]), int(target_index // 7), 0)
        wanted = pitch_midi(root) + int(semitone)
        accidental = int(wanted - pitch_midi(natural))
        if accidental not in (-1, 0, 1):
            accidental = 0
        pitches.append(Pitch(str(natural.letter), int(natural.octave), int(accidental)))
    return tuple(pitches)


def _chord_staff_fit_score(pitches: Tuple[Pitch, ...], *, clef: str = "treble") -> tuple[int, float]:
    steps = [staff_step_for_pitch(pitch, str(clef)) for pitch in pitches]
    out_of_staff = sum(max(0, -step) + max(0, step - 8) for step in steps)
    center = (min(steps) + max(steps)) / 2.0
    return int(out_of_staff), abs(float(center) - 4.0)


def normalize_chord_for_staff(root: Pitch, quality: str) -> Tuple[Pitch, ...]:
    candidates: list[Tuple[tuple[int, float], Tuple[Pitch, ...]]] = []
    for octave in (4, 3, 5):
        candidate_root = Pitch(str(root.letter), int(octave), int(root.accidental))
        pitches = build_chord(candidate_root, str(quality))
        steps = [p.octave * 7 + LETTERS.index(p.letter) for p in pitches]
        if max(steps) - min(steps) <= 6:
            candidates.append((_chord_staff_fit_score(pitches), pitches))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return build_chord(Pitch(str(root.letter), 4, int(root.accidental)), str(quality))


def invert_chord(pitches: Tuple[Pitch, ...], inversion_index: int) -> Tuple[Pitch, ...]:
    ordered = list(sorted(pitches, key=pitch_midi))
    for _ in range(int(inversion_index)):
        p = ordered.pop(0)
        ordered.append(Pitch(str(p.letter), int(p.octave) + 1, int(p.accidental)))
        ordered = list(sorted(ordered, key=pitch_midi))
    return tuple(ordered)


def normalize_inverted_chord_for_staff(root: Pitch, quality: str, inversion_index: int) -> Tuple[Pitch, ...]:
    candidates: list[Tuple[tuple[int, float], Tuple[Pitch, ...]]] = []
    for octave in (4, 3, 5):
        candidate_root = Pitch(str(root.letter), int(octave), int(root.accidental))
        candidate = invert_chord(build_chord(candidate_root, str(quality)), int(inversion_index))
        steps = [p.octave * 7 + LETTERS.index(p.letter) for p in candidate]
        if max(steps) - min(steps) <= 7:
            candidates.append((_chord_staff_fit_score(candidate), candidate))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return invert_chord(normalize_chord_for_staff(root, str(quality)), int(inversion_index))


def duration_partition(total_units: int, rng) -> Tuple[int, ...]:
    remaining = int(total_units)
    parts = []
    choices = (1, 2, 3, 4)
    while remaining > 0:
        valid = [value for value in choices if value <= remaining]
        if remaining in (1, 2, 3, 4):
            value = remaining if rng.random() < 0.55 else rng.choice(valid)
        else:
            value = rng.choice(valid)
        parts.append(int(value))
        remaining -= int(value)
    return tuple(parts)


def duration_name(units: int) -> str:
    for name, value in DURATION_UNITS.items():
        if int(value) == int(units):
            return str(name)
    return f"{int(units)} units"


__all__ = [
    "ARTICULATION_SYMBOLS",
    "CHORD_QUALITY_INTERVALS",
    "COMPOUND_METER_SIGNATURES",
    "DEGREE_NAMES",
    "INVERSION_NAMES",
    "INTERVAL_NAME_OPTION_SUPPORT",
    "KEY_SIGNATURES",
    "NOTE_NAME_OPTION_SUPPORT",
    "QUALITY_BY_MAJOR_DEGREE",
    "ROMAN_BY_DEGREE",
    "SIMPLE_METER_SIGNATURES",
    "TIME_SIGNATURE_UNITS",
    "build_interval_pitch",
    "display_accidental_for_key",
    "duration_name",
    "duration_partition",
    "format_pitch",
    "interval_name",
    "invert_chord",
    "key_signature_accidentals_by_letter",
    "key_signature_text",
    "major_scale_pitch",
    "normalize_inverted_chord_for_staff",
    "normalize_chord_for_staff",
    "pitch_midi",
    "random_staff_pitch",
    "title_for",
]
