"""Shared rendering and notation helpers for symbolic music-staff symbolic tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default
from ....shared.text_rendering import load_font, symbol_safe_font_for_text
from ....shared.text_legibility import draw_text_traced
from ...shared.drawing import draw_rounded_rect
from ...shared.scene_style import SymbolicSceneStyle
from ...shared.unit_size_jitter import resolve_symbolic_unit_size_scale, scale_symbolic_px


LETTERS: Tuple[str, ...] = ("C", "D", "E", "F", "G", "A", "B")
NATURAL_SEMITONES: Dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}
ACCIDENTAL_TEXT: Dict[int, str] = {-1: "b", 0: "", 1: "#"}
ACCIDENTAL_DISPLAY_TEXT: Dict[str, str] = {
    "flat": "\u266d",
    "b": "\u266d",
    "\u266d": "\u266d",
    "sharp": "\u266f",
    "#": "\u266f",
    "\u266f": "\u266f",
    "natural": "\u266e",
    "\u266e": "\u266e",
}
DEGREE_NAMES: Tuple[str, ...] = (
    "tonic",
    "supertonic",
    "mediant",
    "subdominant",
    "dominant",
    "submediant",
    "leading tone",
)
MAJOR_SCALE_STEPS: Tuple[int, ...] = (0, 2, 4, 5, 7, 9, 11)


@dataclass(frozen=True)
class Pitch:
    letter: str
    octave: int
    accidental: int = 0


@dataclass(frozen=True)
class StaffNote:
    item_id: str
    staff_index: int
    slot: float
    pitch: Pitch
    duration_units: int = 2
    filled: bool = True
    accidental_visible: bool = False
    marker: str = ""
    dotted: bool = False
    stem_up: bool = True
    display_accidental: str = ""


@dataclass(frozen=True)
class StaffChord:
    item_id: str
    staff_index: int
    slot: float
    pitches: Tuple[Pitch, ...]
    marker: str = ""
    accidental_visible: bool = False


@dataclass(frozen=True)
class StaffText:
    item_id: str
    staff_index: int
    slot: float
    text: str
    y_offset_steps: float = -4.5
    bold: bool = True


@dataclass(frozen=True)
class StaffTimeSignature:
    item_id: str
    staff_index: int
    slot: float
    text: str


@dataclass(frozen=True)
class StaffBarline:
    item_id: str
    staff_index: int
    slot: float


@dataclass(frozen=True)
class StaffRange:
    item_id: str
    staff_index: int
    start_slot: float
    end_slot: float
    label: str = ""
    show_bracket: bool = True
    bracket_y_offset_px: int = 0
    label_y_offset_px: int = 0


@dataclass(frozen=True)
class StaffSymbol:
    item_id: str
    staff_index: int
    slot: float
    pitch: Pitch
    symbol: str
    marker: str = ""


@dataclass(frozen=True)
class StaffSystem:
    clef: str
    slot_count: int
    key_signature: str = ""
    key_signature_id: str = ""
    time_signature: str = ""
    time_signature_id: str = ""
    subtitle: str = ""
    notes: Tuple[StaffNote, ...] = ()
    chords: Tuple[StaffChord, ...] = ()
    texts: Tuple[StaffText, ...] = ()
    time_signatures: Tuple[StaffTimeSignature, ...] = ()
    barlines: Tuple[StaffBarline, ...] = ()
    ranges: Tuple[StaffRange, ...] = ()
    symbols: Tuple[StaffSymbol, ...] = ()


@dataclass(frozen=True)
class OptionCard:
    item_id: str
    label: str
    text: str = ""
    duration_units: int | None = None
    is_correct: bool = False


@dataclass(frozen=True)
class MusicSceneSpec:
    title: str
    systems: Tuple[StaffSystem, ...]
    option_cards: Tuple[OptionCard, ...] = ()
    footer_text: str = ""


@dataclass(frozen=True)
class MusicRenderParams:
    canvas_width: int
    canvas_height: int
    panel_padding_px: int
    panel_corner_radius_px: int
    panel_border_width_px: int
    staff_gap_px: int
    staff_width_px: int
    staff_spacing_px: int
    slot_gap_px: int
    title_font_size_px: int
    label_font_size_px: int
    small_font_size_px: int
    notehead_width_px: int
    notehead_height_px: int
    stem_height_px: int
    option_card_width_px: int
    option_card_height_px: int
    option_gap_px: int
    unit_size_jitter: Dict[str, Any]


@dataclass(frozen=True)
class RenderedMusicScene:
    image: Image.Image
    scene_bbox_px: Tuple[int, int, int, int]
    item_bboxes: Dict[str, Tuple[int, int, int, int]]
    entities: Tuple[Dict[str, Any], ...]
    layout_jitter: Dict[str, Any]
    style_metadata: Dict[str, Any]


@dataclass(frozen=True)
class _DurationVisualStyle:
    filled: bool
    stem: bool
    dotted: bool
    flag: bool


def make_staff_note_sequence(
    pitches: Sequence[Pitch],
    *,
    item_ids: Sequence[str],
    markers: Sequence[str] | None = None,
    duration_units: Sequence[int] | None = None,
    start_slot: float = 1.0,
    slot_step: float = 2.20,
    staff_index: int = 0,
    accidental_visible: bool = False,
    display_accidentals: Sequence[str] | None = None,
    infer_duration_shape: bool = False,
) -> Tuple[StaffNote, ...]:
    """Create a left-to-right note excerpt from task-owned semantic bindings."""

    pitch_values = tuple(pitches)
    id_values = tuple(str(value) for value in item_ids)
    if len(id_values) != len(pitch_values):
        raise ValueError("music staff note item ids must match pitch count")
    marker_values = tuple(str(value) for value in markers) if markers is not None else tuple("" for _ in pitch_values)
    if len(marker_values) != len(pitch_values):
        raise ValueError("music staff note markers must match pitch count")
    duration_values = (
        tuple(int(value) for value in duration_units)
        if duration_units is not None
        else tuple(2 for _ in pitch_values)
    )
    if len(duration_values) != len(pitch_values):
        raise ValueError("music staff note durations must match pitch count")
    display_values = (
        tuple(str(value) for value in display_accidentals)
        if display_accidentals is not None
        else tuple("" for _ in pitch_values)
    )
    if len(display_values) != len(pitch_values):
        raise ValueError("music staff note display accidentals must match pitch count")
    notes = []
    for note_index, (item_id, marker, units, display_accidental, pitch) in enumerate(zip(id_values, marker_values, duration_values, display_values, pitch_values)):
        notes.append(
            StaffNote(
                item_id,
                int(staff_index),
                float(start_slot) + note_index * float(slot_step),
                pitch,
                duration_units=int(units),
                filled=(int(units) <= 3 if bool(infer_duration_shape) else True),
                accidental_visible=bool(accidental_visible),
                marker=str(marker),
                dotted=(int(units) in (3, 6) if bool(infer_duration_shape) else False),
                stem_up=_stem_up_for_pitch(pitch, "treble"),
                display_accidental=str(display_accidental),
            )
        )
    return tuple(notes)


def pitch_diatonic_index(pitch: Pitch) -> int:
    return int(pitch.octave) * 7 + LETTERS.index(str(pitch.letter))


def pitch_from_diatonic_index(index: int, accidental: int = 0) -> Pitch:
    idx = int(index)
    octave = idx // 7
    letter = LETTERS[idx % 7]
    return Pitch(letter=str(letter), octave=int(octave), accidental=int(accidental))


def pitch_from_staff_step(clef: str, step: int, accidental: int = 0) -> Pitch:
    return pitch_from_diatonic_index(_staff_reference_index(str(clef)) + int(step), accidental=int(accidental))


def pitch_midi(pitch: Pitch) -> int:
    return (int(pitch.octave) + 1) * 12 + int(NATURAL_SEMITONES[str(pitch.letter)]) + int(pitch.accidental)


def format_pitch(pitch: Pitch, *, include_octave: bool = False) -> str:
    accidental = ACCIDENTAL_TEXT.get(int(pitch.accidental), "")
    text = f"{pitch.letter}{accidental}"
    if include_octave:
        text = f"{text}{int(pitch.octave)}"
    return text


def format_key_signature(accidentals: Sequence[str]) -> str:
    return " ".join(str(value) for value in accidentals)


def _staff_reference_index(clef: str) -> int:
    if str(clef) == "bass":
        return pitch_diatonic_index(Pitch("G", 2, 0))
    return pitch_diatonic_index(Pitch("E", 4, 0))


def staff_step_for_pitch(pitch: Pitch, clef: str) -> int:
    return int(pitch_diatonic_index(pitch) - _staff_reference_index(str(clef)))


def major_scale_pitch(root: Pitch, degree_1based: int) -> Pitch:
    degree = int(degree_1based)
    root_index = pitch_diatonic_index(root)
    target_index = root_index + degree - 1
    natural = pitch_from_diatonic_index(target_index, accidental=0)
    wanted = (pitch_midi(root) + MAJOR_SCALE_STEPS[degree - 1]) % 12
    natural_pc = pitch_midi(natural) % 12
    diff = (wanted - natural_pc) % 12
    accidental = diff if diff <= 6 else diff - 12
    if accidental not in (-1, 0, 1):
        accidental = 0
    return Pitch(str(natural.letter), int(natural.octave), int(accidental))


def interval_name(lower: Pitch, upper: Pitch) -> str:
    low = lower if pitch_midi(lower) <= pitch_midi(upper) else upper
    high = upper if pitch_midi(lower) <= pitch_midi(upper) else lower
    number = abs(pitch_diatonic_index(high) - pitch_diatonic_index(low)) + 1
    simple_number = ((number - 1) % 7) + 1
    octaves = (number - 1) // 7
    semitones = abs(pitch_midi(high) - pitch_midi(low))
    simple_semitones = semitones - (12 * octaves)
    if simple_number in (1, 4, 5):
        perfect_base = {1: 0, 4: 5, 5: 7}[simple_number]
        delta = int(simple_semitones - perfect_base)
        quality = "perfect" if delta == 0 else "augmented" if delta == 1 else "diminished"
    else:
        major_base = {2: 2, 3: 4, 6: 9, 7: 11}[simple_number]
        delta = int(simple_semitones - major_base)
        if delta == 0:
            quality = "major"
        elif delta == -1:
            quality = "minor"
        elif delta == 1:
            quality = "augmented"
        else:
            quality = "diminished"
    ordinal = {
        1: "unison",
        2: "2nd",
        3: "3rd",
        4: "4th",
        5: "5th",
        6: "6th",
        7: "7th",
        8: "octave",
    }.get(number, f"{number}th")
    return f"{quality} {ordinal}"


def build_interval_pitch(root: Pitch, number: int, quality: str) -> Pitch | None:
    target_index = pitch_diatonic_index(root) + int(number) - 1
    natural = pitch_from_diatonic_index(target_index, accidental=0)
    simple_number = ((int(number) - 1) % 7) + 1
    octave_offset = (int(number) - 1) // 7
    if simple_number in (1, 4, 5):
        base = {1: 0, 4: 5, 5: 7}[simple_number] + 12 * octave_offset
        delta = {"diminished": -1, "perfect": 0, "augmented": 1}.get(str(quality))
    else:
        base = {2: 2, 3: 4, 6: 9, 7: 11}[simple_number] + 12 * octave_offset
        delta = {"diminished": -2, "minor": -1, "major": 0, "augmented": 1}.get(str(quality))
    if delta is None:
        return None
    wanted = pitch_midi(root) + int(base) + int(delta)
    accidental = int(wanted - pitch_midi(natural))
    if accidental not in (-1, 0, 1):
        return None
    return Pitch(str(natural.letter), int(natural.octave), int(accidental))


def resolve_music_render_params(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    instance_seed: int,
) -> MusicRenderParams:
    # Preserve legacy puzzle seed namespaces so the domain split does not
    # change music-staff scale or layout for existing seeds.
    unit_scale, unit_meta = resolve_symbolic_unit_size_scale(
        params,
        defaults,
        instance_seed=int(instance_seed),
        namespace="puzzles.notation.unit_size",
    )
    return MusicRenderParams(
        canvas_width=int(group_default(defaults, "canvas_width", 1180)),
        canvas_height=int(group_default(defaults, "canvas_height", 820)),
        panel_padding_px=scale_symbolic_px(group_default(defaults, "panel_padding_px", 34), unit_scale, min_px=18),
        panel_corner_radius_px=scale_symbolic_px(group_default(defaults, "panel_corner_radius_px", 18), unit_scale, min_px=8),
        panel_border_width_px=scale_symbolic_px(group_default(defaults, "panel_border_width_px", 2), unit_scale, min_px=1),
        staff_gap_px=scale_symbolic_px(group_default(defaults, "staff_gap_px", 15), unit_scale, min_px=10),
        staff_width_px=scale_symbolic_px(group_default(defaults, "staff_width_px", 760), unit_scale, min_px=760),
        staff_spacing_px=scale_symbolic_px(group_default(defaults, "staff_spacing_px", 142), unit_scale, min_px=98),
        slot_gap_px=scale_symbolic_px(group_default(defaults, "slot_gap_px", 52), unit_scale, min_px=44),
        title_font_size_px=scale_symbolic_px(group_default(defaults, "title_font_size_px", 25), unit_scale, min_px=16),
        label_font_size_px=scale_symbolic_px(group_default(defaults, "label_font_size_px", 21), unit_scale, min_px=13),
        small_font_size_px=scale_symbolic_px(group_default(defaults, "small_font_size_px", 16), unit_scale, min_px=11),
        notehead_width_px=scale_symbolic_px(group_default(defaults, "notehead_width_px", 21), unit_scale, min_px=13),
        notehead_height_px=scale_symbolic_px(group_default(defaults, "notehead_height_px", 14), unit_scale, min_px=9),
        stem_height_px=scale_symbolic_px(group_default(defaults, "stem_height_px", 48), unit_scale, min_px=30),
        option_card_width_px=scale_symbolic_px(group_default(defaults, "option_card_width_px", 254), unit_scale, min_px=170),
        option_card_height_px=scale_symbolic_px(group_default(defaults, "option_card_height_px", 70), unit_scale, min_px=48),
        option_gap_px=scale_symbolic_px(group_default(defaults, "option_gap_px", 12), unit_scale, min_px=8),
        unit_size_jitter=dict(unit_meta),
    )


def _bbox_union(bboxes: Sequence[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    if not bboxes:
        return (0, 0, 0, 0)
    return (
        int(min(b[0] for b in bboxes)),
        int(min(b[1] for b in bboxes)),
        int(max(b[2] for b in bboxes)),
        int(max(b[3] for b in bboxes)),
    )


def _expand_bbox_to_min_side(
    bbox: Tuple[int, int, int, int],
    *,
    min_side_px: int = 24,
) -> Tuple[int, int, int, int]:
    """Pad an annotation-source bbox so tiny glyphs remain reviewable."""

    x0, y0, x1, y1 = (int(value) for value in bbox)
    width = max(0, x1 - x0)
    height = max(0, y1 - y0)
    if width < int(min_side_px):
        pad = (int(min_side_px) - int(width) + 1) // 2
        x0 -= pad
        x1 += pad
    if height < int(min_side_px):
        pad = (int(min_side_px) - int(height) + 1) // 2
        y0 -= pad
        y1 += pad
    return (int(x0), int(y0), int(x1), int(y1))


def _staff_y_for_pitch(staff_top: int, staff_gap: int, pitch: Pitch, clef: str) -> int:
    bottom_y = int(staff_top) + 4 * int(staff_gap)
    return int(round(bottom_y - staff_step_for_pitch(pitch, str(clef)) * (int(staff_gap) / 2.0)))


def _stem_up_for_pitch(pitch: Pitch, clef: str) -> bool:
    """Use the common single-voice engraving convention for stem direction."""

    return staff_step_for_pitch(pitch, str(clef)) <= 4


def _duration_visual_style(
    duration_units: int,
    *,
    filled: bool = True,
    dotted: bool = False,
) -> _DurationVisualStyle:
    """Map Trace duration units to the simplified note glyph we render."""

    units = int(duration_units)
    if units == 1:
        return _DurationVisualStyle(filled=True, stem=True, dotted=False, flag=True)
    if units == 2:
        return _DurationVisualStyle(filled=True, stem=True, dotted=False, flag=False)
    if units == 3:
        return _DurationVisualStyle(filled=True, stem=True, dotted=True, flag=False)
    if units == 4:
        return _DurationVisualStyle(filled=False, stem=True, dotted=False, flag=False)
    if units == 6:
        return _DurationVisualStyle(filled=False, stem=True, dotted=True, flag=False)
    if units == 8:
        return _DurationVisualStyle(filled=False, stem=False, dotted=False, flag=False)
    return _DurationVisualStyle(filled=bool(filled), stem=True, dotted=bool(dotted), flag=False)


def _duration_style_metadata(style: _DurationVisualStyle) -> Dict[str, bool]:
    return {
        "filled_notehead": bool(style.filled),
        "has_stem": bool(style.stem),
        "has_dot": bool(style.dotted),
        "has_flag": bool(style.flag),
    }


def _slot_x(content_x0: int, slot_gap: int, slot: float) -> int:
    return int(round(int(content_x0) + float(slot) * int(slot_gap)))


def _draw_text_bbox(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    *,
    font,
    fill: Tuple[int, int, int],
    stroke_fill: Tuple[int, int, int],
    stroke_width: int = 1,
) -> Tuple[int, int, int, int]:
    x, y = int(xy[0]), int(xy[1])
    draw_text_traced(draw,(x, y), str(text), fill=fill, font=font, stroke_width=int(stroke_width), stroke_fill=stroke_fill, role="readout", required=False)
    bbox = draw.textbbox((x, y), str(text), font=font, stroke_width=int(stroke_width))
    return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))


def _display_accidental_text(value: str) -> str:
    return ACCIDENTAL_DISPLAY_TEXT.get(str(value).strip(), str(value).strip())


def _legacy_pitch_accidental_text(accidental: int) -> str:
    return ACCIDENTAL_DISPLAY_TEXT.get(ACCIDENTAL_TEXT.get(int(accidental), ""), ACCIDENTAL_TEXT.get(int(accidental), ""))


def _note_display_accidental(note: StaffNote) -> str:
    explicit = _display_accidental_text(str(note.display_accidental))
    if explicit:
        return explicit
    if bool(note.accidental_visible) and int(note.pitch.accidental) != 0:
        return _legacy_pitch_accidental_text(int(note.pitch.accidental))
    return ""


def _draw_accidental_bbox(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    accidental_text: str,
    *,
    font,
    fill: Tuple[int, int, int],
    stroke_fill: Tuple[int, int, int],
) -> Tuple[int, int, int, int]:
    resolved_text = _display_accidental_text(str(accidental_text))
    resolved_font = symbol_safe_font_for_text(str(resolved_text), font)
    return _draw_text_bbox(
        draw,
        xy,
        str(resolved_text),
        font=resolved_font,
        fill=fill,
        stroke_fill=stroke_fill,
        stroke_width=0,
    )


def _wrap_text_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font,
    max_width: int,
    max_lines: int = 2,
    stroke_width: int = 1,
) -> tuple[str, ...] | None:
    """Greedily wrap text into at most `max_lines` lines for option cards."""

    words = [word for word in str(text).split() if word]
    if not words:
        return ("",)
    lines: list[str] = []
    current = ""
    for index, word in enumerate(words):
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=int(stroke_width))
        if int(bbox[2] - bbox[0]) <= int(max_width):
            current = candidate
            continue
        if current:
            lines.append(current)
            current = word
        else:
            lines.append(word)
            current = ""
        if len(lines) >= int(max_lines):
            if current or words[index + 1 :]:
                return None
    if current:
        lines.append(current)
    if len(lines) > int(max_lines):
        return None
    if any(
        int(draw.textbbox((0, 0), line, font=font, stroke_width=int(stroke_width))[2]) > int(max_width)
        for line in lines
    ):
        return None
    return tuple(lines)


def _fit_option_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    max_width: int,
    max_height: int,
    base_font_size: int,
    min_font_size: int = 10,
) -> tuple[Any, tuple[str, ...], int]:
    """Resolve an option-card text font and wrapped lines."""

    for font_size in range(int(base_font_size), int(min_font_size) - 1, -1):
        font = load_font(font_size, bold=True)
        lines = _wrap_text_lines(draw, str(text), font=font, max_width=int(max_width), max_lines=2, stroke_width=1)
        if not lines:
            continue
        line_heights = [
            int(draw.textbbox((0, 0), line, font=font, stroke_width=1)[3] - draw.textbbox((0, 0), line, font=font, stroke_width=1)[1])
            for line in lines
        ]
        total_height = sum(line_heights) + max(0, len(lines) - 1) * 3
        if total_height <= int(max_height):
            return font, tuple(lines), int(total_height)
    font = load_font(int(min_font_size), bold=True)
    return font, (str(text),), int(draw.textbbox((0, 0), str(text), font=font, stroke_width=1)[3])


def _measure_text_width(draw: ImageDraw.ImageDraw, text: str, *, font, stroke_width: int = 1) -> int:
    bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=int(stroke_width))
    return int(bbox[2] - bbox[0])


def _parse_time_signature(text: str) -> tuple[str, str] | None:
    parts = str(text).strip().split("/")
    if len(parts) != 2:
        return None
    numerator, denominator = (part.strip() for part in parts)
    if not numerator.isdigit() or not denominator.isdigit():
        return None
    return str(numerator), str(denominator)


def _time_signature_font(render_params: MusicRenderParams):
    return load_font(max(28, int(round(int(render_params.staff_gap_px) * 2.15))), bold=True)


def _measure_stacked_time_signature_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    render_params: MusicRenderParams,
) -> int | None:
    parsed = _parse_time_signature(str(text))
    if parsed is None:
        return None
    font = _time_signature_font(render_params)
    widths = [
        int(draw.textbbox((0, 0), value, font=font, stroke_width=0)[2] - draw.textbbox((0, 0), value, font=font, stroke_width=0)[0])
        for value in parsed
    ]
    return max(max(widths) + 8, int(round(int(render_params.staff_gap_px) * 1.75)))


def _system_content_offset(
    draw: ImageDraw.ImageDraw,
    system: StaffSystem,
    *,
    staff_x0: int,
    label_font,
    render_params: MusicRenderParams,
) -> int:
    cursor_x = int(staff_x0) + 55
    if str(system.key_signature):
        cursor_x += _measure_text_width(draw, str(system.key_signature), font=label_font) + 18
    if str(system.time_signature):
        stacked_width = _measure_stacked_time_signature_width(draw, str(system.time_signature), render_params=render_params)
        cursor_x += (stacked_width if stacked_width is not None else _measure_text_width(draw, str(system.time_signature), font=label_font)) + 26
    return max(int(staff_x0) + 145, int(cursor_x) + 12) - int(staff_x0)


def _system_max_slot(system: StaffSystem) -> float:
    slots: list[float] = [float(max(0, int(system.slot_count) - 1))]
    slots.extend(float(note.slot) for note in system.notes)
    slots.extend(float(chord.slot) for chord in system.chords)
    slots.extend(float(text.slot) for text in system.texts)
    slots.extend(float(time_signature.slot) for time_signature in system.time_signatures)
    slots.extend(float(barline.slot) for barline in system.barlines)
    slots.extend(float(staff_range.end_slot) for staff_range in system.ranges)
    slots.extend(float(symbol.slot) for symbol in system.symbols)
    return max(slots) if slots else 0.0


def _required_staff_draw_width(
    *,
    content_offset_px: int,
    max_slot: float,
    render_params: MusicRenderParams,
) -> int:
    content_extent = int(content_offset_px) + int(round((float(max_slot) + 0.35) * int(render_params.slot_gap_px)))
    glyph_padding = max(24, int(render_params.notehead_width_px) + 14)
    return max(int(render_params.staff_width_px), int(content_extent + glyph_padding))


def _draw_duration_note_glyph(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[int, int],
    duration_units: int,
    render_params: MusicRenderParams,
    text_rgb: Tuple[int, int, int],
    stroke_rgb: Tuple[int, int, int],
    stem_up: bool,
    filled: bool = True,
    dotted: bool = False,
    stem_height_px: int | None = None,
) -> tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], _DurationVisualStyle]:
    """Draw one simplified note-value glyph.

    This is intentionally not a full engraving engine, but it keeps the
    duration vocabulary visually consistent across staff notes and option
    glyphs.
    """

    x, y = int(center[0]), int(center[1])
    style = _duration_visual_style(int(duration_units), filled=bool(filled), dotted=bool(dotted))
    stem_height = int(stem_height_px) if stem_height_px is not None else int(render_params.stem_height_px)
    hw = int(render_params.notehead_width_px) // 2
    hh = int(render_params.notehead_height_px) // 2
    notehead_bbox = (x - hw, y - hh, x + hw, y + hh)
    draw.ellipse(
        notehead_bbox,
        fill=text_rgb if bool(style.filled) else stroke_rgb,
        outline=text_rgb,
        width=2,
    )
    parts: list[Tuple[int, int, int, int]] = [notehead_bbox]
    if bool(style.stem):
        stem_x = notehead_bbox[2] - 1 if bool(stem_up) else notehead_bbox[0] + 1
        stem_y1 = int(y - stem_height) if bool(stem_up) else int(y + stem_height)
        draw.line((stem_x, y, stem_x, stem_y1), fill=text_rgb, width=2)
        parts.append((stem_x - 2, min(y, stem_y1), stem_x + 2, max(y, stem_y1)))
        if bool(style.flag):
            if bool(stem_up):
                flag_bbox = (stem_x - 1, stem_y1, stem_x + 22, stem_y1 + 24)
                draw.arc(flag_bbox, start=250, end=75, fill=text_rgb, width=2)
            else:
                flag_bbox = (stem_x - 22, stem_y1 - 24, stem_x + 1, stem_y1)
                draw.arc(flag_bbox, start=90, end=275, fill=text_rgb, width=2)
            parts.append(flag_bbox)
    if bool(style.dotted):
        dot_bbox = (notehead_bbox[2] + 8, y - 3, notehead_bbox[2] + 14, y + 3)
        draw.ellipse(dot_bbox, fill=text_rgb)
        parts.append(dot_bbox)
    return _bbox_union(parts), notehead_bbox, style


def _draw_note(
    draw: ImageDraw.ImageDraw,
    *,
    note: StaffNote,
    staff_top: int,
    content_x0: int,
    clef: str,
    render_params: MusicRenderParams,
    text_rgb: Tuple[int, int, int],
    stroke_rgb: Tuple[int, int, int],
    mark_rgb: Tuple[int, int, int],
    font,
    small_font,
) -> Tuple[int, int, int, int]:
    """Draw one staff note and return the full bbox of all visible note parts.

    The returned bbox is the annotation source for note-reading tasks, so it
    includes accidentals, dots, stems, and ledger lines that visually identify
    the note. Number markers are rendered separately and are not annotation
    witnesses.
    """

    x = _slot_x(content_x0, int(render_params.slot_gap_px), float(note.slot))
    y = _staff_y_for_pitch(staff_top, int(render_params.staff_gap_px), note.pitch, str(clef))
    glyph_bbox, notehead_bbox, _style = _draw_duration_note_glyph(
        draw,
        center=(x, y),
        duration_units=int(note.duration_units),
        render_params=render_params,
        text_rgb=text_rgb,
        stroke_rgb=stroke_rgb,
        stem_up=_stem_up_for_pitch(note.pitch, str(clef)),
        filled=bool(note.filled),
        dotted=bool(note.dotted),
    )
    parts = [glyph_bbox]
    display_accidental = _note_display_accidental(note)
    if display_accidental:
        accidental_font = load_font(max(int(render_params.label_font_size_px) + 8, int(round(int(render_params.staff_gap_px) * 2.35))), bold=False)
        acc_bbox = _draw_accidental_bbox(
            draw,
            (notehead_bbox[0] - 32, y - int(render_params.label_font_size_px) // 2 - 4),
            display_accidental,
            font=accidental_font,
            fill=text_rgb,
            stroke_fill=stroke_rgb,
        )
        parts.append(acc_bbox)
    step = staff_step_for_pitch(note.pitch, str(clef))
    if step < 0 or step > 8:
        for ledger_step in range(min(0, step), max(8, step) + 1):
            if ledger_step % 2 == 0 and (ledger_step < 0 or ledger_step > 8):
                ly = int(round(staff_top + 4 * int(render_params.staff_gap_px) - ledger_step * (int(render_params.staff_gap_px) / 2.0)))
                draw.line((notehead_bbox[0] - 8, ly, notehead_bbox[2] + 8, ly), fill=text_rgb, width=1)
                parts.append((notehead_bbox[0] - 8, ly - 1, notehead_bbox[2] + 8, ly + 1))
    return _expand_bbox_to_min_side(_bbox_union(parts))


def _draw_item_marker(
    draw: ImageDraw.ImageDraw,
    *,
    item_id: str,
    target_bbox: Tuple[int, int, int, int],
    label: str,
    staff_top: int,
    render_params: MusicRenderParams,
    mark_rgb: Tuple[int, int, int],
    stroke_rgb: Tuple[int, int, int],
    font,
) -> tuple[str, Tuple[int, int, int, int], dict[str, Any]]:
    """Draw a fixed number marker without target-dependent pointer geometry."""

    target_x = int(round((int(target_bbox[0]) + int(target_bbox[2])) / 2))
    label_bbox_at_origin = draw.textbbox((0, 0), str(label), font=font, stroke_width=1)
    label_width = int(label_bbox_at_origin[2] - label_bbox_at_origin[0])
    marker_clearance_px = 10
    marker_offset = max(58, int(round(int(render_params.staff_gap_px) * 3.9)))
    desired_text_y = int(staff_top - marker_offset)
    max_text_y_without_overlap = int(target_bbox[1]) - int(marker_clearance_px) - int(label_bbox_at_origin[3])
    text_y = int(min(desired_text_y, max_text_y_without_overlap))
    text_x = int(target_x - label_width / 2)
    label_bbox = _draw_text_bbox(
        draw,
        (text_x, text_y),
        str(label),
        font=font,
        fill=mark_rgb,
        stroke_fill=stroke_rgb,
        stroke_width=1,
    )
    marker_bbox = tuple(int(value) for value in label_bbox)
    marker_id = f"{item_id}_marker"
    entity = {
        "entity_id": marker_id,
        "entity_type": "music_item_marker",
        "bbox_px": list(marker_bbox),
        "label": str(label),
        "target_item_id": str(item_id),
    }
    return marker_id, marker_bbox, entity


def _draw_chord(
    draw: ImageDraw.ImageDraw,
    *,
    chord: StaffChord,
    staff_top: int,
    content_x0: int,
    clef: str,
    render_params: MusicRenderParams,
    text_rgb: Tuple[int, int, int],
    stroke_rgb: Tuple[int, int, int],
    mark_rgb: Tuple[int, int, int],
    font,
) -> Tuple[int, int, int, int]:
    """Draw a stacked chord glyph and return the bbox used for chord witnesses.

    The chord bbox intentionally unions every notehead, accidental, and stem so
    harmony tasks can bind one visual witness per chord without depending on
    staff-line pixels. Number markers are rendered separately.
    """

    bboxes = []
    x = _slot_x(content_x0, int(render_params.slot_gap_px), float(chord.slot))
    sorted_pitches = tuple(sorted(chord.pitches, key=lambda p: staff_step_for_pitch(p, str(clef))))
    hw = int(render_params.notehead_width_px) // 2
    hh = int(render_params.notehead_height_px) // 2
    note_infos = []
    previous_step: int | None = None
    previous_offset = 0
    for pitch in sorted_pitches:
        step = staff_step_for_pitch(pitch, str(clef))
        if previous_step is not None and abs(int(step) - int(previous_step)) == 1:
            offset = 0 if previous_offset else max(7, int(render_params.notehead_width_px) // 2)
        else:
            offset = 0
        previous_step = int(step)
        previous_offset = int(offset)
        y = _staff_y_for_pitch(staff_top, int(render_params.staff_gap_px), pitch, str(clef))
        bbox = (x - hw + offset, y - hh, x + hw + offset, y + hh)
        draw.ellipse(bbox, fill=text_rgb, outline=text_rgb, width=2)
        bboxes.append(bbox)
        note_infos.append({"pitch": pitch, "step": int(step), "y": int(y), "bbox": bbox, "offset": int(offset)})
        if int(step) < 0 or int(step) > 8:
            for ledger_step in range(min(0, int(step)), max(8, int(step)) + 1):
                if ledger_step % 2 == 0 and (ledger_step < 0 or ledger_step > 8):
                    ly = int(round(staff_top + 4 * int(render_params.staff_gap_px) - ledger_step * (int(render_params.staff_gap_px) / 2.0)))
                    ledger_bbox = (bbox[0] - 7, ly - 1, bbox[2] + 7, ly + 1)
                    draw.line((ledger_bbox[0], ly, ledger_bbox[2], ly), fill=text_rgb, width=1)
                    bboxes.append(ledger_bbox)
    accidental_rows: list[int] = []
    for info in note_infos:
        pitch = info["pitch"]
        if bool(chord.accidental_visible) and int(pitch.accidental) != 0:
            nearby_count = sum(1 for row in accidental_rows if abs(int(row) - int(info["step"])) <= 1)
            accidental_rows.append(int(info["step"]))
            accidental_x = x - hw - 28 - nearby_count * 13
            bboxes.append(
                _draw_accidental_bbox(
                    draw,
                    (int(accidental_x), int(info["y"]) - int(render_params.label_font_size_px) // 2),
                    ACCIDENTAL_TEXT[int(pitch.accidental)],
                    font=font,
                    fill=text_rgb,
                    stroke_fill=stroke_rgb,
                )
            )
    if note_infos:
        center_step = sum(int(info["step"]) for info in note_infos) / max(1, len(note_infos))
        if center_step <= 4:
            stem_x = max(int(info["bbox"][2]) for info in note_infos) + 1
            stem_y0 = min(int(info["y"]) for info in note_infos) - max(22, int(render_params.stem_height_px) // 2)
            stem_y1 = max(int(info["y"]) for info in note_infos)
        else:
            stem_x = min(int(info["bbox"][0]) for info in note_infos) - 1
            stem_y0 = min(int(info["y"]) for info in note_infos)
            stem_y1 = max(int(info["y"]) for info in note_infos) + max(22, int(render_params.stem_height_px) // 2)
        draw.line((stem_x, stem_y0, stem_x, stem_y1), fill=text_rgb, width=2)
        bboxes.append((stem_x - 2, stem_y0, stem_x + 2, stem_y1))
    return _expand_bbox_to_min_side(_bbox_union(bboxes))


def _draw_duration_glyph(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[int, int],
    duration_units: int,
    render_params: MusicRenderParams,
    text_rgb: Tuple[int, int, int],
    stroke_rgb: Tuple[int, int, int],
    font,
    stem_height_px: int | None = None,
) -> Tuple[int, int, int, int]:
    """Draw an isolated duration glyph and return its prompt-facing bbox.

    Duration-option tasks use this bbox for the target glyph and option cards,
    so the union includes dots, stems, flags, and fallback unit labels.
    """

    x, y = int(center[0]), int(center[1])
    units = int(duration_units)
    glyph_bbox, _notehead_bbox, _style = _draw_duration_note_glyph(
        draw,
        center=(x, y),
        duration_units=int(units),
        render_params=render_params,
        text_rgb=text_rgb,
        stroke_rgb=stroke_rgb,
        stem_up=True,
        stem_height_px=stem_height_px,
    )
    parts = [glyph_bbox]
    if units not in (1, 2, 3, 4, 6, 8):
        parts.append(
            _draw_text_bbox(
                draw,
                (x + 18, y - 10),
                f"{units}u",
                font=font,
                fill=text_rgb,
                stroke_fill=stroke_rgb,
            )
        )
    return _bbox_union(parts)


_TREBLE_SHARP_STEPS: Tuple[int, ...] = (8, 5, 9, 6, 3, 7, 4)
_TREBLE_FLAT_STEPS: Tuple[int, ...] = (4, 7, 3, 6, 2, 5, 1)


def _key_signature_tokens(text: str) -> tuple[str, ...]:
    tokens = tuple(part for part in str(text).split() if part)
    if tokens and all(token == "#" for token in tokens):
        return tokens
    if tokens and all(token == "b" for token in tokens):
        return tokens
    return ()


def _key_signature_staff_steps(text: str, clef: str) -> tuple[int, ...]:
    tokens = _key_signature_tokens(str(text))
    if not tokens or str(clef) != "treble":
        return ()
    steps = _TREBLE_SHARP_STEPS if str(tokens[0]) == "#" else _TREBLE_FLAT_STEPS
    return tuple(int(steps[index]) for index in range(len(tokens)))


def _draw_key_accidental_glyph(
    draw: ImageDraw.ImageDraw,
    *,
    token: str,
    center: Tuple[int, int],
    staff_gap: int,
    fill: Tuple[int, int, int],
) -> Tuple[int, int, int, int]:
    x, y = int(center[0]), int(center[1])
    glyph = "\u266f" if str(token) == "#" else "\u266d"
    font = symbol_safe_font_for_text(
        glyph,
        load_font(max(34, int(round(int(staff_gap) * 3.35))), bold=False),
    )
    glyph_bbox = draw.textbbox((0, 0), glyph, font=font, stroke_width=0)
    glyph_width = int(glyph_bbox[2] - glyph_bbox[0])
    glyph_height = int(glyph_bbox[3] - glyph_bbox[1])
    origin_x = int(round(x - (glyph_bbox[0] + glyph_bbox[2]) / 2.0))
    origin_y = int(round(y - (glyph_bbox[1] + glyph_bbox[3]) / 2.0))
    draw_text_traced(draw, (origin_x, origin_y), glyph, fill=fill, font=font, role="symbol", required=False)
    return (
        int(origin_x + glyph_bbox[0]),
        int(origin_y + glyph_bbox[1]),
        int(origin_x + glyph_bbox[0] + glyph_width),
        int(origin_y + glyph_bbox[1] + glyph_height),
    )


def _draw_key_signature(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    clef: str,
    staff_top: int,
    cursor_x: int,
    render_params: MusicRenderParams,
    text_rgb: Tuple[int, int, int],
) -> Tuple[int, int, int, int] | None:
    tokens = _key_signature_tokens(str(text))
    if not tokens:
        return None
    staff_gap = int(render_params.staff_gap_px)
    x_spacing = max(24, int(round(staff_gap * 1.85)))
    if str(clef) != "treble":
        # The current task suite uses treble signatures; fallback keeps future
        # bass usage readable instead of pretending to know a bass order.
        return None
    steps = _key_signature_staff_steps(str(text), str(clef))
    bboxes: list[Tuple[int, int, int, int]] = []
    for index, token in enumerate(tokens):
        step = int(steps[index % len(steps)])
        y = int(round(staff_top + 4 * staff_gap - step * (staff_gap / 2.0)))
        x = int(cursor_x + index * x_spacing + max(10, staff_gap))
        bboxes.append(_draw_key_accidental_glyph(draw, token=str(token), center=(x, y), staff_gap=staff_gap, fill=text_rgb))
    return _expand_bbox_to_min_side(_bbox_union(bboxes))


def _draw_stacked_time_signature(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center_x: int,
    staff_top: int,
    render_params: MusicRenderParams,
    text_rgb: Tuple[int, int, int],
) -> Tuple[int, int, int, int] | None:
    parsed = _parse_time_signature(str(text))
    if parsed is None:
        return None
    numerator, denominator = parsed
    staff_gap = int(render_params.staff_gap_px)
    font = _time_signature_font(render_params)
    centers = (
        int(round(int(staff_top) + staff_gap * 1.12)),
        int(round(int(staff_top) + staff_gap * 2.88)),
    )
    bboxes: list[Tuple[int, int, int, int]] = []
    for value, center_y in zip((numerator, denominator), centers):
        raw_bbox = draw.textbbox((0, 0), value, font=font, stroke_width=0)
        width = int(raw_bbox[2] - raw_bbox[0])
        origin_x = int(round(int(center_x) - (raw_bbox[0] + raw_bbox[2]) / 2.0))
        origin_y = int(round(int(center_y) - (raw_bbox[1] + raw_bbox[3]) / 2.0))
        draw_text_traced(draw, (origin_x, origin_y), value, fill=text_rgb, font=font, role="readout", required=False)
        bboxes.append(
            (
                int(origin_x + raw_bbox[0]),
                int(origin_y + raw_bbox[1]),
                int(origin_x + raw_bbox[0] + width),
                int(origin_y + raw_bbox[3]),
            )
        )
    return _expand_bbox_to_min_side(_bbox_union(bboxes))


def _draw_fermata_symbol(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    fill: Tuple[int, int, int],
) -> Tuple[int, int, int, int]:
    """Draw a compact fermata mark as a curved arch over a dot."""

    arc_bbox = (int(x - 17), int(y - 13), int(x + 17), int(y + 13))
    draw.arc(arc_bbox, start=180, end=360, fill=fill, width=3)
    dot_bbox = (int(x - 3), int(y + 2), int(x + 3), int(y + 8))
    draw.ellipse(dot_bbox, fill=fill)
    return _bbox_union(((arc_bbox[0], arc_bbox[1], arc_bbox[2], int(y) + 1), dot_bbox))


def _draw_clef_symbol(
    draw: ImageDraw.ImageDraw,
    *,
    clef: str,
    x: int,
    staff_top: int,
    staff_gap: int,
    fill: Tuple[int, int, int],
) -> Tuple[int, int, int, int]:
    """Draw a compact vector clef.

    Local fonts do not reliably include SMuFL/music codepoints, so this uses
    simple traced primitives instead of text glyphs.
    """

    gap = int(staff_gap)
    if str(clef) == "bass":
        cx = int(x + gap * 1.05)
        cy = int(staff_top + gap * 1.55)
        arc_bbox = (int(cx - gap * 0.9), int(cy - gap * 0.95), int(cx + gap * 1.15), int(cy + gap * 1.05))
        draw.arc(arc_bbox, start=82, end=285, fill=fill, width=3)
        dot_main = (int(cx - gap * 0.30), int(cy - gap * 0.18), int(cx + gap * 0.18), int(cy + gap * 0.30))
        draw.ellipse(dot_main, fill=fill)
        upper_dot = (int(cx + gap * 1.25), int(cy - gap * 0.45), int(cx + gap * 1.58), int(cy - gap * 0.12))
        lower_dot = (int(cx + gap * 1.25), int(cy + gap * 0.34), int(cx + gap * 1.58), int(cy + gap * 0.67))
        draw.ellipse(upper_dot, fill=fill)
        draw.ellipse(lower_dot, fill=fill)
        return _bbox_union((arc_bbox, dot_main, upper_dot, lower_dot))

    cx = int(x + gap * 1.15)
    top = int(staff_top - gap * 1.15)
    bottom = int(staff_top + gap * 4.85)
    spine = (int(cx - 2), top, int(cx + 2), bottom)
    draw.line((cx, top, cx, bottom), fill=fill, width=3)
    top_loop = (int(cx - gap * 0.48), int(top - gap * 0.05), int(cx + gap * 0.70), int(top + gap * 1.18))
    draw.arc(top_loop, start=210, end=70, fill=fill, width=3)
    main_loop = (
        int(cx - gap * 1.10),
        int(staff_top + gap * 1.15),
        int(cx + gap * 1.12),
        int(staff_top + gap * 4.15),
    )
    draw.arc(main_loop, start=35, end=350, fill=fill, width=3)
    inner_loop = (
        int(cx - gap * 0.58),
        int(staff_top + gap * 2.12),
        int(cx + gap * 0.62),
        int(staff_top + gap * 3.35),
    )
    draw.arc(inner_loop, start=205, end=25, fill=fill, width=3)
    lower_hook = (
        int(cx - gap * 0.72),
        int(staff_top + gap * 3.52),
        int(cx + gap * 0.70),
        int(staff_top + gap * 5.28),
    )
    draw.arc(lower_hook, start=15, end=220, fill=fill, width=3)
    return _bbox_union((spine, top_loop, main_loop, inner_loop, lower_hook))


def render_music_scene(
    image: Image.Image,
    *,
    spec: MusicSceneSpec,
    render_params: MusicRenderParams,
    scene_style: SymbolicSceneStyle,
    instance_seed: int,
) -> RenderedMusicScene:
    """Render the complete music-staff scene and collect item bboxes.

    This is the scene renderer, not task logic: it lays out staff systems,
    option cards, titles, and notation glyphs from a task-owned
    `MusicSceneSpec`, while preserving item ids as bbox projection anchors.
    """

    draw = ImageDraw.Draw(image)
    label_font = load_font(int(render_params.label_font_size_px), bold=True)
    marker_font = load_font(max(int(render_params.label_font_size_px) + 7, 28), bold=True)
    small_font = load_font(int(render_params.small_font_size_px), bold=False)
    text_rgb = tuple(int(value) for value in scene_style.text_rgb)
    stroke_rgb = tuple(int(value) for value in scene_style.text_stroke_rgb)
    line_rgb = tuple(int(value) for value in scene_style.grid_rgb)
    panel_fill = tuple(int(value) for value in scene_style.panel_fill_rgb)
    option_fill = tuple(int(value) for value in scene_style.option_fill_rgb)
    option_border = tuple(int(value) for value in scene_style.mark_rgb)
    mark_rgb = tuple(int(value) for value in scene_style.mark_rgb)

    staff_count = max(1, len(spec.systems))
    option_count = len(spec.option_cards)
    option_columns = 3 if option_count > 4 else max(1, option_count)
    option_rows = (option_count + option_columns - 1) // option_columns if option_count else 0
    option_grid_width = (
        option_columns * int(render_params.option_card_width_px)
        + max(0, option_columns - 1) * int(render_params.option_gap_px)
        if option_count
        else 0
    )
    option_grid_height = (
        option_rows * int(render_params.option_card_height_px)
        + max(0, option_rows - 1) * int(render_params.option_gap_px)
        if option_count
        else 0
    )
    provisional_staff_x0 = int(render_params.panel_padding_px)
    required_staff_widths: Dict[int, int] = {}
    for staff_index, system in enumerate(spec.systems):
        content_offset = _system_content_offset(
            draw,
            system,
            staff_x0=provisional_staff_x0,
            label_font=label_font,
            render_params=render_params,
        )
        required_staff_widths[int(staff_index)] = _required_staff_draw_width(
            content_offset_px=int(content_offset),
            max_slot=float(_system_max_slot(system)),
            render_params=render_params,
        )
    required_staff_width = max(required_staff_widths.values()) if required_staff_widths else int(render_params.staff_width_px)
    content_width = max(int(render_params.staff_width_px) + 190, int(required_staff_width) + 40, int(option_grid_width))
    staff_content_height = 66 + (staff_count - 1) * int(render_params.staff_spacing_px) + 4 * int(render_params.staff_gap_px) + 45
    content_height = (
        staff_content_height + int(render_params.option_gap_px) + int(option_grid_height)
        if option_count
        else staff_content_height
    )
    panel_width = min(int(render_params.canvas_width) - 48, content_width + 2 * int(render_params.panel_padding_px))
    panel_height = min(int(render_params.canvas_height) - 48, content_height + 2 * int(render_params.panel_padding_px))
    max_x0 = max(24, int(render_params.canvas_width) - panel_width - 24)
    max_y0 = max(24, int(render_params.canvas_height) - panel_height - 24)
    rng = spawn_rng(int(instance_seed), "puzzles.notation.layout")
    panel_x0 = int(24 + (rng.randrange(max(1, max_x0 - 23)) if max_x0 > 24 else 0))
    panel_y0 = int(24 + (rng.randrange(max(1, max_y0 - 23)) if max_y0 > 24 else 0))
    panel_x1 = int(panel_x0 + panel_width)
    panel_y1 = int(panel_y0 + panel_height)
    draw_rounded_rect(
        draw,
        (panel_x0, panel_y0, panel_x1, panel_y1),
        radius=int(render_params.panel_corner_radius_px),
        fill=panel_fill,
        outline=line_rgb,
        width=max(1, int(render_params.panel_border_width_px)),
    )
    item_bboxes: Dict[str, Tuple[int, int, int, int]] = {}
    entities: list[Dict[str, Any]] = [
        {
            "entity_id": "music_staff_panel",
            "entity_type": "music_staff_panel",
            "bbox_px": [panel_x0, panel_y0, panel_x1, panel_y1],
        }
    ]

    staff_area_x0 = panel_x0 + int(render_params.panel_padding_px)
    staff_area_x1 = panel_x1 - int(render_params.panel_padding_px)
    first_staff_top = panel_y0 + 76
    content_x0_by_staff: Dict[int, int] = {}
    staff_top_by_index: Dict[int, int] = {}
    staff_line_bboxes: Dict[str, Tuple[int, int, int, int]] = {}
    option_grid_x0 = int(panel_x0 + max(0, (panel_width - option_grid_width) // 2)) if option_count else 0
    option_grid_y0 = int(panel_y0 + staff_content_height + int(render_params.option_gap_px)) if option_count else 0
    for staff_index, system in enumerate(spec.systems):
        staff_top = int(first_staff_top + staff_index * int(render_params.staff_spacing_px))
        staff_top_by_index[staff_index] = staff_top
        available_staff_width = max(320, int(staff_area_x1 - staff_area_x0))
        staff_draw_width = int(available_staff_width)
        staff_x0 = int(staff_area_x0 + max(0, (available_staff_width - staff_draw_width) // 2))
        staff_x1 = int(staff_x0 + staff_draw_width)
        staff_line_bboxes[str(staff_index)] = (
            int(staff_x0),
            int(staff_top),
            int(staff_x1),
            int(staff_top + 4 * int(render_params.staff_gap_px)),
        )
        for line_index in range(5):
            y = int(staff_top + line_index * int(render_params.staff_gap_px))
            draw.line((staff_x0, y, staff_x1, y), fill=line_rgb, width=2)
        clef_bbox = _draw_clef_symbol(
            draw,
            clef=str(system.clef),
            x=staff_x0 + 10,
            staff_top=staff_top,
            staff_gap=int(render_params.staff_gap_px),
            fill=text_rgb,
        )
        item_bboxes[f"staff_{staff_index}_clef"] = clef_bbox
        entities.append({
            "entity_id": f"staff_{staff_index}_clef",
            "entity_type": "music_clef",
            "bbox_px": list(clef_bbox),
            "clef": str(system.clef),
            "rendering": "vector",
        })
        cursor_x = max(staff_x0 + 55, int(clef_bbox[2]) + 14)
        if str(system.key_signature):
            key_bbox = _draw_key_signature(
                draw,
                text=str(system.key_signature),
                clef=str(system.clef),
                staff_top=int(staff_top),
                cursor_x=int(cursor_x),
                render_params=render_params,
                text_rgb=text_rgb,
            )
            if key_bbox is None:
                key_bbox = _expand_bbox_to_min_side(_draw_text_bbox(
                    draw,
                    (cursor_x, staff_top + int(render_params.staff_gap_px)),
                    str(system.key_signature),
                    font=marker_font,
                    fill=text_rgb,
                    stroke_fill=stroke_rgb,
                ))
            if str(system.key_signature_id):
                item_bboxes[str(system.key_signature_id)] = key_bbox
            entities.append({
                "entity_id": str(system.key_signature_id or f"staff_{staff_index}_key_signature"),
                "entity_type": "music_key_signature",
                "bbox_px": list(key_bbox),
                "text": str(system.key_signature),
                "tokens": list(_key_signature_tokens(str(system.key_signature))),
                "staff_steps": list(_key_signature_staff_steps(str(system.key_signature), str(system.clef))),
            })
            cursor_x = key_bbox[2] + 18
        if str(system.time_signature):
            stacked_width = _measure_stacked_time_signature_width(draw, str(system.time_signature), render_params=render_params)
            time_bbox = None
            if stacked_width is not None:
                time_bbox = _draw_stacked_time_signature(
                    draw,
                    text=str(system.time_signature),
                    center_x=int(cursor_x + stacked_width / 2),
                    staff_top=int(staff_top),
                    render_params=render_params,
                    text_rgb=text_rgb,
                )
            if time_bbox is None:
                time_bbox = _expand_bbox_to_min_side(
                    _draw_text_bbox(
                        draw,
                        (cursor_x, staff_top + int(render_params.staff_gap_px) // 2),
                        str(system.time_signature),
                        font=marker_font,
                        fill=text_rgb,
                        stroke_fill=stroke_rgb,
                    )
                )
            if str(system.time_signature_id):
                item_bboxes[str(system.time_signature_id)] = time_bbox
            parsed_time_signature = _parse_time_signature(str(system.time_signature))
            entity: dict[str, Any] = {
                "entity_id": str(system.time_signature_id or f"staff_{staff_index}_time_signature"),
                "entity_type": "music_time_signature",
                "bbox_px": list(time_bbox),
                "text": str(system.time_signature),
            }
            if parsed_time_signature is not None:
                entity["numerator"] = str(parsed_time_signature[0])
                entity["denominator"] = str(parsed_time_signature[1])
            entities.append(entity)
            cursor_x = time_bbox[2] + 26
        content_x0 = max(staff_x0 + 145, cursor_x + 12)
        content_x0_by_staff[staff_index] = int(content_x0)
        if str(system.subtitle):
            sub_bbox = _draw_text_bbox(
                draw,
                (staff_x0, staff_top - int(render_params.staff_gap_px) * 3),
                str(system.subtitle),
                font=small_font,
                fill=text_rgb,
                stroke_fill=stroke_rgb,
            )
            item_bboxes[f"staff_{staff_index}_subtitle"] = sub_bbox
        for barline in system.barlines:
            x = _slot_x(content_x0, int(render_params.slot_gap_px), float(barline.slot))
            bbox = (x - 2, staff_top, x + 2, staff_top + 4 * int(render_params.staff_gap_px))
            draw.line((x, bbox[1], x, bbox[3]), fill=line_rgb, width=2)
            item_bboxes[str(barline.item_id)] = bbox
            entities.append({"entity_id": str(barline.item_id), "entity_type": "music_barline", "bbox_px": list(bbox), "slot": float(barline.slot), "staff_index": int(staff_index)})
        for staff_range in system.ranges:
            x0 = _slot_x(content_x0, int(render_params.slot_gap_px), float(staff_range.start_slot))
            x1 = _slot_x(content_x0, int(render_params.slot_gap_px), float(staff_range.end_slot))
            bracket_y = int(staff_top - int(render_params.staff_gap_px) * 2 + int(staff_range.bracket_y_offset_px))
            bbox = (
                min(x0, x1),
                min(int(staff_top - 28), int(bracket_y), int(bracket_y + 9)),
                max(x0, x1),
                staff_top + 4 * int(render_params.staff_gap_px) + 18,
            )
            bracket_x0 = int(bbox[0])
            bracket_x1 = int(bbox[2])
            if bool(staff_range.show_bracket):
                draw.line((bracket_x0, bracket_y, bracket_x1, bracket_y), fill=mark_rgb, width=2)
                draw.line((bracket_x0, bracket_y, bracket_x0, bracket_y + 9), fill=mark_rgb, width=2)
                draw.line((bracket_x1, bracket_y, bracket_x1, bracket_y + 9), fill=mark_rgb, width=2)
            item_bboxes[str(staff_range.item_id)] = bbox
            if bool(staff_range.show_bracket) and str(staff_range.label):
                label_bbox_at_origin = draw.textbbox((0, 0), str(staff_range.label), font=marker_font, stroke_width=1)
                label_width = int(label_bbox_at_origin[2] - label_bbox_at_origin[0])
                label_height = int(label_bbox_at_origin[3] - label_bbox_at_origin[1])
                label_bbox = _draw_text_bbox(
                    draw,
                    (
                        int((bracket_x0 + bracket_x1 - label_width) / 2),
                        int(bracket_y - label_height - 9 + int(staff_range.label_y_offset_px)),
                    ),
                    str(staff_range.label),
                    font=marker_font,
                    fill=mark_rgb,
                    stroke_fill=stroke_rgb,
                )
                item_bboxes[f"{staff_range.item_id}_label"] = label_bbox
            entities.append({
                "entity_id": str(staff_range.item_id),
                "entity_type": "music_staff_range",
                "bbox_px": list(bbox),
                "label": str(staff_range.label),
                "show_bracket": bool(staff_range.show_bracket),
            })
        for time_signature in system.time_signatures:
            x = _slot_x(content_x0, int(render_params.slot_gap_px), float(time_signature.slot))
            bbox = _draw_stacked_time_signature(
                draw,
                text=str(time_signature.text),
                center_x=int(x),
                staff_top=int(staff_top),
                render_params=render_params,
                text_rgb=text_rgb,
            )
            if bbox is None:
                bbox = _expand_bbox_to_min_side(
                    _draw_text_bbox(
                        draw,
                        (int(x), staff_top + int(render_params.staff_gap_px) // 2),
                        str(time_signature.text),
                        font=marker_font,
                        fill=text_rgb,
                        stroke_fill=stroke_rgb,
                    )
                )
            item_bboxes[str(time_signature.item_id)] = bbox
            parsed_time_signature = _parse_time_signature(str(time_signature.text))
            entity: dict[str, Any] = {
                "entity_id": str(time_signature.item_id),
                "entity_type": "music_time_signature",
                "bbox_px": list(bbox),
                "text": str(time_signature.text),
                "slot": float(time_signature.slot),
                "staff_index": int(staff_index),
            }
            if parsed_time_signature is not None:
                entity["numerator"] = str(parsed_time_signature[0])
                entity["denominator"] = str(parsed_time_signature[1])
            entities.append(entity)
        for text_item in system.texts:
            x = _slot_x(content_x0, int(render_params.slot_gap_px), float(text_item.slot))
            y = int(staff_top + float(text_item.y_offset_steps) * int(render_params.staff_gap_px))
            bbox = _draw_text_bbox(
                draw,
                (x, y),
                str(text_item.text),
                font=label_font if bool(text_item.bold) else small_font,
                fill=mark_rgb if bool(text_item.bold) else text_rgb,
                stroke_fill=stroke_rgb,
            )
            item_bboxes[str(text_item.item_id)] = bbox
            entities.append({"entity_id": str(text_item.item_id), "entity_type": "music_staff_text", "bbox_px": list(bbox), "text": str(text_item.text)})
        for chord in system.chords:
            bbox = _draw_chord(
                draw,
                chord=chord,
                staff_top=staff_top,
                content_x0=content_x0,
                clef=str(system.clef),
                render_params=render_params,
                text_rgb=text_rgb,
                stroke_rgb=stroke_rgb,
                mark_rgb=mark_rgb,
                font=label_font,
            )
            item_bboxes[str(chord.item_id)] = bbox
            entities.append({"entity_id": str(chord.item_id), "entity_type": "music_chord", "bbox_px": list(bbox), "pitches": [format_pitch(p, include_octave=True) for p in chord.pitches], "staff_index": int(staff_index)})
            if str(chord.marker):
                marker_id, marker_bbox, marker_entity = _draw_item_marker(
                    draw,
                    item_id=str(chord.item_id),
                    target_bbox=bbox,
                    label=str(chord.marker),
                    staff_top=staff_top,
                    render_params=render_params,
                    mark_rgb=mark_rgb,
                    stroke_rgb=stroke_rgb,
                    font=marker_font,
                )
                item_bboxes[str(marker_id)] = marker_bbox
                entities.append(marker_entity)
        for note in system.notes:
            duration_style = _duration_visual_style(
                int(note.duration_units),
                filled=bool(note.filled),
                dotted=bool(note.dotted),
            )
            bbox = _draw_note(
                draw,
                note=note,
                staff_top=staff_top,
                content_x0=content_x0,
                clef=str(system.clef),
                render_params=render_params,
                text_rgb=text_rgb,
                stroke_rgb=stroke_rgb,
                mark_rgb=mark_rgb,
                font=label_font,
                small_font=small_font,
            )
            item_bboxes[str(note.item_id)] = bbox
            entities.append({
                "entity_id": str(note.item_id),
                "entity_type": "music_note",
                "bbox_px": list(bbox),
                "pitch": format_pitch(note.pitch, include_octave=True),
                "duration_units": int(note.duration_units),
                "duration_style": _duration_style_metadata(duration_style),
                "stem_up": bool(_stem_up_for_pitch(note.pitch, str(system.clef))),
                "staff_index": int(staff_index),
                "display_accidental": _note_display_accidental(note),
            })
            if str(note.marker):
                marker_id, marker_bbox, marker_entity = _draw_item_marker(
                    draw,
                    item_id=str(note.item_id),
                    target_bbox=bbox,
                    label=str(note.marker),
                    staff_top=staff_top,
                    render_params=render_params,
                    mark_rgb=mark_rgb,
                    stroke_rgb=stroke_rgb,
                    font=marker_font,
                )
                item_bboxes[str(marker_id)] = marker_bbox
                entities.append(marker_entity)
        for symbol in system.symbols:
            x = _slot_x(content_x0, int(render_params.slot_gap_px), float(symbol.slot))
            y = _staff_y_for_pitch(staff_top, int(render_params.staff_gap_px), symbol.pitch, str(system.clef))
            symbol_rgb = text_rgb
            if str(symbol.symbol) == "staccato":
                bbox = (x - 4, y - int(render_params.staff_gap_px) * 3, x + 4, y - int(render_params.staff_gap_px) * 3 + 8)
                draw.ellipse(bbox, fill=symbol_rgb)
            elif str(symbol.symbol) == "tenuto":
                bbox = (x - 15, y - int(render_params.staff_gap_px) * 3, x + 15, y - int(render_params.staff_gap_px) * 3 + 3)
                draw.rectangle(bbox, fill=symbol_rgb)
            elif str(symbol.symbol) == "accent":
                bbox = _draw_text_bbox(draw, (x - 10, y - int(render_params.staff_gap_px) * 4), ">", font=label_font, fill=symbol_rgb, stroke_fill=stroke_rgb)
            elif str(symbol.symbol) == "fermata":
                bbox = _draw_fermata_symbol(
                    draw,
                    x=x,
                    y=y - int(render_params.staff_gap_px) * 3,
                    fill=symbol_rgb,
                )
            else:
                bbox = _draw_text_bbox(draw, (x - 10, y - int(render_params.staff_gap_px) * 4), str(symbol.symbol), font=small_font, fill=symbol_rgb, stroke_fill=stroke_rgb)
            bbox = _expand_bbox_to_min_side(tuple(int(value) for value in bbox))
            item_bboxes[str(symbol.item_id)] = bbox
            entities.append({"entity_id": str(symbol.item_id), "entity_type": "music_articulation_symbol", "bbox_px": list(bbox), "symbol": str(symbol.symbol)})
            if str(symbol.marker):
                marker_id, marker_bbox, marker_entity = _draw_item_marker(
                    draw,
                    item_id=str(symbol.item_id),
                    target_bbox=bbox,
                    label=str(symbol.marker),
                    staff_top=staff_top,
                    render_params=render_params,
                    mark_rgb=mark_rgb,
                    stroke_rgb=stroke_rgb,
                    font=marker_font,
                )
                item_bboxes[str(marker_id)] = marker_bbox
                entities.append(marker_entity)

    for index, option in enumerate(spec.option_cards):
        row = int(index // option_columns)
        col = int(index % option_columns)
        x0 = int(option_grid_x0 + col * (int(render_params.option_card_width_px) + int(render_params.option_gap_px)))
        y0 = int(option_grid_y0 + row * (int(render_params.option_card_height_px) + int(render_params.option_gap_px)))
        bbox = (x0, y0, x0 + int(render_params.option_card_width_px), y0 + int(render_params.option_card_height_px))
        draw.rounded_rectangle(bbox, radius=10, fill=option_fill, outline=option_border, width=2)
        label_bbox = _draw_text_bbox(draw, (bbox[0] + 12, bbox[1] + 10), f"{option.label}.", font=label_font, fill=text_rgb, stroke_fill=stroke_rgb)
        content_parts = [bbox, label_bbox]
        if option.duration_units is not None:
            glyph_bbox = _draw_duration_glyph(
                draw,
                center=(bbox[0] + int(render_params.option_card_width_px) // 2 + 10, bbox[1] + int(render_params.option_card_height_px) // 2 + 7),
                duration_units=int(option.duration_units),
                render_params=render_params,
                text_rgb=text_rgb,
                stroke_rgb=stroke_rgb,
                font=small_font,
                stem_height_px=max(18, int(render_params.option_card_height_px) // 2 - 4),
            )
            content_parts.append(glyph_bbox)
        else:
            option_text = str(option.text)
            text_x0 = bbox[0] + 52
            max_text_width = max(48, int(bbox[2] - text_x0 - 12))
            max_text_height = max(24, int(render_params.option_card_height_px) - 16)
            text_font, text_lines, total_text_height = _fit_option_text(
                draw,
                option_text,
                max_width=int(max_text_width),
                max_height=int(max_text_height),
                base_font_size=max(int(render_params.label_font_size_px) + 3, 22),
                min_font_size=10,
            )
            line_y = int(bbox[1] + max(0, (int(render_params.option_card_height_px) - int(total_text_height)) // 2) - 2)
            for line in text_lines:
                content_parts.append(
                    _draw_text_bbox(
                        draw,
                        (int(text_x0), int(line_y)),
                        str(line),
                        font=text_font,
                        fill=text_rgb,
                        stroke_fill=stroke_rgb,
                        stroke_width=1,
                    )
                )
                line_bbox = draw.textbbox((0, 0), str(line), font=text_font, stroke_width=1)
                line_y += int(line_bbox[3] - line_bbox[1]) + 3
        item_bboxes[str(option.item_id)] = bbox
        entities.append({"entity_id": str(option.item_id), "entity_type": "music_option_card", "bbox_px": list(item_bboxes[str(option.item_id)]), "label": str(option.label), "text": str(option.text), "option_mode": "duration_glyph" if option.duration_units is not None else "text", "is_correct": bool(option.is_correct)})

    if str(spec.footer_text):
        footer_bbox = _draw_text_bbox(
            draw,
            (panel_x0 + int(render_params.panel_padding_px), panel_y1 - int(render_params.panel_padding_px)),
            str(spec.footer_text),
            font=small_font,
            fill=text_rgb,
            stroke_fill=stroke_rgb,
        )
        item_bboxes["footer_text"] = footer_bbox

    return RenderedMusicScene(
        image=image,
        scene_bbox_px=(panel_x0, panel_y0, panel_x1, panel_y1),
        item_bboxes={str(key): tuple(int(v) for v in value) for key, value in item_bboxes.items()},
        entities=tuple(entities),
        layout_jitter={
            "enabled": True,
            "panel_x0_px": int(panel_x0),
            "panel_y0_px": int(panel_y0),
            "panel_width_px": int(panel_width),
            "panel_height_px": int(panel_height),
            "available_x0_min_px": 24,
            "available_x0_max_px": int(max_x0),
            "available_y0_min_px": 24,
            "available_y0_max_px": int(max_y0),
        },
        style_metadata={
            "staff_line_rgb": list(line_rgb),
            "note_rgb": list(text_rgb),
            "mark_rgb": list(mark_rgb),
            "panel_fill_rgb": list(panel_fill),
            "option_fill_rgb": list(option_fill),
            "staff_draw_width_px": int(staff_draw_width),
            "staff_line_bboxes_px": {str(key): list(value) for key, value in staff_line_bboxes.items()},
        },
    )


__all__ = [
    "ACCIDENTAL_TEXT",
    "DEGREE_NAMES",
    "LETTERS",
    "MAJOR_SCALE_STEPS",
    "NATURAL_SEMITONES",
    "MusicRenderParams",
    "MusicSceneSpec",
    "OptionCard",
    "Pitch",
    "RenderedMusicScene",
    "StaffBarline",
    "StaffChord",
    "StaffNote",
    "StaffRange",
    "StaffSymbol",
    "StaffSystem",
    "StaffText",
    "StaffTimeSignature",
    "build_interval_pitch",
    "format_key_signature",
    "format_pitch",
    "interval_name",
    "make_staff_note_sequence",
    "major_scale_pitch",
    "pitch_from_staff_step",
    "pitch_midi",
    "render_music_scene",
    "resolve_music_render_params",
]
