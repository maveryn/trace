"""Shared below-scene text option panels for 3D object-selection tasks."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ...shared.named_colors import available_named_colors
from ...shared.text_legibility import draw_text_traced, text_legibility_metadata_for_surfaces
from ...shared.text_rendering import load_font


PROMPT_COLOR_RGB_BY_NAME: Dict[str, Tuple[int, int, int]] = {
    str(name): (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    for name, rgb in available_named_colors()
}


def _clean_name(value: Any) -> str:
    text = str(value or "").replace("_", " ").strip().lower()
    return " ".join(text.split())


def _candidate_name(spec: Mapping[str, Any]) -> str:
    for key in ("prompt_name", "object_name", "object_type", "shape_type"):
        name = _clean_name(spec.get(str(key)))
        if name:
            return str(name)
    return "object"


def prompt_color_name_for_rgb(rgb: Sequence[Any]) -> str:
    """Return the closest simple prompt color name for an RGB triplet."""

    if len(rgb) < 3:
        raise ValueError("RGB color requires at least three channels")
    target = tuple(max(0, min(255, int(channel))) for channel in rgb[:3])
    best_name = next(iter(PROMPT_COLOR_RGB_BY_NAME), "red")
    best_distance = float("inf")
    for name, color in PROMPT_COLOR_RGB_BY_NAME.items():
        distance = math.sqrt(
            sum((float(channel) - float(target[index])) ** 2 for index, channel in enumerate(color))
        )
        if distance < best_distance:
            best_name = str(name)
            best_distance = float(distance)
    return str(best_name)


def _descriptor_color_name(spec: Mapping[str, Any]) -> str | None:
    explicit = _clean_name(spec.get("option_color_name") or spec.get("prompt_color_name"))
    if explicit:
        return str(explicit)
    for key in ("option_color_rgb", "robot_base_rgb", "fill_rgb", "base_rgb"):
        value = spec.get(str(key))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
            return prompt_color_name_for_rgb(value)
    return None


def _candidate_label(spec: Mapping[str, Any]) -> str:
    return str(spec.get("point_label") or spec.get("object_label") or "").strip().upper()


def assign_independent_prompt_colors(
    candidate_specs: Sequence[Mapping[str, Any]],
    *,
    rng,
) -> List[Dict[str, Any]]:
    """Assign non-semantic option colors independent of geometry/answer slots.

    The assignment order is the already-randomized option label, not support
    placement, depth rank, relation status, or construction order. This keeps
    prompt colors useful for disambiguating option descriptors without letting
    a task-specific slot become a color shortcut.
    """

    specs = [dict(spec) for spec in candidate_specs]
    color_items = list(PROMPT_COLOR_RGB_BY_NAME.items())
    rng.shuffle(color_items)
    label_order = sorted(range(len(specs)), key=lambda index: (_candidate_label(specs[index]), str(index)))
    for color_index, spec_index in enumerate(label_order):
        wrapped_color_index = int(color_index)
        while wrapped_color_index >= len(color_items):
            wrapped_color_index -= len(color_items)
        color_name, color_rgb = color_items[wrapped_color_index]
        specs[spec_index]["fill_rgb"] = [int(channel) for channel in color_rgb]
        specs[spec_index]["option_color_name"] = str(color_name)
        specs[spec_index]["option_color_rgb"] = [int(channel) for channel in color_rgb]
        specs[spec_index]["color_assignment_policy"] = "independent_prompt_color_by_option_label"
    return specs


def apply_independent_prompt_colors_to_dataset(
    dataset: Mapping[str, Any],
    *,
    rng,
    candidate_key: str = "point_specs",
) -> Dict[str, Any]:
    """Return a dataset copy with independent prompt colors on candidates."""

    updated: Dict[str, Any] = dict(dataset)
    candidates = updated.get(str(candidate_key), [])
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError(f"dataset candidate key {candidate_key!r} must contain a sequence")
    colored_candidates = assign_independent_prompt_colors(candidates, rng=rng)
    by_object_id = {str(spec["object_id"]): dict(spec) for spec in colored_candidates if spec.get("object_id") is not None}
    updated[str(candidate_key)] = list(colored_candidates)

    def merge_specs(specs: Any) -> Any:
        if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)):
            return specs
        merged = []
        for spec in specs:
            if not isinstance(spec, Mapping):
                merged.append(spec)
                continue
            object_id = str(spec.get("object_id", ""))
            replacement = by_object_id.get(object_id)
            if replacement is None:
                merged.append(dict(spec))
                continue
            merged_spec: MutableMapping[str, Any] = dict(spec)
            merged_spec.update(replacement)
            merged.append(dict(merged_spec))
        return merged

    for key in ("point_specs", "candidate_specs", "candidate_object_specs", "object_specs"):
        if str(key) in updated and str(key) != str(candidate_key):
            updated[str(key)] = merge_specs(updated[str(key)])
    return updated


def build_text_option_choices(candidate_specs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Build unique below-scene option descriptors for candidate objects.

    Names are used by default. Color is added only for candidate name collisions.
    """

    specs = [dict(spec) for spec in candidate_specs]
    if not specs:
        return []
    names_by_id = {str(spec["object_id"]): _candidate_name(spec) for spec in specs}
    name_counts = Counter(names_by_id.values())
    choices: List[Dict[str, Any]] = []
    for spec in specs:
        label = _candidate_label(spec)
        if not label:
            raise ValueError("candidate option spec is missing point_label/object_label")
        object_id = str(spec["object_id"])
        name = str(names_by_id[object_id])
        color_name = None
        descriptor = str(name)
        if int(name_counts[name]) > 1:
            color_name = _descriptor_color_name(spec)
            if not color_name:
                raise ValueError(f"candidate option descriptor collision for {name!r} without prompt color")
            descriptor = f"{color_name} {name}"
        choices.append(
            {
                "label": str(label),
                "option_id": f"option_{label}",
                "object_id": str(object_id),
                "descriptor": str(descriptor),
                "object_name": str(name),
                "color_name": color_name,
            }
        )
    descriptor_counts = Counter(str(choice["descriptor"]) for choice in choices)
    duplicates = sorted(descriptor for descriptor, count in descriptor_counts.items() if int(count) > 1)
    if duplicates:
        raise ValueError(f"candidate option descriptors are not unique: {duplicates}")
    return sorted(choices, key=lambda choice: str(choice["label"]))


def empty_option_panel_metadata() -> Dict[str, Any]:
    return {
        "option_panel_bbox_px": [],
        "option_choice_bboxes_px": {},
        "option_choices": [],
        "option_panel_height_px": 0,
    }


def append_text_option_panel(
    image: Image.Image,
    *,
    option_choices: Sequence[Mapping[str, Any]],
    font_size_px: int,
    text_rgb: Sequence[int] = (24, 28, 36),
    stroke_rgb: Sequence[int] = (255, 255, 255),
) -> Tuple[Image.Image, Dict[str, Any], List[Dict[str, Any]]]:
    """Append a text option panel below the scene image."""

    choices = [dict(choice) for choice in option_choices]
    if not choices:
        return image, empty_option_panel_metadata(), []

    width, height = image.size
    columns = 2 if len(choices) > 3 else 1
    rows = int(math.ceil(float(len(choices)) / float(columns)))
    font = load_font(int(font_size_px), bold=True)
    row_height = max(42, int(round(float(font_size_px) * 1.62)))
    panel_height = max(116, 24 + rows * row_height + 18)
    panel_top = int(height)
    panel_bbox = [0.0, float(panel_top), float(width), float(panel_top + panel_height)]

    panel_fill_rgb = (246, 248, 250)
    out = Image.new("RGB", (int(width), int(height + panel_height)), panel_fill_rgb)
    out.paste(image.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, panel_top, width, panel_top + panel_height), fill=panel_fill_rgb)
    draw.line((0, panel_top, width, panel_top), fill=(165, 174, 184), width=2)

    label_fill = (31, 41, 55)
    label_text_fill = (255, 255, 255)
    # The option panel owns its light surface, so descriptor ink must be
    # resolved here rather than inherited from potentially dark scene styles.
    text_fill = (24, 28, 36)
    text_stroke = tuple(max(0, min(255, int(channel))) for channel in stroke_rgb[:3])
    margin_x = 42
    col_gap = 28
    col_width = (int(width) - margin_x * 2 - col_gap * (columns - 1)) / float(columns)
    option_bboxes: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = [
        {
            "entity_id": "three_d_option_panel",
            "entity_type": "three_d_option_panel",
            "bbox_px": list(panel_bbox),
            "attrs": {"option_count": int(len(choices)), "columns": int(columns), "rows": int(rows)},
        }
    ]

    for index, choice in enumerate(choices):
        row = int(index // columns)
        col = int(index % columns)
        x0 = float(margin_x + col * (col_width + col_gap))
        y0 = float(panel_top + 18 + row * row_height)
        x1 = float(x0 + col_width)
        y1 = float(y0 + row_height - 8)
        label = str(choice["label"])
        descriptor = str(choice["descriptor"])
        badge_size = min(30, max(24, int(round(float(font_size_px) * 1.18))))
        badge_bbox = [x0, y0 + 4.0, x0 + badge_size, y0 + 4.0 + badge_size]
        draw.rounded_rectangle(
            tuple(badge_bbox),
            radius=5,
            fill=label_fill,
            outline=(14, 19, 30),
            width=1,
        )
        badge_center = (
            float(badge_bbox[0] + badge_size * 0.5),
            float(badge_bbox[1] + badge_size * 0.5),
        )
        label_bbox = draw.textbbox(badge_center, label, font=font, stroke_width=0, anchor="mm")
        label_center = (
            float(badge_center[0] * 2.0 - (float(label_bbox[0]) + float(label_bbox[2])) * 0.5),
            float(badge_center[1] * 2.0 - (float(label_bbox[1]) + float(label_bbox[3])) * 0.5),
        )
        draw_text_traced(
            draw,
            label_center,
            label,
            font=font,
            fill=label_text_fill,
            anchor="mm",
            role="three_d_option_label",
            required=True,
            extra_metadata={
                **text_legibility_metadata_for_surfaces(
                    fill_rgb=label_text_fill,
                    surface_rgbs=(label_fill,),
                ),
                "option_label": str(label),
                "badge_bbox_px": [round(float(value), 3) for value in badge_bbox],
            },
        )
        text_xy = (float(x0 + badge_size + 11), float(y0 + 6))
        draw_text_traced(
            draw,
            text_xy,
            descriptor,
            font=font,
            fill=text_fill,
            stroke_fill=text_stroke,
            stroke_width=0,
            role="three_d_option_descriptor",
            required=True,
            extra_metadata=text_legibility_metadata_for_surfaces(
                fill_rgb=text_fill,
                surface_rgbs=(panel_fill_rgb,),
            ),
        )
        text_bbox = draw.textbbox(text_xy, descriptor, font=font, stroke_width=0)
        option_bbox = [
            round(float(x0), 3),
            round(float(min(badge_bbox[1], text_bbox[1])) - 3.0, 3),
            round(float(max(x1, text_bbox[2])) + 2.0, 3),
            round(float(max(badge_bbox[3], text_bbox[3])) + 4.0, 3),
        ]
        option_bboxes[str(label)] = list(option_bbox)
        entities.append(
            {
                "entity_id": f"three_d_option_{label}",
                "entity_type": "three_d_option_choice",
                "bbox_px": list(option_bbox),
                "attrs": {
                    "label": str(label),
                    "option_id": str(choice["option_id"]),
                    "object_id": str(choice["object_id"]),
                    "descriptor": str(descriptor),
                    "object_name": str(choice["object_name"]),
                    "color_name": choice.get("color_name"),
                },
            }
        )

    metadata = {
        "option_panel_bbox_px": list(panel_bbox),
        "option_choice_bboxes_px": dict(option_bboxes),
        "option_choices": [dict(choice) for choice in choices],
        "option_panel_height_px": int(panel_height),
    }
    return out, metadata, entities


__all__ = [
    "apply_independent_prompt_colors_to_dataset",
    "assign_independent_prompt_colors",
    "append_text_option_panel",
    "build_text_option_choices",
    "empty_option_panel_metadata",
    "prompt_color_name_for_rgb",
]
