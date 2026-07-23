"""Library illustration scene with labeled shelf sections and book metadata."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from ....shared.color_format import format_named_color_with_hex
from ....shared.named_colors import named_color
from ....shared.text_rendering import fit_font_to_box
from ....shared.text_legibility import draw_text_traced
from ...shared.object_library import (
    BBox,
    RGB,
    STYLE_IDS,
    choose_object_colors,
)
from ...shared.object_rendering import (
    IllustrationObjectSpec,
    RenderContext,
    render_illustration_object,
)
from ...shared.object_variants import RENDERER_STYLE_VECTOR
from ...shared.person_rendering import sample_person_gender
from ...shared.render_geometry import scale_bbox as _scale_bbox, scale_points as _scale_points
from .state import (
    LIBRARY_SECTION_LABELS,
    LIBRARY_SETTING_IDS,
    LibraryBook,
    LibraryBookSpec,
    LibraryDecor,
    LibrarySection,
    LibrarySectionSpec,
    RenderedLibraryScene,
    library_section_display_name,
)




def _jitter_rgb(rng, color: RGB, amount: int = 14) -> RGB:
    return tuple(max(0, min(255, int(channel) + int(rng.randint(-int(amount), int(amount))))) for channel in color)  # type: ignore[return-value]


def _choose_weighted(rng, weights: Mapping[str, float], support: Sequence[str]) -> str:
    choices = [(str(value), max(0.0, float(weights.get(str(value), 0.0)))) for value in support]
    total = sum(weight for _value, weight in choices)
    if total <= 0.0:
        return str(rng.choice(tuple(support)))
    threshold = float(rng.random()) * float(total)
    running = 0.0
    for value, weight in choices:
        running += float(weight)
        if running >= threshold:
            return str(value)
    return str(choices[-1][0])


def _draw_fit_text(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    bbox: Sequence[float],
    scale: int,
    fill: RGB = (37, 42, 50),
    bold: bool = True,
    font_family: str | None = None,
    min_size_px: int = 8,
    max_size_px: int = 30,
    stroke_fill: RGB | None = None,
) -> None:
    """Draw a section label centered inside a fixed box while preserving legibility."""

    x0, y0, x1, y1 = [float(value) * int(scale) for value in bbox]
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=max(1.0, float(x1 - x0)),
        max_height=max(1.0, float(y1 - y0)),
        bold=bool(bold),
        font_family=font_family,
        min_size_px=int(min_size_px) * int(scale),
        max_size_px=int(max_size_px) * int(scale),
        fill_ratio=0.84,
    )
    try:
        text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(scale) if stroke_fill else 0))
        text_w = float(text_bbox[2] - text_bbox[0])
        text_h = float(text_bbox[3] - text_bbox[1])
    except Exception:
        text_w, text_h = draw.textsize(str(text), font=font)
    tx = float(x0 + (x1 - x0 - text_w) * 0.5)
    ty = float(y0 + (y1 - y0 - text_h) * 0.5)
    draw_text_traced(draw,
        (int(round(tx)), int(round(ty))),
        str(text),
        font=font,
        fill=tuple(fill),
        stroke_width=max(0, int(scale) if stroke_fill else 0),
        stroke_fill=tuple(stroke_fill) if stroke_fill else None,
     role="readout", required=False,)


def _readable_text_color(fill: RGB) -> RGB:
    luminance = 0.299 * float(fill[0]) + 0.587 * float(fill[1]) + 0.114 * float(fill[2])
    return (245, 244, 236) if luminance < 132.0 else (36, 42, 50)


def _sample_section_label_font_trace(*, instance_seed: int | None, params: Mapping[str, Any] | None) -> Dict[str, Any]:
    font_family = sample_font_family(
        role="readout",
        instance_seed=0 if instance_seed is None else int(instance_seed),
        namespace="illustrations.library.section_labels",
        params=params,
        explicit_key="library_section_label_font_family",
        weights_key="library_section_label_font_family_weights",
    )
    record = get_font_family_record(str(font_family))
    return {
        **record.to_trace(),
        "font_asset_version": font_asset_version(),
        "pool": "global_approved_font_pool",
        "role": "library_section_label",
        "consistent_scope": "library_section_labels",
    }


def _draw_background(draw: ImageDraw.ImageDraw, *, rng, setting_id: str, width: int, height: int, scale: int) -> None:
    palettes = {
        "reading_room": ((232, 224, 209), (185, 154, 118), (122, 82, 53)),
        "archive_room": ((222, 226, 218), (164, 142, 112), (97, 84, 70)),
        "childrens_corner": ((231, 229, 211), (180, 164, 122), (105, 118, 152)),
    }
    wall_base, floor_base, accent_base = palettes.get(str(setting_id), palettes["reading_room"])
    wall = _jitter_rgb(rng, wall_base, amount=8)
    floor = _jitter_rgb(rng, floor_base, amount=12)
    accent = _jitter_rgb(rng, accent_base, amount=10)
    s = int(scale)
    draw.rectangle((0, 0, int(width) * s, int(height) * s), fill=tuple(wall))
    floor_y = int(round(float(height) * float(rng.uniform(0.70, 0.74)) * s))
    draw.rectangle((0, floor_y, int(width) * s, int(height) * s), fill=tuple(floor))
    draw.line([(0, floor_y), (int(width) * s, floor_y)], fill=tuple(_jitter_rgb(rng, (115, 102, 88), amount=8)), width=max(1, 2 * s))
    for index in range(int(rng.randint(1, 3))):
        window_w = float(rng.uniform(120.0, 175.0))
        window_h = float(rng.uniform(82.0, 118.0))
        window_x = float(rng.uniform(58.0, max(60.0, float(width) - window_w - 60.0)))
        window_y = float(rng.uniform(42.0, 96.0))
        box = (window_x, window_y, window_x + window_w, window_y + window_h)
        draw.rounded_rectangle(_scale_bbox(box, s), radius=max(1, 8 * s), fill=(202, 224, 236), outline=tuple(accent), width=max(1, 3 * s))
        sx0, sy0, sx1, sy1 = _scale_bbox(box, s)
        draw.line([((sx0 + sx1) // 2, sy0), ((sx0 + sx1) // 2, sy1)], fill=tuple(accent), width=max(1, 2 * s))
        draw.line([(sx0, (sy0 + sy1) // 2), (sx1, (sy0 + sy1) // 2)], fill=tuple(accent), width=max(1, 2 * s))
        if index == 0 and str(setting_id) == "childrens_corner":
            poster = (window_x + window_w + 22.0, window_y + 8.0, min(float(width) - 48.0, window_x + window_w + 138.0), window_y + window_h - 6.0)
            draw.rounded_rectangle(_scale_bbox(poster, s), radius=max(1, 8 * s), fill=tuple(_jitter_rgb(rng, (237, 188, 92), amount=10)), outline=tuple(accent), width=max(1, 2 * s))


def _section_layouts(rng, *, width: int, height: int, specs: Sequence[LibrarySectionSpec]) -> Tuple[Dict[str, BBox], Dict[str, Any]]:
    count = len(specs)
    cols = 2 if count <= 4 else 3
    rows = max(1, (count + cols - 1) // cols)
    margin_x = float(rng.uniform(48.0, 66.0))
    gap_x = float(rng.uniform(22.0, 34.0))
    gap_y = float(rng.uniform(20.0, 30.0))
    top = float(rng.uniform(134.0, 158.0))
    bottom = float(height) * float(rng.uniform(0.66, 0.69))
    unit_w = (float(width) - 2.0 * margin_x - float(cols - 1) * gap_x) / float(cols)
    unit_h = (bottom - top - float(rows - 1) * gap_y) / float(rows)
    y_shift = float(rng.uniform(-8.0, 8.0))
    boxes: Dict[str, BBox] = {}
    for index, section_spec in enumerate(specs):
        row = index // cols
        col = index % cols
        x0 = margin_x + float(col) * (unit_w + gap_x) + float(rng.uniform(-4.0, 4.0))
        y0 = top + y_shift + float(row) * (unit_h + gap_y) + float(rng.uniform(-5.0, 5.0))
        x1 = x0 + unit_w + float(rng.uniform(-8.0, 8.0))
        y1 = y0 + unit_h + float(rng.uniform(-8.0, 8.0))
        boxes[str(section_spec.section_key)] = (round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3))
    layout = {
        "layout_mode": "wall_grid",
        "section_count": count,
        "columns": cols,
        "rows": rows,
        "shelf_area_bbox": [round(margin_x, 3), round(top, 3), round(float(width) - margin_x, 3), round(bottom, 3)],
    }
    return boxes, layout


def _draw_section_shell(
    draw: ImageDraw.ImageDraw,
    *,
    rng,
    section: LibrarySectionSpec,
    bbox: BBox,
    row_count: int,
    scale: int,
    section_label_font_family: str,
) -> Tuple[BBox, Tuple[BBox, ...], RGB, RGB]:
    """Draw one labeled shelf section and return its label box plus book shelf rows."""

    x0, y0, x1, y1 = [float(v) for v in bbox]
    wood = _jitter_rgb(rng, rng.choice(((139, 91, 58), (151, 103, 70), (121, 92, 70), (162, 118, 78))), amount=14)
    wood_dark = _jitter_rgb(rng, (82, 61, 44), amount=8)
    label_fill = _jitter_rgb(rng, rng.choice(((220, 192, 123), (189, 149, 102), (118, 139, 159), (143, 127, 166))), amount=12)
    s = int(scale)
    draw.rounded_rectangle(_scale_bbox((x0, y0, x1, y1), s), radius=max(1, 8 * s), fill=tuple(wood), outline=tuple(wood_dark), width=max(1, 3 * s))
    inner = (x0 + 10.0, y0 + 10.0, x1 - 10.0, y1 - 10.0)
    draw.rounded_rectangle(_scale_bbox(inner, s), radius=max(1, 5 * s), fill=tuple(_jitter_rgb(rng, (226, 210, 177), amount=10)), outline=tuple(wood_dark), width=max(1, 2 * s))
    label_h = max(25.0, min(34.0, (y1 - y0) * 0.14))
    label_bbox = (x0 + 18.0, y0 + 13.0, x1 - 18.0, y0 + 13.0 + label_h)
    draw.rounded_rectangle(_scale_bbox(label_bbox, s), radius=max(1, 5 * s), fill=tuple(label_fill), outline=tuple(wood_dark), width=max(1, 2 * s))
    _draw_fit_text(
        draw,
        text=LIBRARY_SECTION_LABELS.get(str(section.section_key), str(section.section_key).upper()),
        bbox=label_bbox,
        scale=s,
        fill=_readable_text_color(label_fill),
        bold=True,
        font_family=str(section_label_font_family),
        min_size_px=8,
        max_size_px=24,
        stroke_fill=(36, 42, 50) if _readable_text_color(label_fill) == (245, 244, 236) else None,
    )
    book_area_top = label_bbox[3] + 9.0
    book_area_bottom = y1 - 16.0
    shelf_rows: List[BBox] = []
    row_h = max(38.0, (book_area_bottom - book_area_top) / float(max(1, row_count)))
    for row in range(row_count):
        row_top = book_area_top + float(row) * row_h
        row_bottom = min(book_area_bottom, row_top + row_h - 4.0)
        shelf_line_y = row_bottom
        draw.rectangle(
            _scale_bbox((x0 + 15.0, shelf_line_y - 7.0, x1 - 15.0, shelf_line_y), s),
            fill=tuple(wood_dark),
        )
        shelf_rows.append((x0 + 20.0, row_top + 2.0, x1 - 20.0, shelf_line_y - 8.0))
    return label_bbox, tuple(shelf_rows), wood, wood_dark


def _draw_book(draw: ImageDraw.ImageDraw, *, rng, spec: LibraryBookSpec, bbox: BBox, scale: int) -> RGB:
    color_rgb = tuple(int(channel) for channel in named_color(str(spec.color_name)))
    x0, y0, x1, y1 = [float(v) for v in bbox]
    s = int(scale)
    outline = (45, 48, 55)
    page = (244, 239, 217)
    if str(spec.orientation) == "horizontal":
        draw.rounded_rectangle(_scale_bbox((x0, y0, x1, y1), s), radius=max(1, 2 * s), fill=tuple(color_rgb), outline=tuple(outline), width=max(1, s))
        draw.rectangle(_scale_bbox((x0 + 0.12 * (x1 - x0), y0 + 0.26 * (y1 - y0), x1 - 0.10 * (x1 - x0), y0 + 0.46 * (y1 - y0)), s), fill=page)
        draw.line(_scale_points(((x0 + 0.16 * (x1 - x0), y1 - 0.24 * (y1 - y0)), (x1 - 0.12 * (x1 - x0), y1 - 0.24 * (y1 - y0))), s), fill=tuple(outline), width=max(1, s))
    else:
        radius = max(1, 2 * s)
        draw.rounded_rectangle(_scale_bbox((x0, y0, x1, y1), s), radius=radius, fill=tuple(color_rgb), outline=tuple(outline), width=max(1, s))
        stripe_w = max(2.0, min(5.0, 0.24 * (x1 - x0)))
        stripe_x = x0 + float(rng.uniform(0.16, 0.58)) * (x1 - x0)
        draw.rectangle(_scale_bbox((stripe_x, y0 + 4.0, min(x1 - 2.0, stripe_x + stripe_w), y1 - 4.0), s), fill=page)
        if y1 - y0 > 44.0:
            draw.line(_scale_points(((x0 + 3.0, y0 + 0.70 * (y1 - y0)), (x1 - 3.0, y0 + 0.70 * (y1 - y0))), s), fill=tuple(outline), width=max(1, s))
    return color_rgb


def _book_row_groups(rng, specs: Sequence[LibraryBookSpec], row_count: int) -> List[List[LibraryBookSpec]]:
    ordered = list(specs)
    rng.shuffle(ordered)
    groups: List[List[LibraryBookSpec]] = [[] for _ in range(max(1, int(row_count)))]
    for index, spec in enumerate(ordered):
        groups[index % len(groups)].append(spec)
    return groups


def _draw_section_books(
    draw: ImageDraw.ImageDraw,
    *,
    rng,
    section: LibrarySectionSpec,
    section_id: str,
    shelf_rows: Sequence[BBox],
    scale: int,
) -> Tuple[Tuple[LibraryBook, ...], Tuple[str, ...]]:
    """Draw all books assigned to one section and preserve exact semantic ids."""

    rows = _book_row_groups(rng, section.book_specs, len(shelf_rows))
    rendered: List[LibraryBook] = []
    book_ids: List[str] = []
    book_counter = 0
    for row_index, row_specs in enumerate(rows):
        if not row_specs:
            continue
        rx0, ry0, rx1, ry1 = [float(v) for v in shelf_rows[int(row_index)]]
        slot_w = max(12.0, (rx1 - rx0) / float(len(row_specs)))
        row_h = max(20.0, ry1 - ry0)
        for slot_index, spec in enumerate(row_specs):
            cx = rx0 + (float(slot_index) + 0.5) * slot_w + float(rng.uniform(-0.10, 0.10)) * slot_w
            if str(spec.orientation) == "horizontal":
                book_w = min(max(30.0, slot_w * float(rng.uniform(0.68, 0.92))), max(34.0, slot_w - 3.0))
                book_h = min(max(14.0, row_h * float(rng.uniform(0.34, 0.48))), min(28.0, row_h - 4.0))
                y1 = ry1 - float(rng.uniform(2.0, 5.0))
                y0 = y1 - book_h
            else:
                book_w = min(max(9.0, slot_w * float(rng.uniform(0.36, 0.58))), 22.0)
                book_h = min(row_h - 2.0, max(28.0, row_h * float(rng.uniform(0.62, 0.94))))
                y1 = ry1 - float(rng.uniform(1.0, 4.0))
                y0 = y1 - book_h
            x0 = max(rx0, min(rx1 - book_w, cx - 0.5 * book_w))
            box = (round(x0, 3), round(max(ry0, y0), 3), round(x0 + book_w, 3), round(y1, 3))
            color_rgb = _draw_book(draw, rng=rng, spec=spec, bbox=box, scale=scale)
            book_id = f"{section_id}_book_{book_counter:02d}"
            book_counter += 1
            book_ids.append(book_id)
            rendered.append(
                LibraryBook(
                    book_id=book_id,
                    section_id=str(section_id),
                    section_key=str(section.section_key),
                    section_name=library_section_display_name(str(section.section_key)),
                    color_name=str(spec.color_name),
                    color_rgb=tuple(int(v) for v in color_rgb),
                    color_label=format_named_color_with_hex(str(spec.color_name), color_rgb),
                    orientation=str(spec.orientation),
                    bbox_xyxy=tuple(float(v) for v in box),
                    row_index=int(row_index),
                    slot_index=int(slot_index),
                    role=str(spec.role),
                    attributes=dict(spec.attributes),
                )
            )
    return tuple(rendered), tuple(book_ids)


def _draw_foreground_decor(draw: ImageDraw.ImageDraw, *, rng, width: int, height: int, scale: int, style_id: str) -> Tuple[LibraryDecor, ...]:
    """Draw non-query foreground decor below shelves without creating book witnesses."""

    decor: List[LibraryDecor] = []
    s = int(scale)
    table_count = int(rng.randint(1, 3))
    for index in range(table_count):
        table_w = float(rng.uniform(210.0, 310.0))
        table_h = float(rng.uniform(62.0, 82.0))
        x0 = float(rng.uniform(74.0, max(76.0, float(width) - table_w - 74.0)))
        y0 = float(rng.uniform(float(height) * 0.73, float(height) * 0.84))
        table_box = (x0, y0, x0 + table_w, y0 + table_h)
        fill = _jitter_rgb(rng, rng.choice(((166, 113, 71), (148, 103, 73), (178, 128, 82))), amount=12)
        outline = _jitter_rgb(rng, (80, 61, 47), amount=8)
        draw.rounded_rectangle(_scale_bbox(table_box, s), radius=max(1, 14 * s), fill=tuple(fill), outline=tuple(outline), width=max(1, 3 * s))
        for leg_x in (x0 + 0.16 * table_w, x0 + 0.78 * table_w):
            draw.rectangle(_scale_bbox((leg_x, y0 + table_h - 4.0, leg_x + 16.0, min(float(height) - 28.0, y0 + table_h + 64.0)), s), fill=tuple(outline))
        decor.append(LibraryDecor(f"decor_table_{index}", "reading_table", tuple(round(v, 3) for v in table_box), {"role": "distractor"}))
        for item_index in range(int(rng.randint(1, 3))):
            bx0 = x0 + float(rng.uniform(18.0, max(20.0, table_w - 70.0)))
            by0 = y0 - float(rng.uniform(16.0, 28.0))
            book_box = (bx0, by0, bx0 + float(rng.uniform(36.0, 58.0)), by0 + float(rng.uniform(10.0, 16.0)))
            book_color = tuple(int(v) for v in rng.choice(tuple(named_color(name) for name in ("red", "blue", "green", "orange", "purple"))))
            draw.rounded_rectangle(_scale_bbox(book_box, s), radius=max(1, 2 * s), fill=book_color, outline=(44, 48, 55), width=max(1, s))
            decor.append(LibraryDecor(f"decor_desk_book_{index}_{item_index}", "desk_book_distractor", tuple(round(v, 3) for v in book_box), {"role": "not_a_section_book"}))
    for index in range(int(rng.randint(2, 5))):
        object_type = str(rng.choice(("person", "pedestrian_with_bag", "potted_plant", "lamp")))
        if object_type in {"person", "pedestrian_with_bag"}:
            h = float(rng.uniform(92.0, 132.0))
            w = h * float(rng.uniform(0.48, 0.62))
            y1 = float(rng.uniform(float(height) * 0.82, float(height) - 26.0))
        else:
            h = float(rng.uniform(64.0, 98.0))
            w = h * float(rng.uniform(0.55, 0.82))
            y1 = float(rng.uniform(float(height) * 0.76, float(height) - 34.0))
        x0 = float(rng.uniform(36.0, max(38.0, float(width) - w - 36.0)))
        box = (x0, y1 - h, x0 + w, y1)
        primary, accent = choose_object_colors(rng, object_type)
        visual_attributes: dict[str, Any] = {
            "primary_color_rgb": primary,
            "accent_color_rgb": accent,
            "style_id": str(style_id),
        }
        gender_id = sample_person_gender(rng) if object_type in {"person", "pedestrian_with_bag"} else None
        if gender_id is not None:
            visual_attributes["gender_id"] = gender_id
        rendered = render_illustration_object(
            IllustrationObjectSpec(
                object_id=f"decor_object_{index}",
                object_type=object_type,
                bbox_xyxy=box,
                semantic_attributes={"decor_type": object_type},
                visual_attributes=visual_attributes,
                role="distractor",
                source_entity_type="library_decor",
            ),
            RenderContext(
                renderer_style=RENDERER_STYLE_VECTOR,
                draw=draw,
                render_scale=s,
            ),
        )
        attributes = {"role": "distractor"}
        if object_type in {"person", "pedestrian_with_bag"}:
            attributes["gender_id"] = str(rendered.visual_attributes.get("gender_id", "male"))
        decor.append(
            LibraryDecor(
                f"decor_object_{index}",
                object_type,
                tuple(float(v) for v in rendered.bbox_xyxy),
                attributes,
                object_record=rendered.object_record,
            )
        )
    return tuple(decor)


def render_library_scene(
    *,
    rng,
    section_specs: Sequence[LibrarySectionSpec],
    canvas_width: int = 1280,
    canvas_height: int = 900,
    render_scale: int = 2,
    setting_weights: Mapping[str, float] | None = None,
    style_weights: Mapping[str, float] | None = None,
    instance_seed: int | None = None,
    font_params: Mapping[str, Any] | None = None,
) -> RenderedLibraryScene:
    """Render one synthetic library scene from semantic section/book specs."""

    width = int(canvas_width)
    height = int(canvas_height)
    scale = max(1, int(render_scale))
    if not section_specs:
        raise ValueError("library scene needs at least one section")
    setting_id = _choose_weighted(rng, setting_weights or {setting: 1.0 for setting in LIBRARY_SETTING_IDS}, LIBRARY_SETTING_IDS)
    style_id = _choose_weighted(rng, style_weights or {style: 1.0 for style in STYLE_IDS}, STYLE_IDS)
    section_label_font = _sample_section_label_font_trace(instance_seed=instance_seed, params=font_params)
    section_boxes, layout = _section_layouts(rng, width=width, height=height, specs=section_specs)
    layout["section_label_font"] = dict(section_label_font)

    image = Image.new("RGB", (width * scale, height * scale), (245, 244, 238))
    draw = ImageDraw.Draw(image)
    _draw_background(draw, rng=rng, setting_id=str(setting_id), width=width, height=height, scale=scale)

    sections: List[LibrarySection] = []
    books: List[LibraryBook] = []
    for index, section_spec in enumerate(section_specs):
        section_id = f"section_{index:02d}"
        book_count = len(section_spec.book_specs)
        row_count = 2 if book_count <= 10 else 3
        if book_count >= 13:
            row_count = 3
        label_bbox, shelf_rows, _wood, _wood_dark = _draw_section_shell(
            draw,
            rng=rng,
            section=section_spec,
            bbox=section_boxes[str(section_spec.section_key)],
            row_count=row_count,
            scale=scale,
            section_label_font_family=str(section_label_font["font_family"]),
        )
        section_books, book_ids = _draw_section_books(
            draw,
            rng=rng,
            section=section_spec,
            section_id=section_id,
            shelf_rows=shelf_rows,
            scale=scale,
        )
        books.extend(section_books)
        sections.append(
            LibrarySection(
                section_id=section_id,
                section_key=str(section_spec.section_key),
                section_name=library_section_display_name(str(section_spec.section_key)),
                label=LIBRARY_SECTION_LABELS.get(str(section_spec.section_key), str(section_spec.section_key).upper()),
                bbox_xyxy=tuple(float(v) for v in section_boxes[str(section_spec.section_key)]),
                label_bbox_xyxy=tuple(float(v) for v in label_bbox),
                shelf_bboxes_xyxy=tuple(tuple(float(v) for v in row) for row in shelf_rows),
                book_ids=tuple(book_ids),
                role=str(section_spec.role),
                attributes={**dict(section_spec.attributes), "row_count": int(row_count), "label_font": dict(section_label_font)},
            )
        )
    decor = _draw_foreground_decor(draw, rng=rng, width=width, height=height, scale=scale, style_id=str(style_id))

    if scale != 1:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return RenderedLibraryScene(
        image=image,
        setting_id=str(setting_id),
        sections=tuple(sections),
        books=tuple(books),
        decor=tuple(decor),
        canvas_width=width,
        canvas_height=height,
        render_scale=scale,
        style_id=str(style_id),
        layout=dict(layout),
    )


__all__ = [
    "render_library_scene",
]
