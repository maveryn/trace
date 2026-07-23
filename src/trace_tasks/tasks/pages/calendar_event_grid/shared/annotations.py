"""Annotation helpers for calendar event-grid scene packages."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


def chip_key(day: int, slot_id: str) -> str:
    """Return the stable render key for one date-slot chip."""

    return f"date_{int(day)}__slot_{str(slot_id)}"


def round_box(box: Sequence[float]) -> list[float]:
    """Return a JSON-stable rounded bbox."""

    return [round(float(value), 3) for value in box]


def date_box(rendered: Any, day: int) -> list[float]:
    """Return one rounded date-cell bbox from rendered metadata."""

    return round_box(rendered.rendered_scene.date_cell_bboxes_by_day[int(day)])


def event_chip_box(rendered: Any, key: str) -> list[float]:
    """Return one rounded event-chip bbox from rendered metadata."""

    return round_box(rendered.rendered_scene.event_chip_bboxes_by_key[str(key)])


def date_chip_box_map(*, rendered: Any, day: int, slot_id: str) -> Dict[str, list[float]]:
    """Return keyed date-cell and event-chip boxes for one date-slot lookup."""

    key = chip_key(int(day), str(slot_id))
    return {
        "date_cell": date_box(rendered, int(day)),
        "event_chip": event_chip_box(rendered, key),
    }


def date_slot_event_chip_box(*, rendered: Any, day: int, slot_id: str) -> list[float]:
    """Return the event-chip bbox for one date-slot lookup."""

    return event_chip_box(rendered, chip_key(int(day), str(slot_id)))


def event_chip_box_set(*, rendered: Any, chip_keys: Sequence[str]) -> list[list[float]]:
    """Return matching event-chip bboxes in deterministic key order."""

    return [event_chip_box(rendered, str(key)) for key in chip_keys]


def event_chip_records(*, rendered: Any, chips: Sequence[Any]) -> list[Mapping[str, Any]]:
    """Return event chip trace records with final projected bboxes."""

    records: list[Mapping[str, Any]] = []
    for chip in chips:
        key = chip_key(int(chip.day), str(chip.slot_id))
        bbox = rendered.rendered_scene.event_chip_bboxes_by_key.get(str(key))
        records.append(
            {
                "chip_key": str(key),
                "date_number": int(chip.day),
                "slot_id": str(chip.slot_id),
                "slot_label": str(chip.slot_label),
                "category_label": str(chip.category_label),
                "bbox_px": round_box(bbox) if bbox is not None else None,
            }
        )
    return records
