"""Annotation helpers for signal-transform scenes."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence


def normalize_signal_transform_annotation_bbox_map(
    annotation_bbox_map: Mapping[str, Sequence[float]],
) -> Dict[str, List[float]]:
    """Normalize role-bound bbox annotation for prompt and verifier output."""

    required = ("input_waveform", "selected_spectrum")
    missing = [key for key in required if key not in annotation_bbox_map]
    if missing:
        raise ValueError(f"missing signal-transform annotation keys: {missing}")
    return {
        str(key): [round(float(value), 3) for value in annotation_bbox_map[str(key)]]
        for key in required
    }
