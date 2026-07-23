"""Shared render-parameter resolver for GUI-style page scenes."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, TypeVar

from ...shared.render_variation import resolve_render_int


_RenderParamsT = TypeVar("_RenderParamsT")

GUI_WINDOW_RENDER_PARAM_FIELDS = (
    "canvas_width",
    "canvas_height",
    "window_margin_px",
    "title_bar_height_px",
    "menu_bar_height_px",
    "corner_radius_px",
    "control_corner_radius_px",
    "control_outline_width_px",
    "badge_size_px",
    "title_font_size_px",
    "body_font_size_px",
    "small_font_size_px",
    "label_font_size_px",
)


def resolve_gui_window_render_params(
    params: Mapping[str, Any],
    *,
    defaults: Any,
    render_defaults: Mapping[str, Any],
    task_id: str,
    render_params_cls: type[_RenderParamsT],
    instance_seed: int | None = None,
) -> _RenderParamsT:
    """Resolve the common GUI-window render params into a task-local dataclass."""

    values = asdict(defaults) if is_dataclass(defaults) else dict(defaults)

    def int_value(key: str) -> int:
        return resolve_render_int(
            params,
            render_defaults,
            str(key),
            int(values[str(key)]),
            instance_seed=instance_seed,
            namespace=str(task_id),
        )

    return render_params_cls(**{key: int_value(str(key)) for key in GUI_WINDOW_RENDER_PARAM_FIELDS})


__all__ = ["GUI_WINDOW_RENDER_PARAM_FIELDS", "resolve_gui_window_render_params"]
