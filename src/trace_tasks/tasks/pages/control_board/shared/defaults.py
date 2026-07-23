"""Defaults, palettes, and semantic resources for control-board pages."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.pages.shared.visual_defaults import (
    load_pages_scene_noise_defaults,
)
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

from .state import CommandOption, ControlBoardDefaults, ControlBoardProfile


DOMAIN = "pages"
SCENE = "control_board"
PROMPT_BUNDLE = "pages_control_board_v1"
PROMPT_SCENE_KEY = "control_board"
PROMPT_TASK_KEY = "control_board_count_query"
NAMESPACE_ROOT = "pages.control_board"

DISABLED_MODE = "disabled"
SELECTED_ENABLED_MODE = "selected_enabled"

SCENE_VARIANTS: Tuple[str, ...] = (
    "office_document",
    "creative_workspace",
    "developer_ide",
    "cad_workspace",
    "scientific_plotter",
    "os_file_manager",
)

APP_PROFILES: Dict[str, ControlBoardProfile] = {
    "office_document": ControlBoardProfile("Document Studio", "Quarterly Review", "Home", "Review", "Control Center", "Edited now"),
    "creative_workspace": ControlBoardProfile("Canvas Lab", "Campaign Layout", "Design", "Assets", "Tool Controls", "RGB / 100%"),
    "developer_ide": ControlBoardProfile("Code Desk", "trace_app.py", "Build", "Debug", "Command Palette", "main / clean"),
    "cad_workspace": ControlBoardProfile("Model Works", "Bracket Assembly", "Sketch", "Inspect", "Model Tools", "Units: mm"),
    "scientific_plotter": ControlBoardProfile("Lab Plot", "Sensor Run 18", "Analyze", "Plot", "Analysis Tools", "Sample 2.4k"),
    "os_file_manager": ControlBoardProfile("File Center", "Research Folder", "Files", "View", "Folder Actions", "23 items"),
}

COMMAND_OPTIONS: Tuple[CommandOption, ...] = (
    CommandOption("open_panel", "Open", "folder"),
    CommandOption("save_file", "Save", "save"),
    CommandOption("sync_now", "Sync", "sync"),
    CommandOption("share_item", "Share", "share"),
    CommandOption("search_view", "Search", "search"),
    CommandOption("filter_rows", "Filter", "filter"),
    CommandOption("sort_list", "Sort", "sort"),
    CommandOption("copy_item", "Copy", "copy"),
    CommandOption("paste_item", "Paste", "paste"),
    CommandOption("group_items", "Group", "group"),
    CommandOption("align_left", "Align", "align"),
    CommandOption("crop_item", "Crop", "crop"),
    CommandOption("rotate_item", "Rotate", "rotate"),
    CommandOption("measure_item", "Measure", "measure"),
    CommandOption("zoom_fit", "Fit", "zoom"),
    CommandOption("preview_item", "Preview", "preview"),
    CommandOption("validate_item", "Check", "check"),
    CommandOption("export_item", "Export", "export"),
    CommandOption("print_item", "Print", "print"),
    CommandOption("lock_item", "Lock", "lock"),
    CommandOption("bookmark_item", "Bookmark", "bookmark"),
    CommandOption("comment_item", "Comment", "comment"),
    CommandOption("settings_item", "Settings", "settings"),
    CommandOption("history_item", "History", "history"),
    CommandOption("run_action", "Run", "run"),
    CommandOption("inspect_item", "Inspect", "inspect"),
)

DEFAULTS = ControlBoardDefaults()
SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE)
GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
)
RENDER_FALLBACKS: Dict[str, Any] = asdict(DEFAULTS)
POST_IMAGE_NOISE_DEFAULTS = load_pages_scene_noise_defaults(scene_id=SCENE, apply_prob=0.5)
