"""Regression tests for graph scene default config loading."""

from __future__ import annotations

from pathlib import Path

import yaml

from trace_tasks.core.scene_config import get_scene_defaults, resolve_scene_section_defaults
from trace_tasks.tasks.registry import TASK_REGISTRY, ensure_scene_tasks_registered


GRAPH_SCENES = (
    "adjacency",
    "automaton",
    "binary_tree",
    "flow_network",
    "graph_options",
    "metro",
    "node_link",
    "pedigree_chart",
    "phylogeny_tree",
    "pipe_network",
)

RETIRED_GRAPH_CONFIG_MARKERS = (
    "graph_node_link_source_sink_count_internal",
    "source_sink_count_query",
    "source_sink_mode_weights",
    "task_graph__metro__transfer_count",
    "metro_transfer_count_query",
)


def _active_graph_task_ids() -> set[str]:
    for scene_id in GRAPH_SCENES:
        ensure_scene_tasks_registered("graph", scene_id)
    return {
        str(task_id)
        for task_id in dict.keys(TASK_REGISTRY)
        if str(task_id).startswith("task_graph__")
    }


def _active_scene_task_ids(scene_id: str) -> set[str]:
    prefix = f"task_graph__{scene_id}__"
    return {task_id for task_id in _active_graph_task_ids() if task_id.startswith(prefix)}


def _load_graph_scene_config(scene_id: str) -> dict:
    path = Path("src/trace_tasks/resources/configs/domains/graph") / f"{scene_id}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def test_graph_scene_task_override_keys_match_active_tasks() -> None:
    """Scene YAML task overrides should not keep retired public/internal ids."""

    active_ids = _active_graph_task_ids()
    assert len(active_ids) == 60

    for scene_id in GRAPH_SCENES:
        scene_active_ids = _active_scene_task_ids(scene_id)
        cfg = _load_graph_scene_config(scene_id)
        for section_name in ("generation", "rendering", "prompt"):
            section = cfg.get(section_name, {})
            overrides = section.get("task_overrides", {}) if isinstance(section, dict) else {}
            assert isinstance(overrides, dict)
            for task_id in overrides:
                assert task_id in scene_active_ids, f"{scene_id}.{section_name} has stale override {task_id}"


def test_graph_node_link_config_has_no_retired_internal_blocks() -> None:
    text = Path("src/trace_tasks/resources/configs/domains/graph/node_link.yaml").read_text(encoding="utf-8")
    for marker in RETIRED_GRAPH_CONFIG_MARKERS:
        assert marker not in text


def test_all_active_graph_task_defaults_resolve() -> None:
    """Every active graph task should resolve generation/rendering/prompt defaults from its scene."""

    for task_id in sorted(_active_graph_task_ids()):
        scene_id = task_id.split("__")[1]
        cfg = get_scene_defaults("graph", scene_id)
        generation = resolve_scene_section_defaults(cfg, "generation", task_id=task_id)
        rendering = resolve_scene_section_defaults(cfg, "rendering", task_id=task_id)
        prompt = resolve_scene_section_defaults(cfg, "prompt", task_id=task_id)

        assert isinstance(generation, dict)
        assert isinstance(rendering, dict)
        assert isinstance(prompt, dict)
        assert "query_id_weights" not in generation
        assert "balanced_query_id_sampling" not in generation


def test_graph_node_link_representative_prompt_defaults_are_active() -> None:
    cfg = get_scene_defaults("graph", "node_link")

    degree_prompt = resolve_scene_section_defaults(
        cfg,
        "prompt",
        task_id="task_graph__node_link__degree_value_filter_count",
    )
    assert degree_prompt["bundle_id"] == "graph_node_link_counting_v1"
    assert degree_prompt["scene_key"] == "single_graph_counting"
    assert degree_prompt["task_key"] == "degree_count_query"

    edge_text_generation = resolve_scene_section_defaults(
        cfg,
        "generation",
        task_id="task_graph__node_link__edge_text_count",
    )
    edge_text_rendering = resolve_scene_section_defaults(
        cfg,
        "rendering",
        task_id="task_graph__node_link__edge_text_count",
    )
    edge_text_prompt = resolve_scene_section_defaults(
        cfg,
        "prompt",
        task_id="task_graph__node_link__edge_text_count",
    )
    assert int(edge_text_generation["node_count_max"]) <= 8
    assert int(edge_text_generation["max_labeled_edge_count"]) <= 12
    assert int(edge_text_rendering["edge_text_label_font_size_px"]) >= 20
    assert edge_text_prompt["task_key"] == "edge_text_label_count_query"


def test_graph_scene_prompt_bundles_resolve_for_migrated_scenes() -> None:
    expected_bundle_by_scene = {
        "adjacency": "graph_adjacency_v1",
        "automaton": "automaton_v1",
        "binary_tree": "graph_binary_tree_v1",
        "flow_network": "graph_flow_network_v1",
        "graph_options": "graph_options_v1",
        "metro": "graph_metro_v1",
        "pedigree_chart": "graph_pedigree_chart_v1",
        "phylogeny_tree": "graph_phylogeny_tree_v1",
        "pipe_network": "graph_pipe_network_v1",
    }

    for scene_id, bundle_id in expected_bundle_by_scene.items():
        cfg = get_scene_defaults("graph", scene_id)
        prompt = resolve_scene_section_defaults(cfg, "prompt")
        assert prompt["bundle_id"] == bundle_id
