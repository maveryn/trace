import json
from pathlib import Path
from typing import Any

import yaml


CHART_CONFIG_DIR = Path("src/trace_tasks/resources/configs/domains/charts")
CHART_PROMPT_DIR = Path("src/trace_tasks/resources/prompts/charts")

_ALLOWED_SINGLE_CHAR_EXAMPLES = {
    ("dashboard.yaml", "json_example_answer_only_option_letter"),
    ("dashboard.yaml", "json_example_statement_option"),
    ("region_map.yaml", "json_example_marker_region_extremum_label"),
    ("region_map.yaml", "json_example_answer_only_marker_region_extremum_label"),
    ("scientific.yaml", "json_example_cross_panel_delta_extremum_label"),
    ("scientific.yaml", "json_example_earliest_maximum_panel_label"),
    ("scientific.yaml", "json_example_cross_panel_threshold_earliest_label"),
    ("scientific.yaml", "json_example_answer_only_cross_panel_delta_extremum_label"),
    ("scientific.yaml", "json_example_answer_only_earliest_maximum_panel_label"),
    ("scientific.yaml", "json_example_answer_only_cross_panel_threshold_earliest_label"),
    ("scatter_cluster.yaml", "json_example_answer_only_option_letter"),
    ("scatter_cluster.yaml", "json_example_centroid_option_selection_label"),
    ("scatter_points.yaml", "json_example_answer_only_option_letter"),
    ("scatter_points.yaml", "json_example_centroid_option_selection_label"),
    ("scatter_readout.yaml", "json_example_answer_only_option_letter"),
    ("scatter_readout.yaml", "json_example_centroid_option_selection_label"),
}


def _iter_json_example_strings(node: Any) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            key_text = str(key)
            if "json_example" in key_text and isinstance(value, str):
                found.append((key_text, value))
            found.extend(_iter_json_example_strings(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_iter_json_example_strings(value))
    return found


def _iter_string_values(node: Any, *, path: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            child_path = f"{path}.{key}" if path else str(key)
            found.extend(_iter_string_values(value, path=child_path))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_iter_string_values(value, path=f"{path}[{index}]"))
    elif isinstance(node, str):
        found.append((path, node))
    return found


def test_chart_label_json_examples_use_multi_character_labels_when_supported() -> None:
    offenders: list[str] = []
    for config_path in sorted(CHART_CONFIG_DIR.glob("*.yaml")):
        data = yaml.safe_load(config_path.read_text()) or {}
        for key, raw_example in _iter_json_example_strings(data):
            try:
                parsed = json.loads(raw_example)
            except json.JSONDecodeError:
                continue
            answer = parsed.get("answer") if isinstance(parsed, dict) else None
            if not (isinstance(answer, str) and len(answer) == 1):
                continue
            if (config_path.name, key) not in _ALLOWED_SINGLE_CHAR_EXAMPLES:
                offenders.append(f"{config_path}:{key} uses one-character example answer {answer!r}")
    assert not offenders, "\n".join(offenders)


def test_chart_interval_prompt_templates_name_endpoint_slots() -> None:
    offenders: list[str] = []
    vague_endpoint_phrases = (
        "Using the endpoint values",
        "Using the two endpoint categories",
        "Using the interval endpoints",
        "over this interval",
        "first endpoint to the second endpoint",
        "across the displayed interval",
        "Using the two highlighted axes",
    )
    for prompt_path in sorted(CHART_PROMPT_DIR.glob("**/*.json")):
        data = json.loads(prompt_path.read_text())
        for key_path, template in _iter_string_values(data):
            if not key_path.startswith("query_templates."):
                continue
            if not any(phrase in template for phrase in vague_endpoint_phrases):
                continue
            has_endpoint_slots = any(
                slot in template
                for slot in (
                    "{start_label}",
                    "{end_label}",
                    "{start_category}",
                    "{end_category}",
                    "{start_category_label}",
                    "{end_category_label}",
                    "{query_year_start}",
                    "{query_year_end}",
                    "{query_interval_label}",
                    "{axis_i}",
                    "{axis_j}",
                )
            )
            if not has_endpoint_slots:
                offenders.append(f"{prompt_path}:{key_path} omits explicit endpoint/range slots: {template}")
    assert not offenders, "\n".join(offenders)
