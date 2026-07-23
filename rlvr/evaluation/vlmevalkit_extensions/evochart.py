from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load
from .image_base import ImageBaseDataset


_NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\d*\.\d+"
)
BOXED_POST_PROMPT = "\nPut the final answer inside \\boxed{}."


def _last_boxed(text: str) -> str | None:
    start = 0
    last = None
    while True:
        idx = text.find("\\boxed{", start)
        if idx < 0:
            return last
        i = idx + len("\\boxed{")
        depth = 1
        j = i
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            last = text[i:j - 1].strip()
            start = j
        else:
            return last


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_final_answer(value: Any) -> str:
    text = _coerce_text(value)
    boxed = _last_boxed(text)
    if boxed:
        return boxed.strip()
    answer_tag = re.search(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    if answer_tag:
        return answer_tag.group(1).strip()
    json_match = re.search(r"\{.*\"answer\"\s*:\s*(.+?)\}", text, flags=re.S)
    if json_match:
        return json_match.group(1).strip().strip("\"'")
    matches = list(
        re.finditer(
            r"\b(?:final\s+answer|answer)\b\s*(?:is|=|:|：)?\s*(.+)",
            text,
            flags=re.I | re.S,
        )
    )
    if matches:
        return matches[-1].group(1).strip().splitlines()[0].strip()
    return text


def _parse_whole_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = _coerce_text(value)
    match = _NUMBER_PATTERN.fullmatch(text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _extract_single_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    matches = _NUMBER_PATTERN.findall(_coerce_text(value))
    if len(matches) != 1:
        return None
    try:
        return float(matches[0].replace(",", ""))
    except ValueError:
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return bool(value)


def _save_image(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    image.save(path)


def _load_cached_with_images(data_path: Path) -> pd.DataFrame | None:
    frame = load(str(data_path))
    if "image_path" not in frame:
        return frame
    image_paths = [Path(str(path)) for path in frame["image_path"].dropna().unique()]
    missing = [path for path in image_paths if not path.exists()]
    if missing:
        print(
            f"[evochart] rebuilding {data_path.name}: "
            f"{len(missing)} cached image paths are missing"
        )
        return None
    return frame


def score_prediction(prediction: Any, answer: Any, is_clear: Any) -> float:
    final_answer = _extract_final_answer(prediction)
    target = _coerce_text(answer)
    target_number = _parse_whole_number(target)
    if target_number is None:
        return float(final_answer.casefold() == target.casefold())

    prediction_number = _extract_single_number(final_answer)
    if prediction_number is None:
        return 0.0
    if _as_bool(is_clear):
        return float(prediction_number == target_number)
    if target_number == 0.0:
        return float(prediction_number == 0.0)
    return float(abs(prediction_number - target_number) / abs(target_number) <= 0.05)


class EvoChart(ImageBaseDataset):
    TYPE = "VQA"
    DATASET_URL = {
        "EvoChart": "",
        "EvoChart_boxed": "",
        "EvoChart_Qwen25_ZS": "",
        "EvoChart_Qwen3_ZS": "",
        "EvoChart_reasoning": "",
    }
    DATASET_MD5 = {}
    HF_DATASET = "gsarch/EvoChart-QA"
    HF_SPLIT = "train"

    def load_data(self, dataset):
        root = Path(LMUDataRoot())
        data_path = root / f"{dataset}.tsv"
        if data_path.exists() and not os.environ.get("TRACE_FORCE_REBUILD_LOCAL_VLMEVAL"):
            cached = _load_cached_with_images(data_path)
            if cached is not None:
                return cached

        from datasets import load_dataset

        hf = load_dataset(self.HF_DATASET, split=self.HF_SPLIT, token=os.environ.get("HF_TOKEN"))
        image_root = root / "images" / dataset
        rows = []
        for idx, example in enumerate(hf):
            image_path = image_root / f"{idx}.png"
            _save_image(example["image"], image_path)
            rows.append(
                {
                    "index": str(idx),
                    "question": str(example.get("question", "")).strip(),
                    "answer": str(example.get("answer", "")).strip(),
                    "image_path": str(image_path),
                    "attribute": str(example.get("attribute", "")),
                    "is_clear": bool(example.get("is_clear", True)),
                    "chart_type": str(example.get("chart_type", "")),
                }
            )
        frame = pd.DataFrame(rows)
        dump(frame, str(data_path))
        return frame

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]["type"] == "text"
        question = str(msgs[-1]["value"]).strip()
        if self.dataset_name in {"EvoChart", "EvoChart_boxed"}:
            question += BOXED_POST_PROMPT
        elif self.dataset_name == "EvoChart_Qwen25_ZS":
            question += "\nAnswer the question with a single word."
        elif self.dataset_name == "EvoChart_Qwen3_ZS":
            question += "\nAnswer the question using a single word or phrase."
        msgs[-1]["value"] = question
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        del judge_kwargs
        data = load(eval_file)
        scores = []
        extracted = []
        for _, row in data.iterrows():
            final_answer = _extract_final_answer(row.get("prediction", ""))
            extracted.append(final_answer)
            scores.append(score_prediction(final_answer, row.get("answer", ""), row.get("is_clear", True)))
        data["eval_pred"] = extracted
        data["eval_score"] = scores
        dump(data, get_intermediate_file_path(eval_file, "_results"))

        rows = [
            {"split": "Overall", "tot": len(data), "hit": sum(scores), "acc": sum(scores) / len(scores) * 100 if scores else 0.0}
        ]
        for field in ("chart_type", "attribute"):
            if field not in data:
                continue
            for name, group in data.groupby(field, dropna=False):
                vals = [float(x) for x in group["eval_score"].tolist()]
                rows.append(
                    {
                        "split": f"{field}:{name}",
                        "tot": len(vals),
                        "hit": sum(vals),
                        "acc": sum(vals) / len(vals) * 100 if vals else 0.0,
                    }
                )
        result = pd.DataFrame(rows)
        dump(result, get_intermediate_file_path(eval_file, "_acc", "csv"))
        return result
