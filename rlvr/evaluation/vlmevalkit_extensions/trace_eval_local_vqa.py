from __future__ import annotations

import os
import os.path as osp
import re
from pathlib import Path
from typing import Any

import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load
from .image_base import ImageBaseDataset


BOXED_POST_PROMPT = (
    "\nThink through the visual evidence carefully. "
    "Put only the final answer inside \\boxed{}."
)
COUNTQA_BOXED_POST_PROMPT = "\nPut the final answer inside \\boxed{}."


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


def _extract_final_text(text: Any) -> str:
    text = str(text or "").strip()
    boxed = _last_boxed(text)
    if boxed:
        return boxed.strip()
    answer_tag = re.search(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    if answer_tag:
        return answer_tag.group(1).strip()
    json_match = re.search(r"\{.*\"answer\"\s*:\s*(.+?)\}", text, flags=re.S)
    if json_match:
        return json_match.group(1).strip().strip("\"'")
    matches = list(re.finditer(r"\b(?:final\s+answer|answer|choice|option)\b\s*[:：]?\s*(.+)", text, flags=re.I | re.S))
    if matches:
        return matches[-1].group(1).strip().splitlines()[0].strip()
    return text


def _extract_last_int(text: Any) -> int | None:
    text = _extract_final_text(text).lower()
    matches = re.findall(r"[-+]?\d+", text)
    if matches:
        return int(matches[-1])
    words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }
    found = None
    for word, value in words.items():
        if re.search(rf"\b{word}\b", text):
            found = value
    return found


def _extract_choice(text: Any) -> str:
    text = _extract_final_text(text)
    text = re.sub(r"^(answer|option|choice)\s*[:：-]?\s*", "", text, flags=re.I).strip()
    letter = re.search(r"\b([A-M])\b", text, flags=re.I)
    if letter:
        return letter.group(1).upper()
    number = re.search(r"\b([0-9]+)\b", text)
    if number:
        return str(int(number.group(1)))
    return text.strip()


def _save_image(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    image = image.convert("RGB") if hasattr(image, "convert") else image
    image.save(path)


def _load_cached_with_images(data_path: Path) -> pd.DataFrame | None:
    frame = load(str(data_path))
    if "image_path" not in frame:
        return frame
    image_paths = [Path(str(path)) for path in frame["image_path"].dropna().unique()]
    missing = [path for path in image_paths if not path.exists()]
    if missing:
        print(
            f"[trace_eval_local_vqa] rebuilding {data_path.name}: "
            f"{len(missing)} cached image paths are missing"
        )
        return None
    return frame


class CountQA(ImageBaseDataset):
    TYPE = "VQA"
    DATASET_URL = {"CountQA": ""}
    DATASET_MD5 = {}
    HF_DATASET = "Jayant-Sravan/CountQA"
    HF_SPLIT = "test"

    def load_data(self, dataset):
        root = Path(LMUDataRoot())
        data_path = root / f"{dataset}.tsv"
        if data_path.exists() and not os.environ.get("TRACE_FORCE_REBUILD_LOCAL_VLMEVAL"):
            cached = _load_cached_with_images(data_path)
            if cached is not None:
                # Fresh Hugging Face rows use answer strings, while pandas can
                # infer integers when the cached TSV is reloaded. Normalizing
                # only the representation keeps preparation hashes invariant.
                cached = cached.copy()
                cached["answer"] = [
                    "" if pd.isna(value) else str(value).strip()
                    for value in cached["answer"]
                ]
                return cached

        from datasets import load_dataset

        hf = load_dataset(self.HF_DATASET, split=self.HF_SPLIT, token=os.environ.get("HF_TOKEN"))
        image_root = root / "images" / dataset
        rows = []
        for source_idx, example in enumerate(hf):
            image_path = image_root / f"{source_idx}.jpg"
            _save_image(example["image"], image_path)
            questions = example.get("questions") or []
            answers = example.get("answers") or []
            for qa_idx, question in enumerate(questions):
                answer = answers[qa_idx] if qa_idx < len(answers) else ""
                rows.append(
                    {
                        "index": f"{source_idx}_{qa_idx}",
                        "question": str(question).strip(),
                        "answer": str(answer).strip(),
                        "image_path": str(image_path),
                        "objects": str(example.get("objects", "")),
                        "categories": str(example.get("categories", "")),
                        "is_focused": bool(example.get("is_focused", False)),
                        "full_config": str(example.get("full_config", "")),
                    }
                )
        frame = pd.DataFrame(rows)
        dump(frame, str(data_path))
        return frame

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]["type"] == "text"
        msgs[-1]["value"] = str(msgs[-1]["value"]).strip() + COUNTQA_BOXED_POST_PROMPT
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        del judge_kwargs
        data = load(eval_file)
        correct = []
        pred_ints = []
        gt_ints = []
        for _, row in data.iterrows():
            pred = _extract_last_int(row.get("prediction", ""))
            gt = _extract_last_int(row.get("answer", ""))
            pred_ints.append(pred)
            gt_ints.append(gt)
            correct.append(float(pred is not None and gt is not None and pred == gt))
        data["eval_pred"] = pred_ints
        data["eval_gt"] = gt_ints
        data["eval_score"] = correct
        detailed = get_intermediate_file_path(eval_file, "_results")
        dump(data, detailed)
        result = pd.DataFrame([{"split": "Overall", "tot": len(data), "hit": sum(correct), "acc": sum(correct) / len(correct) * 100 if correct else 0.0}])
        dump(result, get_intermediate_file_path(eval_file, "_acc", "csv"))
        return result
