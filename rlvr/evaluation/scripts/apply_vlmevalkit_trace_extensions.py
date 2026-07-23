#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VLMEVAL_ROOT = REPO_ROOT / "external" / "VLMEvalKit"
EXT_ROOT = REPO_ROOT / "rlvr" / "evaluation" / "vlmevalkit_extensions"


def insert_once(text: str, needle: str, line: str) -> str:
    if line in text:
        return text
    if needle not in text:
        raise RuntimeError(f"Could not find insertion anchor: {needle}")
    return text.replace(needle, f"{needle}\n{line}", 1)


def replace_once(text: str, old: str, new: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Could not find replacement anchor: {old}")
    return text.replace(old, new, 1)


def insert_symbol_once(text: str, anchor: str, symbol: str, present: str) -> str:
    if present in text:
        return text
    if anchor not in text:
        raise RuntimeError(f"Could not find insertion anchor: {anchor}")
    return text.replace(anchor, f"{anchor} {symbol},", 1)


def patch_tablevqabench_answer_wrapper(text: str) -> str:
    prefix, marker, table = text.partition("class TableVQABench")
    if not marker:
        raise RuntimeError("Could not find TableVQABench in image_vqa.py")
    table = replace_once(
        table,
        """        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data

        data['prediction'] = data['prediction'].str.replace('^Answer: ',
                                                            '',
                                                            regex=True)
""",
        """        from .utils.trace_eval_answer_parsing import unwrap_single_answer_block

        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data

        data['prediction'] = data['prediction'].map(unwrap_single_answer_block)
        data['prediction'] = data['prediction'].str.replace('^Answer: ',
                                                            '',
                                                            regex=True)
""",
    )
    return prefix + marker + table


def patch_puzzle_answer_parser(text: str) -> str:
    import_line = "from .trace_eval_answer_parsing import extract_unambiguous_abcd"
    if import_line not in text:
        anchors = ("from vlmeval.smp import load", "from vlmeval.smp.file import load")
        anchor = next((candidate for candidate in anchors if candidate in text), None)
        if anchor is None:
            raise RuntimeError("Could not find puzzle parser import anchor")
        text = insert_once(text, anchor, import_line)
    return replace_once(
        text,
        """def extract_answer(ans):
    matches = re.findall(r"\\banswer\\s*:\\s*([A-Z])\\b", ans, re.IGNORECASE)
    if matches:
        ans = matches[-1]
        return ans
    else:
        return "Z"
""",
        """def extract_answer(ans):
    return extract_unambiguous_abcd(ans)
""",
    )


def apply_extensions(vlmeval_root: Path) -> None:
    dataset_root = vlmeval_root / "vlmeval" / "dataset"
    if not dataset_root.exists():
        raise RuntimeError(
            f"VLMEvalKit dataset directory does not exist: {dataset_root}"
        )
    scripts_root = vlmeval_root / "scripts"
    if not scripts_root.exists():
        raise RuntimeError(
            f"VLMEvalKit scripts directory does not exist: {scripts_root}"
        )

    # These are the only local dataset extensions used by trace_eval_v1.
    for name in ("trace_eval_local_vqa.py", "evochart.py"):
        shutil.copy2(EXT_ROOT / name, dataset_root / name)
    shutil.copy2(
        EXT_ROOT / "trace_eval_answer_parsing.py",
        dataset_root / "utils" / "trace_eval_answer_parsing.py",
    )
    shutil.copy2(
        EXT_ROOT / "batched_vlmevalkit_qwen3vl.py",
        scripts_root / "batched_vlmevalkit_qwen3vl.py",
    )

    image_vqa_path = dataset_root / "image_vqa.py"
    if image_vqa_path.exists():
        image_vqa_path.write_text(
            patch_tablevqabench_answer_wrapper(image_vqa_path.read_text())
        )

    for name in ("puzzlevqa.py", "visualpuzzles.py"):
        path = dataset_root / "utils" / name
        path.write_text(patch_puzzle_answer_parser(path.read_text()))

    init_path = dataset_root / "__init__.py"
    text = init_path.read_text()
    text = insert_once(
        text,
        "from .text_mcq import CustomTextMCQDataset, TextMCQDataset",
        "from .trace_eval_local_vqa import CountQA",
    )
    text = insert_once(
        text,
        "from .erqabench import ERQABench",
        "from .evochart import EvoChart",
    )
    text = insert_symbol_once(
        text,
        "ChartQAPro,",
        "EvoChart",
        "ChartQAPro, EvoChart,",
    )
    text = insert_symbol_once(
        text,
        "ReasonMap_Plus,",
        "CountQA",
        "ReasonMap_Plus, CountQA,",
    )
    init_path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install Trace local VLMEvalKit dataset extensions."
    )
    parser.add_argument("--vlmeval-root", type=Path, default=DEFAULT_VLMEVAL_ROOT)
    args = parser.parse_args()
    apply_extensions(args.vlmeval_root.resolve())
    print(f"Installed Trace VLMEvalKit extensions into {args.vlmeval_root}")


if __name__ == "__main__":
    main()
