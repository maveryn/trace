"""Convert TRACE parquet shards to verl's RL dataset format.

    python -m trace_tool.rl.prepare_verl_data --split train --agent tool_agent --out data/verl/train_tool_agent.parquet
    python -m trace_tool.rl.prepare_verl_data --split validation --agent single_turn_agent --out data/verl/val_single_turn_agent.parquet

(scripts/rl/run_grpo_template.sh reads $DATA_DIR/{train,val}_<agent>.parquet.)

Rows keep TRACE's prompt (with its ``<image>`` marker, which verl's dataset
splits on), the TRACE system prompt, the embedded image bytes, and the reward
inputs (answer_gt in ``reward_model.ground_truth``; annotation_gt and
reward_contract in ``extra_info``). ``agent_name`` selects verl's agent loop per
row: ``tool_agent`` for the tool arm, ``single_turn_agent`` for the no-tool arm.
The tool set itself comes from the tool_config YAML, so both arms share data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from trace_tool.conditions import TRACE_PROMPT_FILE, read_system_prompt
from trace_tool.data import REPO_ID, REVISION, SPLIT_FILES, TRAIN_FILES

DATA_SOURCE = "maveryn/trace"


def _source_files(split: str, local_dir: str | None) -> list[Path]:
    names = TRAIN_FILES if split == "train" else [SPLIT_FILES[split]]
    if local_dir:
        return [Path(local_dir) / name for name in names]
    from huggingface_hub import hf_hub_download

    return [Path(hf_hub_download(REPO_ID, name, repo_type="dataset", revision=REVISION)) for name in names]


def convert_table(table: pa.Table, *, system_prompt: str, agent: str, split: str, offset: int) -> pa.Table:
    records = table.select(
        ["images", "prompt_answer", "answer_gt", "annotation_gt", "reward_contract", "instance_id", "domain", "task"]
    ).to_pylist()
    out = []
    for i, r in enumerate(records):
        out.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": r["prompt_answer"]}],
                "images": [{"bytes": img["bytes"], "path": None} for img in r["images"]],
                "ability": "visual_reasoning",
                "reward_model": {"style": "rule", "ground_truth": r["answer_gt"]},
                "extra_info": {
                    "split": split,
                    "index": offset + i,
                    "instance_id": r["instance_id"],
                    "domain": r["domain"],
                    "task": r["task"],
                    "annotation_gt": r["annotation_gt"],
                    "reward_contract": r["reward_contract"],
                },
                "agent_name": agent,
            }
        )
    return pa.Table.from_pylist(out)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["train", "validation"], required=True)
    p.add_argument("--agent", choices=["tool_agent", "single_turn_agent"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--local-dir", help="local snapshot of maveryn/trace (default: download pinned revision)")
    args = p.parse_args(argv)

    system_prompt = read_system_prompt([TRACE_PROMPT_FILE])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    offset = 0
    for source in _source_files(args.split, args.local_dir):
        converted = convert_table(pq.read_table(source), system_prompt=system_prompt, agent=args.agent, split=args.split, offset=offset)
        offset += converted.num_rows
        if writer is None:
            writer = pq.ParquetWriter(out_path, converted.schema)
        writer.write_table(converted)
    if writer is not None:
        writer.close()
    print(f"wrote {offset} rows to {out_path}")


if __name__ == "__main__":
    main()
