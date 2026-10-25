"""Copy a Qwen2.5-VL checkpoint and install a chat template that supports tools.

verl's ToolAgentLoop renders the tool schemas (``tools=``) and tool results
(``role: tool``) through the checkpoint's chat template. Qwen2.5-VL's template
has neither, so the policy would never see the tools and results would render
as ``<|im_start|>tool``. configs/verl/qwen2_5_vl_tool_chat_template.jinja
renders both exactly like the evaluation prompts (tests/test_rl_template.py).
Use the prepared copy for every RL arm, so the arms differ only in tools.

    python -m trace_tool.rl.prepare_policy --model Qwen/Qwen2.5-VL-7B-Instruct --out models/qwen2.5-vl-7b-tools
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_tool.conditions import REPO_ROOT
from trace_tool.models import PINNED_REVISIONS

TEMPLATE_PATH = REPO_ROOT / "configs" / "verl" / "qwen2_5_vl_tool_chat_template.jinja"


def install_template(model_dir: Path, template: str) -> list[str]:
    """Write the template wherever transformers may read one; returns the files changed."""
    changed = []
    jinja_file = model_dir / "chat_template.jinja"
    json_file = model_dir / "chat_template.json"
    if jinja_file.exists():
        jinja_file.write_text(template)
        changed.append(jinja_file.name)
    if json_file.exists() or not jinja_file.exists():
        json_file.write_text(json.dumps({"chat_template": template}, ensure_ascii=False, indent=2) + "\n")
        changed.append(json_file.name)
    tokenizer_config = model_dir / "tokenizer_config.json"
    if tokenizer_config.exists():
        config = json.loads(tokenizer_config.read_text())
        if "chat_template" in config:
            config["chat_template"] = template
            tokenizer_config.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
            changed.append(tokenizer_config.name)
    return changed


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF id or local directory of a Qwen2.5-VL checkpoint")
    p.add_argument("--revision", help="default: TRACE's pinned revision for known models")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    out = Path(args.out)
    if Path(args.model).is_dir():
        import shutil

        shutil.copytree(args.model, out, dirs_exist_ok=True)
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(args.model, revision=args.revision or PINNED_REVISIONS.get(args.model), local_dir=out)
    changed = install_template(out, TEMPLATE_PATH.read_text())
    print(f"{out}: installed the tool chat template in {', '.join(changed)}")


if __name__ == "__main__":
    main()
