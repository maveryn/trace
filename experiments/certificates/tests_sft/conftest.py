"""Fixtures for the M2 SFT tests (experiments/certificates/tests_sft).

Run in the training env (torch 2.8 / transformers 4.57.6 / vLLM 0.10.2) from the repository root,
with the machine environment sourced:

    source "$TRACE_CERT_RUNS/host/env.sh"
    TRC_DATA=$TRACE_CERT_RUNS/data/v2 "$TRC_RL_PYTHON" -m pytest experiments/certificates/tests_sft -q -p no:cacheprovider

TRC_DATA selects the frozen data root (default $TRACE_CERT_RUNS/data/v2, common.sh's default); the
scorer steps need a root whose receipts match the library's export version. The CPU training and
job tests start two torchrun ranks (about 3 GiB together): on a login node under a reduced
per-user memory cap, run them on a compute allocation instead.

Tests that need the frozen data (TRACE_CERT_RUNS), the cached base snapshot (HF_HOME) or the
trace-rlvr checkout (TRACE_RLVR_ROOT) skip when those are absent. Nothing here touches a GPU: the
training tests run torchrun on CPU (gloo) with a tiny random Qwen2.5-VL that uses the real
tokenizer and processor files.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.dont_write_bytecode = True  # never write __pycache__ into the read-only trace-rlvr checkout
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import sft_helpers as h  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# EasyR1's RLHFDataset goes through `datasets`, which caches parquet conversions: keep them out of HF_HOME
os.environ["HF_DATASETS_CACHE"] = tempfile.mkdtemp(prefix="sft_tests_hfds_")
if str(h.TRACE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(h.TRACE_ROOT / "src"))


@pytest.fixture(scope="session")
def sft():
    return h.load_sft()


@pytest.fixture(scope="session")
def select():
    return h.load_select()


@pytest.fixture(scope="session")
def cfg3b(sft):
    return sft.load_config(h.CONFIG_3B)


@pytest.fixture(scope="session")
def cfg7b(sft):
    return sft.load_config(h.CONFIG_7B)


@pytest.fixture(scope="session")
def data_root():
    runs = os.environ.get("TRACE_CERT_RUNS")
    if not runs:
        pytest.skip("TRACE_CERT_RUNS not set (source $TRACE_CERT_RUNS/host/env.sh)")
    root = Path(os.environ.get("TRC_DATA") or Path(runs) / "data" / "v2")
    for rel in ("sft/sft.jsonl", "public/e1_dev.cert_px.parquet", "rl/e1_dev/e1_dev.parquet", "private/e1_dev.extensions_v1.jsonl.zst"):
        if not (root / rel).exists():
            pytest.skip(f"frozen data {root / rel} not found")
    return root


@pytest.fixture(scope="session")
def base_dir(sft, cfg3b):
    if not os.environ.get("HF_HOME"):
        pytest.skip("HF_HOME not set")
    try:
        return sft.base_snapshot(cfg3b)
    except Exception as exc:  # not cached
        pytest.skip(f"base snapshot not available: {exc}")


@pytest.fixture(scope="session")
def rlvr_root():
    root = os.environ.get("TRACE_RLVR_ROOT")
    if not root or not (Path(root) / "rlvr" / "easyr1_backend" / "verl" / "utils" / "dataset.py").exists():
        pytest.skip("TRACE_RLVR_ROOT not set or not a trace-rlvr checkout")
    return Path(root)


@pytest.fixture(scope="session")
def processor(sft, base_dir):
    return sft.load_processor(base_dir)


@pytest.fixture(scope="session")
def encoder(sft, processor, cfg3b):
    from trace_tasks.experimental.certificates import prompts

    return sft.Encoder(processor, prompts.system_prompt(), cfg3b["data"]["min_pixels"], cfg3b["data"]["max_pixels"])


@pytest.fixture(scope="session")
def easyr1(rlvr_root, base_dir):
    """EasyR1's own dataset module, tokenizer and processor (exactly as verl.trainer.main loads them)."""
    backend = rlvr_root / "rlvr" / "easyr1_backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from verl.utils import dataset as ds
    from verl.utils.tokenizer import get_processor, get_tokenizer

    return {"dataset": ds, "tokenizer": get_tokenizer(str(base_dir), use_fast=True), "processor": get_processor(str(base_dir), use_fast=True)}


@pytest.fixture(scope="session")
def dev_store(data_root):
    from trace_tasks.experimental.certificates.export import read_store
    from trace_tasks.experimental.certificates.types import CertificateExtension

    return {e.instance_id: e for e in (CertificateExtension.from_dict(p) for p in read_store(data_root / "private" / "e1_dev.extensions_v1.jsonl.zst"))}


@pytest.fixture(scope="session")
def tiny_model(base_dir, tmp_path_factory):
    """Built in a subprocess: the CPU training tests keep torch out of the pytest process (the
    login node caps a user at 8 GiB, and two torchrun ranks need about 3 GiB)."""
    out = tmp_path_factory.mktemp("tiny") / "model"
    code = f"import sys; sys.path.insert(0, {str(HERE)!r}); import sft_helpers as h; from pathlib import Path; h.build_tiny_model(Path({str(base_dir)!r}), Path({str(out)!r}))"
    res = h.run([sys.executable, "-c", code], env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]
    return out


@pytest.fixture(scope="session")
def mini_data(data_root, tmp_path_factory):
    return h.mini_data_root(data_root, tmp_path_factory.mktemp("mini_data"), 4)


@pytest.fixture(scope="session")
def cpu_env(base_dir):
    env = dict(os.environ)
    env.update({"TRC_SFT_DEVICE": "cpu", "TRC_SFT_WORKERS": "0", "PYTHONPATH": f"{h.TRACE_ROOT / 'src'}:{env.get('PYTHONPATH', '')}",
                "OMP_NUM_THREADS": "1", "MALLOC_ARENA_MAX": "2", "TRC_SFT_CE_CHUNK": "64",
                "PYTHONDONTWRITEBYTECODE": "1"})
    return env
