"""Shared helpers of the RL runner tests (imported by conftest.py and the test modules)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

CERT_DIR = Path(__file__).resolve().parents[1]
TRACE_ROOT = CERT_DIR.parents[1]
DUMMY_SHA = "0" * 64


def load_train_rl():
    name = "cert_train_rl"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, CERT_DIR / "train_rl.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve(trl, spec, release, *, arm="P", seed=1, n_gpus=2, run_dir=None, smoke=False, **kwargs):
    """resolve_config with a dummy store sha and SFT dir (pure: touches no file)."""
    paths = trl.Paths(spec)
    return trl.resolve_config(
        spec,
        release,
        arm=arm,
        seed=seed,
        n_gpus=n_gpus,
        model_path="/sft/dir",
        run_dir=str(run_dir or paths.run_dir(spec, arm, seed)),
        data_root=str(paths.data),
        store_sha256=DUMMY_SHA,
        smoke=smoke,
        **kwargs,
    )
