"""Fixtures for the M3 RL runner tests (experiments/certificates/tests_rl).

Run in the RL env from the repository root, with the machine environment sourced:

    source "$TRACE_CERT_RUNS/host/env.sh"
    "$TRC_RL_PYTHON" -m pytest experiments/certificates/tests_rl -q -p no:cacheprovider

Tests that need the trace-rlvr checkout (TRACE_RLVR_ROOT), the frozen data (TRACE_CERT_RUNS) or the
cached base snapshot (HF_HOME) skip when those are absent. Nothing here touches a GPU.
"""

from __future__ import annotations

import os

import pytest

from rl_helpers import load_train_rl


@pytest.fixture(scope="session")
def trl():
    return load_train_rl()


@pytest.fixture(scope="session")
def rlvr_env():
    if not os.environ.get("TRACE_RLVR_ROOT") or not os.environ.get("TRACE_CERT_RUNS"):
        pytest.skip("TRACE_RLVR_ROOT / TRACE_CERT_RUNS not set (source $TRACE_CERT_RUNS/host/env.sh)")
    return os.environ


@pytest.fixture(scope="session")
def specs(trl):
    return {model: trl.load_spec(model) for model in trl.MODELS}


@pytest.fixture(scope="session")
def releases(trl, specs, rlvr_env):
    return {model: trl.release_easyr1(spec, trl.Paths(spec)) for model, spec in specs.items()}


@pytest.fixture(scope="session")
def data_ready(trl, specs, rlvr_env):
    paths = trl.Paths(specs["3b"])
    if not paths.train_parquet.is_file() or not paths.val_parquet.is_file() or not paths.store.is_file():
        pytest.skip(f"frozen RL data not found under {paths.data}")
    return paths


@pytest.fixture(scope="session")
def base_snapshot(trl, specs, rlvr_env):
    try:
        return trl.Paths(specs["3b"]).base_snapshot(specs["3b"])
    except trl.ConfigError as exc:
        pytest.skip(str(exc))
