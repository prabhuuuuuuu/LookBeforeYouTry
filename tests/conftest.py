"""Shared fixtures. Tests marked ``jetson`` need CUDA and local weights (``pytest -m jetson``)."""

import os
from pathlib import Path

import pytest


def _model_dir(env: str, default: str) -> Path:
    return Path(os.environ.get(env, default))


@pytest.fixture(scope="session")
def moondream_path() -> Path:
    return _model_dir("LBYR_MOONDREAM_PATH", "/home/sentrix/models/moondream2")


@pytest.fixture(scope="session")
def qwen_path() -> Path:
    return _model_dir("LBYR_QWEN_PATH", "/home/sentrix/models/qwen2.5-vl-3b")


@pytest.fixture(scope="session")
def cuda() -> None:
    """Skip (not fail) when this machine has no torch / CUDA."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
