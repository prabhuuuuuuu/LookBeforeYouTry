"""CUDA memory logging helpers. ``torch`` is imported lazily so laptops work without it."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_MB = 1024.0 * 1024.0


def cuda_allocated_mb() -> float | None:
    """``torch.cuda.memory_allocated`` in MB, or None if torch / CUDA is unavailable."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.memory_allocated()) / _MB


def cuda_peak_mb() -> float | None:
    """``torch.cuda.max_memory_allocated`` in MB, or None if torch / CUDA is unavailable."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated()) / _MB


def reset_cuda_peak() -> None:
    """``torch.cuda.reset_peak_memory_stats`` (no-op without torch / CUDA)."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def log_cuda_allocated(label: str) -> float | None:
    """Log (INFO) and return ``torch.cuda.memory_allocated`` after loading ``label``."""
    mb = cuda_allocated_mb()
    if mb is None:
        log.info("%s loaded (CUDA memory not available)", label)
    else:
        log.info("%s loaded: torch.cuda.memory_allocated = %.0f MB", label, mb)
    return mb
