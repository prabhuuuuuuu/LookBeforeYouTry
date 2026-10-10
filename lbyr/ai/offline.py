"""Edge-only guard: make sure Hugging Face libraries never touch the network (bible rule 1).

Moondream2's remote code loads its tokenizer by hub id (``moondream/starmie-v1``), so
``local_files_only=True`` alone is not enough: it must resolve from the local HF cache.
"""

from __future__ import annotations

import os
import sys
from typing import Any


def force_offline() -> None:
    """Set ``HF_HUB_OFFLINE=1``, also when huggingface_hub was already imported."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    constants: Any = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        constants.HF_HUB_OFFLINE = True
