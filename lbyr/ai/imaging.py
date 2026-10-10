"""Image conversion helpers for the VLM backends. Pillow is imported lazily."""

from __future__ import annotations

from typing import Any

import numpy as np


def bgr_to_pil(image_bgr: np.ndarray) -> Any:
    """OpenCV-style BGR uint8 (H, W, 3) -> PIL RGB image. Pillow is imported lazily."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) BGR image, got shape {image_bgr.shape}")
    from PIL import Image

    return Image.fromarray(np.ascontiguousarray(image_bgr[..., ::-1]), "RGB")
