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


# Moondream2 vision tile (config.vision.crop_size). An image this size is encoded as a
# single tile instead of a global view + overlapping local crops.
MOONDREAM_TILE_PX = 378


def critic_crop(
    image_bgr: np.ndarray,
    box_xywh: tuple[int, int, int, int] | None = None,
    size_px: int = MOONDREAM_TILE_PX,
) -> np.ndarray:
    """Crop the wrist frame to a ``size_px`` square for the Critic.

    ``box_xywh`` is the gripper ROI (``WristCameraConfig.crop_xywh``, OQ-12); None = the
    centred square of side ``min(size_px, H, W)``. The box is clipped to the frame; if the
    result is not ``size_px`` x ``size_px`` it is resized (Pillow, lazily). Cost: a slice
    (centred default) or one resize, well under 1 ms.
    """
    h, w = image_bgr.shape[:2]
    if box_xywh is None:
        side = min(size_px, h, w)
        x, y, bw, bh = (w - side) // 2, (h - side) // 2, side, side
    else:
        x, y, bw, bh = box_xywh
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"crop box {box_xywh} is outside the {w}x{h} frame")
    out = image_bgr[y0:y1, x0:x1]
    if out.shape[:2] != (size_px, size_px):
        out = _resize_square(out, size_px)
    return np.ascontiguousarray(out)


def _resize_square(image: np.ndarray, size_px: int) -> np.ndarray:
    """Bilinear resize with Pillow; nearest-neighbour NumPy fallback if Pillow is missing."""
    try:
        from PIL import Image
    except ImportError:
        h, w = image.shape[:2]
        rows = (np.arange(size_px) * h // size_px).clip(0, h - 1)
        cols = (np.arange(size_px) * w // size_px).clip(0, w - 1)
        return image[rows][:, cols]
    return np.asarray(Image.fromarray(np.ascontiguousarray(image)).resize((size_px, size_px)))
