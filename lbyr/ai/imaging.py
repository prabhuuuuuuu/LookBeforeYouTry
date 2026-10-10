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
    box_norm: tuple[float, float, float, float] | None = None,
    size_px: int = MOONDREAM_TILE_PX,
) -> np.ndarray:
    """Gripper region of the wrist frame, RESIZED to ``size_px`` x ``size_px`` for the Critic.

    ``box_norm`` = (x0, y0, x1, y1) in normalized frame coordinates (0..1;
    ``CriticConfig.crop_box_norm``, OQ-12). None = the width-centred full-height square
    (side min(H, W)). The box is clipped to the frame, then resized with cv2.INTER_AREA
    (Pillow BOX if cv2 is missing). A box that is not square in PIXELS is stretched.
    Cost: one area resize, ~1-3 ms for a 1080x1080 region.
    """
    h, w = image_bgr.shape[:2]
    if box_norm is None:
        side = min(h, w)
        x0, y0, x1, y1 = (w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2
    else:
        x0, y0 = max(0, round(box_norm[0] * w)), max(0, round(box_norm[1] * h))
        x1, y1 = min(w, round(box_norm[2] * w)), min(h, round(box_norm[3] * h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"crop box {box_norm} is outside the {w}x{h} frame")
    out = image_bgr[y0:y1, x0:x1]
    if out.shape[:2] != (size_px, size_px):
        out = _resize_square(out, size_px)
    return np.ascontiguousarray(out)


def _resize_square(image: np.ndarray, size_px: int) -> np.ndarray:
    """Area resize (cv2.INTER_AREA); Pillow BOX, then NumPy nearest-neighbour fallbacks."""
    try:
        import cv2  # lazy

        return np.asarray(cv2.resize(image, (size_px, size_px), interpolation=cv2.INTER_AREA))
    except ImportError:
        pass
    try:
        from PIL import Image
    except ImportError:
        h, w = image.shape[:2]
        rows = (np.arange(size_px) * h // size_px).clip(0, h - 1)
        cols = (np.arange(size_px) * w // size_px).clip(0, w - 1)
        return image[rows][:, cols]
    img = Image.fromarray(np.ascontiguousarray(image))
    return np.asarray(img.resize((size_px, size_px), Image.Resampling.BOX))
