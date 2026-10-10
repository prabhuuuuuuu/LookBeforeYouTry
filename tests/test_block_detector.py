"""BlockDetector on synthetic frames: a brown (wood) block on a white table."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from lbyr.config import BlockDetectorConfig  # noqa: E402
from lbyr.errors import ConfigError  # noqa: E402
from lbyr.perception.block import BlockDetector, long_side_yaw_deg  # noqa: E402
from lbyr.perception.geometry import Intrinsics, unproject  # noqa: E402
from lbyr.perception.overhead import RGBDFrame  # noqa: E402

INTR = Intrinsics(615.0, 615.0, 320.0, 240.0, 640, 480)
WOOD_BGR = (70, 130, 185)  # light brown
TABLE_MM, BLOCK_TOP_MM = 900.0, 860.0


def table_frame(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    color = np.clip(rng.normal(235, 4, (480, 640, 3)), 0, 255).astype(np.uint8)  # white + noise
    depth = np.full((480, 640), TABLE_MM, np.float32)
    return color, depth


def draw_block(
    color: np.ndarray,
    depth: np.ndarray,
    center: tuple[float, float],
    size: tuple[float, float],
    angle: float,
    bgr: tuple[int, int, int] = WOOD_BGR,
    top_mm: float = BLOCK_TOP_MM,
) -> None:
    pts = np.round(cv2.boxPoints((center, size, angle))).astype(np.int32)
    cv2.fillPoly(color, [pts], bgr)
    m = np.zeros(depth.shape, np.uint8)
    cv2.fillPoly(m, [pts], 1)
    depth[m > 0] = top_mm


def frame_with_block(**kw: object) -> RGBDFrame:
    color, depth = table_frame()
    draw_block(color, depth, **kw)  # type: ignore[arg-type]
    return RGBDFrame(color, depth, INTR)


def test_brown_block_on_white_table() -> None:
    det = BlockDetector(BlockDetectorConfig())
    frame = frame_with_block(center=(300.0, 260.0), size=(44.0, 26.0), angle=30.0)
    dets = det.detect(frame)
    assert len(dets) == 1
    d = dets[0]
    assert d.id == 1 and d.label == "wood_block"
    u, v = d.centroid
    assert u == pytest.approx(300.0, abs=1.0) and v == pytest.approx(260.0, abs=1.0)
    assert d.yaw_deg == pytest.approx(30.0, abs=3.0)
    assert d.area_px == pytest.approx(44 * 26, rel=0.15)
    x1, y1, x2, y2 = d.bbox_xyxy
    assert x1 < u < x2 and y1 < v < y2
    assert d.xyz_cam_mm is not None
    expected = unproject(u, v, BLOCK_TOP_MM, INTR)
    assert np.allclose(d.xyz_cam_mm, expected, atol=1.0)
    assert d.score > 0.8  # rectangle fill ratio
    assert det.last_ms > 0.0


@pytest.mark.parametrize("angle", [-60.0, -10.0, 0.0, 45.0, 80.0])
def test_yaw_is_the_long_side_angle(angle: float) -> None:
    d = BlockDetector(BlockDetectorConfig()).detect(
        frame_with_block(center=(320.0, 240.0), size=(60.0, 24.0), angle=angle)
    )[0]
    diff = (d.yaw_deg - angle + 90.0) % 180.0 - 90.0  # yaw is modulo 180
    assert abs(diff) < 3.0


def test_long_side_yaw_helper() -> None:
    box = np.array([[0, 0], [10, 0], [10, 4], [0, 4]], float)
    assert long_side_yaw_deg(box) == pytest.approx(0.0)
    assert long_side_yaw_deg(box[:, ::-1]) == pytest.approx(-90.0)  # vertical long side


def test_largest_blob_wins_and_small_noise_is_ignored() -> None:
    color, depth = table_frame()
    draw_block(color, depth, (150.0, 150.0), (20.0, 20.0), 0.0)  # smaller block
    draw_block(color, depth, (450.0, 300.0), (40.0, 40.0), 10.0)  # the target
    color[400:404, 100:104] = (40, 60, 120)  # 16 px crumb < min_area_px
    d = BlockDetector(BlockDetectorConfig()).detect(RGBDFrame(color, depth, INTR))
    assert len(d) == 1 and d[0].centroid[0] == pytest.approx(450.0, abs=1.5)


def test_gray_shadow_is_not_an_object() -> None:
    color, depth = table_frame()
    color[200:300, 200:320] = (175, 175, 175)  # soft shadow on white
    assert BlockDetector(BlockDetectorConfig()).detect(RGBDFrame(color, depth, INTR)) == []


def test_workspace_polygon_masks_outside_clutter() -> None:
    color, depth = table_frame()
    draw_block(color, depth, (100.0, 100.0), (120.0, 60.0), 0.0, bgr=(30, 30, 30))  # arm/clutter
    draw_block(color, depth, (400.0, 300.0), (40.0, 30.0), 0.0)
    ws = ((250.0, 200.0), (600.0, 200.0), (600.0, 450.0), (250.0, 450.0))
    d = BlockDetector(BlockDetectorConfig(), workspace_px=ws).detect(RGBDFrame(color, depth, INTR))
    assert len(d) == 1 and d[0].centroid == pytest.approx((400.0, 300.0), abs=1.5)
    assert (
        BlockDetector(BlockDetectorConfig(), workspace_px=ws[:2] + ((250.0, 199.0),)).detect(
            RGBDFrame(*table_frame(), INTR)
        )
        == []
    )


def test_too_big_blob_is_rejected_and_no_depth_gives_no_xyz() -> None:
    color, depth = table_frame()
    draw_block(color, depth, (320.0, 240.0), (200.0, 150.0), 0.0)  # 30000 px > max_area_px
    assert BlockDetector(BlockDetectorConfig()).detect(RGBDFrame(color, depth, INTR)) == []
    f = frame_with_block(center=(320.0, 240.0), size=(40.0, 30.0), angle=0.0)
    f.depth_mm[:] = 0.0  # no valid depth anywhere
    d = BlockDetector(BlockDetectorConfig()).detect(f)[0]
    assert d.xyz_cam_mm is None and d.centroid == pytest.approx((320.0, 240.0), abs=1.0)


def test_zone_lookup_hsv_and_no_frame() -> None:
    cfg = BlockDetectorConfig(color_space="hsv")
    det = BlockDetector(cfg, zone_of=lambda u, v: "A" if u < 320 else None)
    assert det.detect(frame_with_block(center=(200.0, 240.0), size=(40.0, 30.0), angle=0.0))[
        0
    ].zone == ("A")
    assert det.detect(None) == []


def test_config_validation() -> None:
    with pytest.raises(ConfigError):
        BlockDetector(BlockDetectorConfig(color_space="rgb"))
    with pytest.raises(ConfigError):
        BlockDetector(BlockDetectorConfig(min_area_px=500.0, max_area_px=100.0))
