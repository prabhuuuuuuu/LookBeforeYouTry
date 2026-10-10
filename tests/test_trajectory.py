import numpy as np
import pytest

from lbyr.config import ArmConfig, MotionConfig, mock_profile
from lbyr.errors import ConfigError, HardwareNotConfirmedError, IKError
from lbyr.motion.ik import CartesianTarget, IkpyIK, MockIK
from lbyr.motion.trajectory import plan_capped, plan_joint_trajectory

V, A, RATE = 60.0, 200.0, 50.0


@pytest.mark.parametrize(
    ("start", "goal"),
    [
        ((0, 0, 0, 0, 0, 0), (90, -45, 30, 10, 0, 50)),  # trapezoid (reaches v_max)
        ((0, 0, 0, 0, 0, 0), (5, 1, 0, 0, 0, 0)),  # triangular (short move)
        ((10, 10, 10, 10, 10, 10), (10, 10, 10, 10, 10, 10)),  # zero move
    ],
)
def test_trapezoid_endpoints_and_velocity_cap(start: tuple, goal: tuple) -> None:
    traj = plan_joint_trajectory(start, goal, V, A, RATE)
    np.testing.assert_array_equal(traj.final, np.asarray(goal, float))
    pts = np.vstack([np.asarray(start, float), traj.points])
    vel = np.diff(pts, axis=0) * RATE
    assert np.max(np.abs(vel)) <= V + 1e-6
    # joints start and stop together: every joint finishes on the last sample
    assert traj.dt_s == pytest.approx(1 / RATE)


def test_trapezoid_duration_reasonable() -> None:
    traj = plan_joint_trajectory((0,) * 6, (90, 0, 0, 0, 0, 0), V, A, RATE)
    expected = 90 / V + V / A  # cruise + accel/decel
    assert traj.duration_s == pytest.approx(expected, abs=1 / RATE)


def test_speed_cap_applied() -> None:
    cfg = mock_profile()
    traj = plan_capped((0,) * 6, (90, 0, 0, 0, 0, 0), cfg.arm, cfg.motion)
    pts = np.vstack([np.zeros(6), traj.points])
    v_cap = cfg.arm.max_joint_speed_deg_s * cfg.motion.dev_speed_cap  # type: ignore[operator]
    assert np.max(np.abs(np.diff(pts, axis=0))) * cfg.motion.rate_hz <= v_cap + 1e-6
    with pytest.raises(HardwareNotConfirmedError, match="A-1"):
        plan_capped((0,) * 6, (1,) * 6, ArmConfig(), MotionConfig())


def test_mock_ik_round_trip() -> None:
    ik = MockIK(mock_profile().arm)
    target = CartesianTarget((220.0, 60.0, 70.0), wrist_roll_deg=15.0)
    q = ik.solve(target, (0,) * 5)
    np.testing.assert_allclose(ik.forward(q), target.xyz_mm, atol=1e-6)
    assert q[4] == 15.0
    with pytest.raises(IKError):
        ik.solve(CartesianTarget((900.0, 0.0, 0.0)), (0,) * 5)


def test_real_ik_refuses_until_confirmed() -> None:
    with pytest.raises(HardwareNotConfirmedError, match="OQ-4"):
        IkpyIK(ArmConfig())
    with pytest.raises(ConfigError):
        IkpyIK(mock_profile().arm)
    with pytest.raises(ConfigError):
        MockIK(ArmConfig())
