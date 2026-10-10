"""Trapezoidal joint-space trajectories sampled at the stream rate (bible 11.4).

All joints share one normalised profile scaled by their displacement, so they start and
stop together and no joint exceeds the velocity/acceleration caps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lbyr.config import ArmConfig, MotionConfig, require


@dataclass(frozen=True)
class Trajectory:
    points: np.ndarray  # (N, 6) degrees; points[k] is streamed at tick k
    dt_s: float

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[0] < 1:
            raise ValueError("trajectory needs at least one point")

    @property
    def duration_s(self) -> float:
        return len(self.points) * self.dt_s

    @property
    def final(self) -> np.ndarray:
        return self.points[-1]


def trapezoid_position(t: float, distance: float, v_max: float, a_max: float) -> float:
    """Distance travelled at time t along a trapezoidal (or triangular) profile."""
    if distance <= 0:
        return 0.0
    t_acc = v_max / a_max
    if distance < v_max * t_acc:  # triangular: never reaches v_max
        t_acc = math.sqrt(distance / a_max)
        v_peak = a_max * t_acc
        t_flat = 0.0
    else:
        v_peak = v_max
        t_flat = (distance - v_max * t_acc) / v_max
    total = 2 * t_acc + t_flat
    t = min(max(t, 0.0), total)
    if t < t_acc:
        return 0.5 * a_max * t * t
    if t < t_acc + t_flat:
        return 0.5 * a_max * t_acc * t_acc + v_peak * (t - t_acc)
    td = total - t
    return distance - 0.5 * a_max * td * td


def trapezoid_duration(distance: float, v_max: float, a_max: float) -> float:
    if distance <= 0:
        return 0.0
    if distance < v_max * v_max / a_max:
        return 2 * math.sqrt(distance / a_max)
    return distance / v_max + v_max / a_max


def plan_joint_trajectory(
    start_deg: Sequence[float],
    goal_deg: Sequence[float],
    v_max_deg_s: float,
    a_max_deg_s2: float,
    rate_hz: float,
) -> Trajectory:
    if v_max_deg_s <= 0 or a_max_deg_s2 <= 0 or rate_hz <= 0:
        raise ValueError("v_max, a_max and rate must be > 0")
    start = np.asarray(start_deg, dtype=float)
    goal = np.asarray(goal_deg, dtype=float)
    delta = goal - start
    dist = float(np.max(np.abs(delta)))
    dt = 1.0 / rate_hz
    total = trapezoid_duration(dist, v_max_deg_s, a_max_deg_s2)
    n = max(1, math.ceil(total / dt - 1e-9))
    pts = np.empty((n, len(start)))
    for k in range(1, n + 1):
        s = trapezoid_position(k * dt, dist, v_max_deg_s, a_max_deg_s2) / dist if dist > 0 else 1.0
        pts[k - 1] = start + s * delta
    pts[-1] = goal  # exact endpoint
    return Trajectory(pts, dt)


def capped_limits(arm: ArmConfig, motion: MotionConfig) -> tuple[float, float]:
    """(v_max, a_max) after the dev/stage speed cap (bible 8: 30% / 50%)."""
    v = require(arm.max_joint_speed_deg_s, "max joint speed", "A-1")
    a = require(arm.max_joint_accel_deg_s2, "max joint acceleration", "A-1")
    return v * motion.speed_cap, a * motion.speed_cap


def plan_capped(
    start_deg: Sequence[float], goal_deg: Sequence[float], arm: ArmConfig, motion: MotionConfig
) -> Trajectory:
    v, a = capped_limits(arm, motion)
    return plan_joint_trajectory(start_deg, goal_deg, v, a, motion.rate_hz)
