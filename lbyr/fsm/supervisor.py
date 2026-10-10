"""10 Hz FSM supervisor (bible Sections 9.3, 10, 13).

- Each state has an ``_enter_*`` (runs once) and an optional ``_tick_*`` (runs at 10 Hz).
  Both return ``None`` (stay) or ``(next_state, reason)``.
- Motion: joint goals -> trapezoidal trajectories -> MotionStreamer (non-blocking hand-off).
  Jam detection: bridge status, load threshold (if confirmed, A-4), commanded-vs-actual
  position error, motion timeout.
- Verify states submit a Critic request and poll the verdict queue without blocking.
  INVALID (unparseable answer): re-query once with a fresh frame, then treat as OCCLUDED
  (logged as ``critic_invalid_retry``). TIMEOUT: re-query once, then ABORT.
- PLAN calls the planner under its 3 s hard timeout. This blocks only the FSM thread
  while the arm is idle; the streamer keeps holding position on its own thread.
- Failures -> RECOVER(k) per the recovery table, else ABORT with a human-readable report.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lbyr.ai.critic import CriticResult, CriticWorker, Verdict
from lbyr.ai.planner import Planner, PlanRequest, SkillChain, plan_with_timeout
from lbyr.arm.interface import ArmInterface
from lbyr.config import GRIPPER_INDEX, WRIST_PITCH_INDEX, AppConfig
from lbyr.errors import IKError
from lbyr.fsm.recovery import RECOVERY_TABLE, FailureClass, RecoveryBudget, classify
from lbyr.fsm.states import VERIFY_STATES, State, next_in_pipeline
from lbyr.logging_.episodes import EpisodeLogger, VramMonitor, make_record
from lbyr.motion.ik import CartesianTarget, IKSolver
from lbyr.motion.streamer import MotionStreamer
from lbyr.motion.trajectory import plan_capped
from lbyr.perception.buffers import LatestValueBuffer
from lbyr.perception.detector import Detection, Detector
from lbyr.perception.geometry import locate_in_base
from lbyr.perception.overhead import RGBDFrame

log = logging.getLogger(__name__)

Next = tuple[State, str] | None


def _shape_str(image: np.ndarray | None) -> str | None:
    return None if image is None else f"{image.shape[1]}x{image.shape[0]}"


PERCEIVE_MAX_WAIT_TICKS = 20
VERIFY_WATCHDOG_MARGIN_S = 1.0


@dataclass(frozen=True)
class EpisodeSpec:
    instruction: str
    expected_label: str | None = None  # WRONG_OBJECT re-check vs detected class


@dataclass
class SupervisorDeps:
    arm: ArmInterface
    streamer: MotionStreamer
    ik: IKSolver
    planner: Planner
    detector: Detector
    overhead: LatestValueBuffer[RGBDFrame]
    wrist: LatestValueBuffer[np.ndarray]
    base_T_cam: np.ndarray
    critic: CriticWorker | None  # None = "Critic OFF" baseline
    logger: EpisodeLogger | None = None
    vram: VramMonitor | None = None


@dataclass
class GraspAdjustments:
    """Accumulated recovery deltas, applied on top of the planner's skill parameters."""

    closure_delta_pct: float = 0.0
    wrist_roll_deg: float = 0.0
    lateral_offset_mm: float = 0.0
    grasp_z_offset_mm: float = 0.0  # EMPTY recovery: "-5 mm standoff" = grasp 5 mm lower


@dataclass(frozen=True)
class Transition:
    t: float
    src: State
    dst: State
    reason: str


@dataclass
class EpisodeResult:
    episode_id: str
    outcome: str  # "success" | "abort"
    final_state: State
    failure: FailureClass | None
    recoveries: int
    recovery_history: list[tuple[FailureClass, str]]
    verdicts: list[Verdict]
    transitions: list[Transition]
    report: str


@dataclass
class _EpisodeCtx:
    spec: EpisodeSpec
    episode_id: str
    budget: RecoveryBudget
    adj: GraspAdjustments = field(default_factory=GraspAdjustments)
    detections: dict[int, Detection] = field(default_factory=dict)
    positions: dict[int, np.ndarray] = field(default_factory=dict)
    chain: SkillChain | None = None
    grasp_xyz: np.ndarray | None = None
    target: dict[str, Any] | None = None
    failure: FailureClass | None = None
    failure_detail: str = ""
    failed_state: State | None = None
    reentry: State | None = None
    planner_latency_ms: float | None = None
    critic_invalid_retries: int = 0  # INVALID re-queries since the last valid verdict
    verdicts: list[Verdict] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    abort_actions: list[str] = field(default_factory=list)
    finished: bool = False
    frame_n: int = 0


class Supervisor:
    def __init__(self, cfg: AppConfig, deps: SupervisorDeps) -> None:
        cfg.validate()
        self.cfg = cfg
        self.d = deps
        self.state = State.IDLE
        self._ctx: _EpisodeCtx | None = None
        # motion bookkeeping
        self._goals: list[np.ndarray] = []
        self._goal_idx = 0
        self._traj_id = -1
        self._motion_deadline = 0.0
        # verify bookkeeping
        self._pending_rid: int | None = None
        self._verify_deadline = 0.0
        self._perceive_waits = 0
        self._enter: dict[State, Callable[[], Next]] = {
            State.IDLE: lambda: None,
            State.PERCEIVE: self._enter_perceive,
            State.PLAN: self._enter_plan,
            State.APPROACH: self._enter_approach,
            State.GRASP: self._enter_grasp,
            State.LIFT: self._enter_lift,
            State.TRANSPORT: self._enter_transport,
            State.PLACE: self._enter_place,
            State.VERIFY_GRASP: self._enter_verify,
            State.VERIFY_HOLD: self._enter_verify,
            State.VERIFY_PLACE: self._enter_verify,
            State.RECOVER: self._enter_recover,
            State.ABORT: self._enter_abort,
            State.DONE: self._enter_done,
        }
        self._tick: dict[State, Callable[[], Next]] = {
            State.IDLE: lambda: (State.PERCEIVE, "start"),
            State.PERCEIVE: self._tick_perceive,
            State.ABORT: self._tick_abort,
            State.DONE: lambda: None,
        }
        for s in (State.APPROACH, State.GRASP, State.LIFT, State.TRANSPORT, State.PLACE):
            self._tick[s] = self._tick_motion_state
        for s in VERIFY_STATES:
            self._tick[s] = self._tick_verify
        self._tick[State.RECOVER] = self._tick_recover

    @property
    def ctx(self) -> _EpisodeCtx:
        assert self._ctx is not None, "no episode running"
        return self._ctx

    # ================================================================ main loop
    def run_episode(self, spec: EpisodeSpec, timeout_s: float = 120.0) -> EpisodeResult:
        self._ctx = _EpisodeCtx(
            spec=spec,
            episode_id=time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
            budget=RecoveryBudget(max_total=self.cfg.recovery.max_total_recoveries),
        )
        self.state = State.IDLE
        log.info(
            "Episode %s: %r (critic %s)",
            self.ctx.episode_id,
            spec.instruction,
            "ON" if self.d.critic else "OFF",
        )
        self.d.streamer.set_output_enabled(True)
        self.d.arm.torque(True)
        self._record("episode_start")
        period = 1.0 / self.cfg.fsm_rate_hz
        t_end = time.perf_counter() + timeout_s
        next_t = time.perf_counter()
        while not self.ctx.finished:
            if time.perf_counter() > t_end and self.state is not State.ABORT:
                self._transition(State.ABORT, f"episode timeout ({timeout_s:.0f} s)")
            else:
                self._step()
            next_t += period
            now = time.perf_counter()
            if next_t < now:  # e.g. after the (bounded) planner call: re-anchor, no burst
                next_t = now
            time.sleep(next_t - now)
        return self._result()

    def _step(self) -> None:
        nxt = self._tick[self.state]()
        if nxt is not None:
            self._transition(*nxt)

    def _transition(self, dst: State, reason: str) -> None:
        src = self.state
        self.state = dst
        self.ctx.transitions.append(Transition(time.time(), src, dst, reason))
        log.info("FSM %-12s -> %-12s | %s", src.value, dst.value, reason)
        self._record("transition", from_state=src, reason=reason)
        nxt = self._enter[dst]()
        if nxt is not None:
            self._transition(*nxt)

    # ================================================================ failures
    def _fail(self, failure: FailureClass, detail: str) -> Next:
        c = self.ctx
        c.failure, c.failure_detail, c.failed_state = failure, detail, self.state
        reason = f"{failure.value}: {detail}"
        if RECOVERY_TABLE[failure].max_attempts > 0:
            return State.RECOVER, reason
        return State.ABORT, reason

    def _enter_recover(self) -> Next:
        c = self.ctx
        assert c.failure is not None and c.failed_state is not None
        action = c.budget.next_action(c.failure)
        if action is None:
            used = c.budget.per_class.get(c.failure, 0)
            return State.ABORT, f"{c.failure.value}: recovery budget exhausted ({used} used)"
        c.adj.closure_delta_pct += action.closure_delta_pct
        c.adj.wrist_roll_deg += action.wrist_roll_delta_deg
        c.adj.lateral_offset_mm += action.lateral_offset_mm
        c.adj.grasp_z_offset_mm += action.standoff_delta_mm
        c.reentry = action.reentry or c.failed_state
        k = c.budget.total
        log.info("RECOVER(k=%d) %s -> %s", k, action.description, c.reentry.value)
        self._record("recover", action=action.description, attempt=k)
        if action.reposition_wrist_cam:  # one-off camera move; later poses come from IK again
            goal = self._setpoint()
            goal[WRIST_PITCH_INDEX] += self.cfg.recovery.occluded_wrist_pitch_delta_deg
            self._start_motion([goal])
            return None
        return c.reentry, action.description

    def _tick_recover(self) -> Next:
        res = self._motion_tick()
        if res is not None:
            return res
        if not self._motion_active():
            assert self.ctx.reentry is not None
            return self.ctx.reentry, "wrist camera re-positioned"
        return None

    # ================================================================ perceive / plan
    def _enter_perceive(self) -> Next:
        self._perceive_waits = 0
        return self._tick_perceive()

    def _tick_perceive(self) -> Next:
        item = self.d.overhead.get()
        if item is None:
            self._perceive_waits += 1
            if self._perceive_waits > PERCEIVE_MAX_WAIT_TICKS:
                return self._fail(FailureClass.PERCEPTION_FAILED, "no overhead frame")
            return None
        frame = item.value
        dets = self.d.detector.detect(frame.color)
        c = self.ctx
        c.detections, c.positions = {}, {}
        for det in dets:
            u, v = det.centroid
            p = locate_in_base(
                u,
                v,
                frame.depth_mm,
                frame.intrinsics,
                self.d.base_T_cam,
                self.cfg.cameras.overhead.depth_median_window,
            )
            if p is not None:
                c.detections[det.id] = det
                c.positions[det.id] = p
        if not c.positions:
            return self._fail(FailureClass.PERCEPTION_FAILED, "no detections with valid depth")
        return State.PLAN, f"{len(c.positions)} objects localised"

    def _enter_plan(self) -> Next:
        c = self.ctx
        req = PlanRequest(c.spec.instruction, list(c.detections.values()))
        res = plan_with_timeout(self.d.planner, req, self.cfg.planner)
        c.planner_latency_ms = res.latency_ms
        # Reasoning trace (prompt, raw output, parsed chain) for the dashboard.
        trace = res.trace.to_dict() if res.trace is not None else None
        self._record("reasoning_trace", plan_error=res.error, trace=trace)
        if res.chain is None:
            return self._fail(FailureClass.PLANNER_FAILED, res.error or "no plan")
        label = c.detections[res.chain.approach.target_id].label
        if c.spec.expected_label is not None and label != c.spec.expected_label:
            return self._fail(
                FailureClass.WRONG_OBJECT,
                f"planner chose {label!r}, expected {c.spec.expected_label!r}",
            )
        c.chain = res.chain
        return State.APPROACH, f"plan ok in {res.latency_ms:.0f} ms"

    # ================================================================ motion states
    def _closure_pct(self) -> float:
        assert self.ctx.chain is not None
        return min(100.0, self.ctx.chain.grasp.closure + self.ctx.adj.closure_delta_pct)

    def _gripper_deg(self, closure_pct: float) -> float:
        return self.cfg.arm.gripper.closure_to_deg(closure_pct)

    def _pose(self, xyz: np.ndarray, gripper_deg: float) -> np.ndarray:
        """IK for a top-down TCP pose + gripper -> 6-joint goal (deg)."""
        target = CartesianTarget(
            (float(xyz[0]), float(xyz[1]), float(xyz[2])), self.ctx.adj.wrist_roll_deg
        )
        q = self.d.ik.solve(target, self._setpoint()[:GRIPPER_INDEX].tolist())
        return np.array([*q, gripper_deg], dtype=float)

    def _plan_goals(self, poses: list[tuple[np.ndarray, float]], label: str) -> Next:
        try:
            goals = [self._pose(xyz, g) for xyz, g in poses]
        except IKError as e:
            return self._fail(FailureClass.IK_FAILED, f"{label}: {e}")
        self.ctx.target = {"label": label, "xyz_mm": poses[-1][0]}
        self._start_motion(goals)
        return None

    def _enter_approach(self) -> Next:
        c = self.ctx
        assert c.chain is not None
        obj = c.positions[c.chain.approach.target_id]
        grasp = np.array(
            [
                obj[0],
                obj[1] + c.adj.lateral_offset_mm,
                obj[2] / 2.0
                + c.adj.grasp_z_offset_mm,  # mid-height of an object resting on the mat
            ]
        )
        c.grasp_xyz = grasp
        pre = grasp + [0.0, 0.0, c.chain.approach.standoff_mm]
        open_deg = self._gripper_deg(0.0)
        open_in_place = self._setpoint()
        open_in_place[GRIPPER_INDEX] = open_deg
        try:
            goals = [open_in_place, self._pose(pre, open_deg), self._pose(grasp, open_deg)]
        except IKError as e:
            return self._fail(FailureClass.IK_FAILED, f"approach: {e}")
        c.target = {"label": "grasp", "xyz_mm": grasp}
        self._start_motion(goals)
        return None

    def _enter_grasp(self) -> Next:
        goal = self._setpoint()
        goal[GRIPPER_INDEX] = self._gripper_deg(self._closure_pct())
        self.ctx.target = {"label": "close", "closure_pct": self._closure_pct()}
        self._start_motion([goal])
        return None

    def _enter_lift(self) -> Next:
        assert self.ctx.grasp_xyz is not None
        up = self.ctx.grasp_xyz + [0.0, 0.0, self.cfg.skills.lift_height_mm]
        return self._plan_goals([(up, self._gripper_deg(self._closure_pct()))], "lift")

    def _place_xy(self) -> np.ndarray:
        assert self.ctx.chain is not None
        return self.ctx.positions[self.ctx.chain.place.target_id]

    def _enter_transport(self) -> Next:
        p = self._place_xy()
        xyz = np.array([p[0], p[1], self.cfg.skills.transport_height_mm])
        return self._plan_goals([(xyz, self._gripper_deg(self._closure_pct()))], "transport")

    def _enter_place(self) -> Next:
        assert self.ctx.grasp_xyz is not None
        p = self._place_xy()
        z = p[2] + self.ctx.grasp_xyz[2] + self.cfg.skills.place_release_height_mm
        release = np.array([p[0], p[1], z])
        retreat = release + [0.0, 0.0, self.cfg.skills.lift_height_mm]
        closed, opened = self._gripper_deg(self._closure_pct()), self._gripper_deg(0.0)
        return self._plan_goals([(release, closed), (release, opened), (retreat, opened)], "place")

    def _tick_motion_state(self) -> Next:
        res = self._motion_tick()
        if res is not None:
            return res
        if not self._motion_active():
            return next_in_pipeline(self.state), "motion complete"
        return None

    # ================================================================ verify
    def _enter_verify(self) -> Next:
        if self.d.critic is None:
            self._record("verdict", verdict=None, critic_skipped=True)
            return next_in_pipeline(self.state), "Critic OFF: pass-through"
        item = self.d.wrist.get()
        if item is None:
            return self._fail(FailureClass.CRITIC_UNAVAILABLE, "no wrist frame")
        rid = self.d.critic.submit(item.value, self.state.value)
        if rid is None:
            return self._fail(FailureClass.CRITIC_UNAVAILABLE, "critic queue full")
        self._pending_rid = rid
        self._verify_deadline = (
            time.perf_counter() + self.cfg.critic.timeout_s + VERIFY_WATCHDOG_MARGIN_S
        )
        self._save_frame(item.value, "raw")
        return None

    def _tick_verify(self) -> Next:
        assert self.d.critic is not None
        while (r := self.d.critic.poll()) is not None:
            if r.request_id == self._pending_rid:
                return self._on_verdict(r)
            log.debug("discarding stale critic result %d", r.request_id)
        if time.perf_counter() > self._verify_deadline:
            self._pending_rid = None
            return self._fail(FailureClass.CRITIC_UNAVAILABLE, "no verdict (watchdog)")
        return None

    def _on_verdict(self, r: CriticResult) -> Next:
        c = self.ctx
        self._pending_rid = None
        c.verdicts.append(r.verdict)
        frame = self._save_frame(r.image_bgr, "judged") if r.image_bgr is not None else None
        self._record(
            "verdict",
            verdict=r.verdict,
            verdict_latency_ms=r.latency_ms,
            raw_text=r.raw_text,
            critic_infer_ms=r.infer_ms,
            critic_input=_shape_str(r.image_bgr),
            frame=frame,
        )
        log.info("Critic %s -> %s (%.0f ms)", self.state.value, r.verdict.value, r.latency_ms)
        verdict = r.verdict
        if verdict is Verdict.INVALID:
            if c.critic_invalid_retries == 0:  # re-query once with a fresh frame
                c.critic_invalid_retries = 1
                self._record("critic_invalid_retry", action="requery", raw_text=r.raw_text)
                return self.state, "Critic INVALID: re-query once"
            self._record("critic_invalid_retry", action="treat_as_occluded", raw_text=r.raw_text)
            verdict = Verdict.OCCLUDED
        else:
            c.critic_invalid_retries = 0
        failure = classify(self.state, verdict)
        if failure is None:
            return next_in_pipeline(self.state), f"Critic={verdict.value}"
        return self._fail(failure, f"Critic={r.verdict.value} in {self.state.value}")

    # ================================================================ terminal
    def _enter_done(self) -> Next:
        self._finish("success")
        return None

    def _enter_abort(self) -> Next:
        c = self.ctx
        self._goals = []
        if c.failure is FailureClass.JAMMED:
            self.d.streamer.set_output_enabled(False)
            self.d.arm.stop()
            self.d.arm.torque(False)
            c.abort_actions += ["stop", "torque OFF (JAMMED) - support the arm by hand"]
            self._finish("abort")
            return None
        safe = self.cfg.arm.safe_pose_deg
        if safe is None:
            self.d.streamer.hold(self._setpoint().tolist())
            c.abort_actions.append("holding position (safe pose unconfirmed, A-3)")
            self._finish("abort")
            return None
        goal = np.asarray(safe, dtype=float)
        goal[GRIPPER_INDEX] = self._setpoint()[GRIPPER_INDEX]  # do not drop a held object
        c.abort_actions.append("moving to safe pose")
        self._start_motion([goal])
        return None

    def _tick_abort(self) -> Next:
        if self.ctx.finished:
            return None
        res = self._motion_tick()
        if res is not None:  # jam while retreating: stop + torque off
            self.d.streamer.set_output_enabled(False)
            self.d.arm.stop()
            self.d.arm.torque(False)
            self.ctx.abort_actions.append(f"safe-pose move failed ({res[1]}): stop, torque OFF")
            self._finish("abort")
        elif not self._motion_active():
            self.ctx.abort_actions.append("reached safe pose, holding")
            self._finish("abort")
        return None

    def _finish(self, outcome: str) -> None:
        c = self.ctx
        c.finished = True
        report = self._report(outcome)
        level = logging.INFO if outcome == "success" else logging.WARNING
        log.log(level, "\n%s", report)
        self._record(
            "episode_end",
            outcome=outcome,
            report=report,
            recoveries=c.budget.total,
            failure_class=c.failure,
        )

    # ================================================================ motion engine
    def _setpoint(self) -> np.ndarray:
        return np.asarray(self.d.streamer.current().setpoint, dtype=float)

    def _start_motion(self, goals: list[np.ndarray]) -> None:
        self._goals = list(goals)
        self._goal_idx = -1
        self._next_goal()

    def _motion_active(self) -> bool:
        return self._goal_idx < len(self._goals)

    def _next_goal(self) -> None:
        self._goal_idx += 1
        if self._goal_idx >= len(self._goals):
            return
        traj = plan_capped(
            self._setpoint().tolist(),
            self._goals[self._goal_idx].tolist(),
            self.cfg.arm,
            self.cfg.motion,
        )
        self._traj_id = self.d.streamer.command(traj)
        self._motion_deadline = (
            time.perf_counter() + traj.duration_s + self.cfg.motion.motion_timeout_margin_s
        )

    def _motion_tick(self) -> Next:
        """Advance multi-goal motion; return a JAMMED failure if anything looks wrong."""
        if not self._motion_active():
            return None
        st = self.d.arm.read_state()
        actual = np.asarray(st.joints)
        if not st.ok:
            return self._fail(FailureClass.JAMMED, f"bridge status {st.status!r}")
        thr = self.cfg.arm.jam_load_threshold
        if thr is not None and st.load is not None and max(st.load) > thr:
            return self._fail(FailureClass.JAMMED, f"load {max(st.load):.2f} > {thr}")
        ss = self.d.streamer.current()
        err = float(np.max(np.abs(actual - np.asarray(ss.setpoint))))
        if err > self.cfg.motion.jam_position_error_deg:
            return self._fail(FailureClass.JAMMED, f"position error {err:.1f} deg")
        if ss.traj_id == self._traj_id and ss.done:
            goal_err = float(np.max(np.abs(actual - self._goals[self._goal_idx])))
            if goal_err <= self.cfg.motion.position_tolerance_deg:
                self._next_goal()
                return None
        if time.perf_counter() > self._motion_deadline:
            return self._fail(FailureClass.JAMMED, "motion timeout")
        return None

    # ================================================================ logging
    def _record(self, event: str, **extra: Any) -> None:
        if self.d.logger is None:
            return
        c = self.ctx
        try:
            joints: Any = self.d.arm.read_state().joints
        except Exception:
            joints = None
        fields: dict[str, Any] = {
            "episode_id": c.episode_id,
            "event": event,
            "fsm_state": self.state,
            "joints": joints,
            "target": c.target,
            "planner_latency_ms": c.planner_latency_ms,
            "vram_mb": self.d.vram.latest_mb if self.d.vram else None,
            "critic_enabled": self.d.critic is not None,
            "attempt": c.budget.total,
            "failure_class": c.failure,
        }
        fields.update(extra)
        self.d.logger.log(make_record(**fields))

    def _save_frame(self, image: np.ndarray, tag: str) -> str | None:
        if self.d.logger is None:
            return None
        c = self.ctx
        c.frame_n += 1
        return self.d.logger.save_frame(
            f"{c.episode_id}/{c.frame_n:03d}_{self.state.value}_{tag}.jpg", image
        )

    def _report(self, outcome: str) -> str:
        c = self.ctx
        path = [State.IDLE.value] + [t.dst.value for t in c.transitions]
        lines = [
            f"===== Episode {c.episode_id}: {outcome.upper()} =====",
            f"Instruction : {c.spec.instruction}",
            f"Final state : {self.state.value}",
            f"Path        : {' -> '.join(path)}",
            f"Verdicts    : {[v.value for v in c.verdicts] or 'none (Critic OFF?)'}",
            f"Recoveries  : {c.budget.total} (max {c.budget.max_total})",
        ]
        for i, (fc, desc) in enumerate(c.budget.history, 1):
            lines.append(f"  {i}. {fc.value}: {desc}")
        if outcome != "success":
            failure = c.failure.value if c.failure else "unknown"
            where = c.failed_state.value if c.failed_state else "?"
            lines.append(f"Failure     : {failure} in {where} - {c.failure_detail}")
            lines.append(f"Abort steps : {'; '.join(c.abort_actions) or '-'}")
            lines.append("Operator    : inspect the arm before the next run; E-stop within reach.")
        return "\n".join(lines)

    def _result(self) -> EpisodeResult:
        c = self.ctx
        return EpisodeResult(
            episode_id=c.episode_id,
            outcome="success" if self.state is State.DONE else "abort",
            final_state=self.state,
            failure=c.failure if self.state is not State.DONE else None,
            recoveries=c.budget.total,
            recovery_history=list(c.budget.history),
            verdicts=list(c.verdicts),
            transitions=list(c.transitions),
            report=self._report("success" if self.state is State.DONE else "abort"),
        )
