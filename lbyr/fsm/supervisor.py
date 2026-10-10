"""10 Hz FSM supervisor (bible Sections 9.3, 10, 13).

- Each state has an ``_enter_*`` (runs once) and an optional ``_tick_*`` (runs at 10 Hz).
  Both return ``None`` (stay) or ``(next_state, reason)``.
- Motion (PP-1): each joint goal is ONE segment handed to the MotionController (non-blocking);
  the bridge firmware generates the trajectory and reports DONE. No DONE within the
  predicted time x 1.5 + 0.5 s, an ERR reply or a serial error = LINK -> ABORT.
  JAMMED: bridge status, and (only with measured feedback, i.e. the mock) load threshold
  and measured-vs-commanded position error. The real arm has no feedback (OQ-5).
- Verify states submit a Critic request and poll the verdict queue without blocking.
  INVALID (unparseable answer): re-query once with a fresh frame, then treat as OCCLUDED
  (logged as ``critic_invalid_retry``). TIMEOUT: re-query once, then ABORT.
- PLAN calls the planner under its 3 s hard timeout. This blocks only the FSM thread
  while the arm is idle; the board holds position by itself.
- Failures -> RECOVER(k) per the recovery table, else ABORT with a human-readable report.
- Joint targets come from a :class:`~lbyr.motion.targets.TargetProvider` (the supervisor never
  computes them): ``motion.mode = "taught"`` (default, demo path) = taught joint poses + zone
  polygons; ``"ik"`` = D435 depth + hand-eye + IK (needs OQ-4).
- Optional hooks: ``SupervisorDeps.observer`` gets one :class:`LiveEvent` per transition
  (terminal live view); ``EpisodeSpec.preset`` carries a plan made before the episode
  (language mode, demo intro); ``EpisodeSpec.induce_slip`` makes the first grasp use
  ``SkillConfig.induced_slip_closure_pct`` (logged as ``induced_slip``).
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
from lbyr.ai.planner import Planner, PlanRequest, PlanResult, SkillChain, plan_with_timeout
from lbyr.arm.interface import ArmInterface
from lbyr.config import WRIST_PITCH_INDEX, AppConfig
from lbyr.errors import ConfigError, IKError, PerceptionError
from lbyr.fsm.recovery import RECOVERY_TABLE, FailureClass, RecoveryBudget, classify
from lbyr.fsm.states import VERIFY_STATES, State, next_in_pipeline
from lbyr.logging_.episodes import EpisodeLogger, VramMonitor, make_record
from lbyr.motion.controller import FAULT, MotionController
from lbyr.motion.targets import GoalPlan, GraspAdjustments, Scene, TargetProvider, Targets
from lbyr.perception.buffers import LatestValueBuffer
from lbyr.perception.detector import Detection, Detector
from lbyr.perception.overhead import RGBDFrame

__all__ = [
    "EpisodeResult",
    "EpisodeSpec",
    "GraspAdjustments",
    "LiveEvent",
    "Supervisor",
    "SupervisorDeps",
    "VerdictInfo",
]

log = logging.getLogger(__name__)

Next = tuple[State, str] | None


def _shape_str(image: np.ndarray | None) -> str | None:
    return None if image is None else f"{image.shape[1]}x{image.shape[0]}"


def _det_dict(d: Detection) -> dict[str, Any]:
    out: dict[str, Any] = {"id": d.id, "label": d.label, "bbox_xyxy": d.bbox_xyxy}
    out["centroid_px"] = d.centroid
    for k in ("yaw_deg", "xyz_cam_mm", "zone", "area_px"):
        v = getattr(d, k, None)
        if v is not None:
            out[k] = v
    return out


PERCEIVE_MAX_WAIT_TICKS = 20
VERIFY_WATCHDOG_MARGIN_S = 1.0


@dataclass(frozen=True)
class EpisodeSpec:
    instruction: str
    expected_label: str | None = None  # WRONG_OBJECT re-check vs detected class
    induce_slip: bool = False  # first grasp at SkillConfig.induced_slip_closure_pct
    preset: PlanResult | None = None  # plan made before the episode (language mode intro)


@dataclass(frozen=True)
class VerdictInfo:
    state: State
    verdict: Verdict
    raw_text: str | None
    latency_ms: float


@dataclass(frozen=True)
class LiveEvent:
    """One FSM transition for the terminal live view."""

    t: float
    src: State
    dst: State
    reason: str
    verdict: VerdictInfo | None = None  # the Critic answer that caused this transition
    recovery: str | None = None  # "k=1 +15% closure" on transitions out of RECOVER


@dataclass
class SupervisorDeps:
    arm: ArmInterface
    motion: MotionController
    planner: Planner
    detector: Detector
    overhead: LatestValueBuffer[RGBDFrame] | None  # None = no overhead camera (--zone)
    wrist: LatestValueBuffer[np.ndarray] | None  # None only with the Critic OFF
    targets: TargetProvider  # joint targets: taught poses or IK (motion.mode)
    critic: CriticWorker | None  # None = "Critic OFF" baseline
    logger: EpisodeLogger | None = None
    vram: VramMonitor | None = None
    observer: Callable[[LiveEvent], None] | None = None  # terminal live view


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
    verdict_log: list[VerdictInfo] = field(default_factory=list)
    induced_slip: bool = False  # the first grasp really used the induced closure
    planner_latency_ms: float | None = None
    duration_s: float = 0.0
    detections: list[Detection] = field(default_factory=list)  # last PERCEIVE


@dataclass
class _EpisodeCtx:
    spec: EpisodeSpec
    episode_id: str
    budget: RecoveryBudget
    adj: GraspAdjustments = field(default_factory=GraspAdjustments)
    scene: Scene | None = None
    chain: SkillChain | None = None
    targets: Targets | None = None
    grasps: int = 0  # GRASP entries this episode
    induced_active: bool = False  # current grasp uses the induced-slip closure
    induced_used: bool = False
    t_start: float = 0.0
    last_verdict: VerdictInfo | None = None  # consumed by the next transition (live view)
    last_recovery: str | None = None
    verdict_log: list[VerdictInfo] = field(default_factory=list)
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
        self.targets = deps.targets
        if deps.targets.name != cfg.motion.mode:
            raise ConfigError(
                f"motion.mode = {cfg.motion.mode!r} but the target provider is "
                f"{deps.targets.name!r} (taught mode needs taught poses)"
            )
        if deps.critic is not None and deps.wrist is None:
            raise ConfigError("the Critic needs the wrist camera buffer")
        self.state = State.IDLE
        self._ctx: _EpisodeCtx | None = None
        # motion bookkeeping
        self._goals: list[np.ndarray] = []
        self._goal_idx = 0
        self._seg_id = -1
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
            t_start=time.perf_counter(),
        )
        self.state = State.IDLE
        log.info(
            "Episode %s: %r (critic %s)",
            self.ctx.episode_id,
            spec.instruction,
            "ON" if self.d.critic else "OFF",
        )
        self.d.motion.set_output_enabled(True)
        self.d.arm.torque(True)
        self._record("episode_start", induce_slip=spec.induce_slip, motion_mode=self.targets.name)
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
        c = self.ctx
        c.transitions.append(Transition(time.time(), src, dst, reason))
        log.info("FSM %-12s -> %-12s | %s", src.value, dst.value, reason)
        self._record("transition", from_state=src, reason=reason)
        if self.d.observer is not None:
            event = LiveEvent(
                time.time(),
                src,
                dst,
                reason,
                c.last_verdict,
                c.last_recovery if src is State.RECOVER else None,
            )
            try:
                self.d.observer(event)
            except Exception:  # the live view must never break the episode
                log.exception("live view observer failed")
        c.last_verdict = None
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
        c.last_recovery = f"k={k} {action.description}"
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
        frame: RGBDFrame | None = None
        if self.d.overhead is not None:
            item = self.d.overhead.get()
            if item is None:
                self._perceive_waits += 1
                if self._perceive_waits > PERCEIVE_MAX_WAIT_TICKS:
                    return self._fail(FailureClass.PERCEPTION_FAILED, "no overhead frame")
                return None
            frame = item.value
            self._save_frame(frame.color, "overhead")
        dets = self.d.detector.detect(frame)
        c = self.ctx
        self._record("detections", detections=[_det_dict(d) for d in dets])
        c.scene = None
        try:
            c.scene = self.targets.scene(dets, frame)
        except PerceptionError as e:
            return self._fail(FailureClass.PERCEPTION_FAILED, str(e))
        return State.PLAN, c.scene.summary

    def _enter_plan(self) -> Next:
        c = self.ctx
        assert c.scene is not None
        dets = c.scene.detections
        if c.spec.preset is not None:  # language mode: planned before the episode
            res = c.spec.preset
            chain = res.chain
            if chain is not None and not {chain.approach.target_id, chain.place.target_id} <= set(
                dets
            ):
                res = PlanResult(
                    None,
                    res.latency_ms,
                    res.raw_text,
                    "preset plan ids not detected now",
                    res.trace,
                )
        else:
            req = PlanRequest(c.spec.instruction, list(dets.values()))
            res = plan_with_timeout(self.d.planner, req, self.cfg.planner)
        c.planner_latency_ms = res.latency_ms
        # Reasoning trace (prompt, raw output, parsed chain) for the dashboard.
        trace = res.trace.to_dict() if res.trace is not None else None
        self._record("reasoning_trace", plan_error=res.error, trace=trace)
        if res.chain is None:
            return self._fail(FailureClass.PLANNER_FAILED, res.error or "no plan")
        label = dets[res.chain.approach.target_id].label
        if c.spec.expected_label is not None and label != c.spec.expected_label:
            return self._fail(
                FailureClass.WRONG_OBJECT,
                f"planner chose {label!r}, expected {c.spec.expected_label!r}",
            )
        try:
            c.targets = self.targets.bind(
                c.scene,
                res.chain.approach.target_id,
                res.chain.place.target_id,
                res.chain.approach.standoff_mm,
            )
        except IKError as e:
            return self._fail(FailureClass.IK_FAILED, str(e))
        c.chain = res.chain
        zone = f", zone {c.targets.zone.name}" if c.targets.zone is not None else ""
        return State.APPROACH, f"plan ok in {res.latency_ms:.0f} ms{zone}"

    # ================================================================ motion states
    def _closure_pct(self) -> float:
        c = self.ctx
        assert c.chain is not None
        if c.induced_active:  # --induce-slip: the first grasp is deliberately loose
            return self.cfg.skills.induced_slip_closure_pct
        return min(100.0, c.chain.grasp.closure + c.adj.closure_delta_pct)

    def _skill_goals(self, label: str, make: Callable[[Targets, np.ndarray], GoalPlan]) -> Next:
        """Ask the target provider for this skill's goals and start the motion."""
        c = self.ctx
        assert c.targets is not None
        try:
            plan = make(c.targets, self._setpoint())
        except IKError as e:  # IK failure, or a missing taught pose / radius
            return self._fail(FailureClass.IK_FAILED, f"{label}: {e}")
        c.target = {"label": label, **plan.info}
        self._start_motion(plan.goals)
        return None

    def _enter_approach(self) -> Next:
        a = self.ctx.adj
        return self._skill_goals("grasp", lambda t, here: self.targets.approach_goals(t, here, a))

    def _enter_grasp(self) -> Next:
        c = self.ctx
        c.induced_active = c.spec.induce_slip and c.grasps == 0
        c.grasps += 1
        pct = self._closure_pct()
        if c.induced_active:
            c.induced_used = True
            self._record("induced_slip", closure_pct=pct)
            log.warning("INDUCED SLIP: first grasp closes to %.0f%% only", pct)
        return self._skill_goals("close", lambda t, here: self.targets.grasp_goals(here, pct))

    def _enter_lift(self) -> Next:
        a, pct = self.ctx.adj, self._closure_pct()
        return self._skill_goals("lift", lambda t, here: self.targets.lift_goals(t, here, pct, a))

    def _enter_transport(self) -> Next:
        a, pct = self.ctx.adj, self._closure_pct()
        return self._skill_goals(
            "transport", lambda t, here: self.targets.transport_goals(t, here, pct, a)
        )

    def _enter_place(self) -> Next:
        a, pct = self.ctx.adj, self._closure_pct()
        return self._skill_goals("place", lambda t, here: self.targets.place_goals(t, here, pct, a))

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
            if self.d.wrist is not None and (w := self.d.wrist.get()) is not None:
                self._save_frame(w.value, "raw")
            self._save_overhead("overhead")
            self._record("verdict", verdict=None, critic_skipped=True)
            return next_in_pipeline(self.state), "Critic OFF: pass-through"
        assert self.d.wrist is not None
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
        self._save_overhead("overhead")
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
        info = VerdictInfo(self.state, r.verdict, r.raw_text, r.latency_ms)
        c.verdict_log.append(info)
        c.last_verdict = info
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
            self.d.motion.set_output_enabled(False)
            self.d.arm.stop()
            self.d.arm.torque(False)
            c.abort_actions += ["stop", "torque OFF (JAMMED) - support the arm by hand"]
            self._finish("abort")
            return None
        if c.failure is FailureClass.LINK:
            # The link is unreliable: do not plan more motion and do not release torque
            # (the arm would fall). PWM servos hold the last pulse by themselves.
            self.d.motion.set_output_enabled(False)
            try:
                self.d.arm.stop()
                c.abort_actions.append("stop sent (x), best effort")
            except Exception as e:
                c.abort_actions.append(f"stop could not be sent ({e})")
            c.abort_actions.append(
                "LINK fault: servos hold the last pulse; check the USB cable / bridge, keep "
                "the E-stop in reach, support the arm before power-off"
            )
            self._finish("abort")
            return None
        safe = self._safe_pose()
        if safe is None:
            self.d.arm.stop()  # the board brakes and holds
            c.abort_actions.append("holding position (safe pose unconfirmed, A-3)")
            self._finish("abort")
            return None
        c.abort_actions.append("moving to safe pose")
        self._start_motion([safe])
        return None

    def _tick_abort(self) -> Next:
        if self.ctx.finished:
            return None
        res = self._motion_tick()
        if res is not None:  # jam / link fault while retreating: stop, hold
            self.d.motion.set_output_enabled(False)
            try:
                self.d.arm.stop()
            except Exception:
                log.exception("stop failed")
            jammed = "JAMMED" in res[1]
            if jammed:
                self.d.arm.torque(False)
            self.ctx.abort_actions.append(
                f"safe-pose move failed ({res[1]}): stop" + (", torque OFF" if jammed else "")
            )
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
        """Goal of the latest segment (the board's target), host deg."""
        return np.asarray(self.d.motion.current().goal, dtype=float)

    def _safe_pose(self) -> np.ndarray | None:
        return self.targets.safe_pose(self._setpoint())

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
        self._seg_id = self.d.motion.command(self._goals[self._goal_idx])

    def _motion_tick(self) -> Next:
        """Advance multi-segment motion; return a LINK / JAMMED failure if anything is wrong."""
        if not self._motion_active():
            return None
        ms = self.d.motion.current()
        if ms.seg_id != self._seg_id:
            return None  # not dispatched yet (next poll tick)
        if ms.phase == FAULT:
            return self._fail(FailureClass.LINK, ms.fault or "motion fault")
        st = self.d.arm.read_state()
        if not st.ok:
            return self._fail(FailureClass.JAMMED, f"bridge status {st.status!r}")
        if st.feedback == "measured":  # mock / future sensors only (OQ-5: none on the arm)
            thr = self.cfg.arm.jam_load_threshold
            if thr is not None and st.load is not None and max(st.load) > thr:
                return self._fail(FailureClass.JAMMED, f"load {max(st.load):.2f} > {thr}")
            if st.commanded is not None:
                err = float(np.max(np.abs(np.asarray(st.joints) - np.asarray(st.commanded))))
                if err > self.cfg.motion.jam_position_error_deg:
                    return self._fail(FailureClass.JAMMED, f"position error {err:.1f} deg")
        if ms.done:
            self._next_goal()
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

    def _save_overhead(self, tag: str) -> None:
        """Decision-point overhead frame (if an overhead camera runs)."""
        if self.d.overhead is not None and (item := self.d.overhead.get()) is not None:
            self._save_frame(item.value.color, tag)

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
            verdict_log=list(c.verdict_log),
            induced_slip=c.induced_used,
            planner_latency_ms=c.planner_latency_ms,
            duration_s=time.perf_counter() - c.t_start,
            detections=list(c.scene.detections.values()) if c.scene is not None else [],
        )
