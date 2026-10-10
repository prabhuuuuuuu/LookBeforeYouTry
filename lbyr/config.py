"""Dataclass configuration for LBYR.

Rules (bible Sections 20/21):
- Every hardware-dependent value defaults to ``None`` with a comment citing its open
  question. ``OQ-n`` refers to bible Section 20; ``A-n`` refers to software-surfaced
  questions listed in docs/OPEN_QUESTIONS.md.
- Code that needs such a value calls :func:`require`, which raises
  :class:`HardwareNotConfirmedError` instead of guessing.
- :func:`mock_profile` returns obviously-fake SIMULATION-ONLY values for mock mode.
  Real hardware drivers refuse any config with ``ArmConfig.simulated = True``.

Units: millimetres and degrees at every boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from lbyr.errors import ConfigError, HardwareNotConfirmedError

T = TypeVar("T")

# Bible 3.1: base yaw, shoulder pitch, elbow pitch, wrist pitch, wrist roll, gripper.
# Joint ORDER on the real bus is still unconfirmed (OQ-2 IDs / bible 3.1 "confirm joint mapping").
JOINT_NAMES: tuple[str, ...] = (
    "base_yaw",
    "shoulder_pitch",
    "elbow_pitch",
    "wrist_pitch",
    "wrist_roll",
    "gripper",
)
NUM_JOINTS = len(JOINT_NAMES)
GRIPPER_INDEX = 5
WRIST_PITCH_INDEX = 3
WRIST_ROLL_INDEX = 4

DATA_DIR = Path("data")

PLANNER_BACKENDS: tuple[str, ...] = ("qwen", "rules")
MOTION_MODES: tuple[str, ...] = ("taught", "ik")
CALIB_DIR = DATA_DIR / "calib"
CRITIC_DTYPES: tuple[str, ...] = ("fp16", "bf16")
MODEL_DTYPES: tuple[str, ...] = ("bf16", "fp16")
# Language mode (demo intro): "qwen3_text" (Qwen3 LLM) | "qwen_vl" (Qwen2.5-VL, text-only)
LANGUAGE_BACKENDS: tuple[str, ...] = ("qwen3_text", "qwen_vl")


def require(value: T | None, what: str, question: str) -> T:
    """Return ``value`` or raise if it is still unconfirmed (``None``)."""
    if value is None:
        raise HardwareNotConfirmedError(what, question)
    return value


@dataclass
class GripperConfig:
    open_deg: float | None = None  # OQ-6: gripper stroke / closure-to-angle mapping, unconfirmed
    closed_deg: float | None = None  # OQ-6: gripper stroke / closure-to-angle mapping, unconfirmed

    def closure_to_deg(self, closure_pct: float) -> float:
        """Map closure in percent (0 = open, 100 = fully closed) to gripper servo degrees."""
        open_deg = require(self.open_deg, "gripper open angle", "OQ-6")
        closed_deg = require(self.closed_deg, "gripper closed angle", "OQ-6")
        c = min(max(closure_pct, 0.0), 100.0) / 100.0
        return open_deg + c * (closed_deg - open_deg)


@dataclass
class ArmConfig:
    simulated: bool = False
    servo_bus: str | None = "pwm"  # OQ-1 closed (bible v1.3): PWM servos
    servo_ids: tuple[int, ...] | None = None  # OQ-2: not applicable (PWM)
    servo_baud: int | None = None  # OQ-2: not applicable (PWM)
    servo_voltage_v: float | None = 6.0  # OQ-3 closed (bible v1.3): 6.0 V rail. NEVER 5 V.
    serial_port: str | None = None  # OQ-10: XIAO device name (/dev/ttyACM*?), unconfirmed
    host_baud: int | None = 115200  # A-5 closed: firmware Serial.begin(115200)
    joints_path: Path = CALIB_DIR / "joints.yaml"  # zeros/signs, verify vs board `list`
    joint_min_deg: tuple[float, ...] | None = None  # OQ-4 / bible 3.1 #4: joint limits
    joint_max_deg: tuple[float, ...] | None = None  # OQ-4 / bible 3.1 #4: joint limits
    link_lengths_mm: tuple[float, ...] | None = None  # OQ-4: URDF / measured link lengths
    feedback_readable: bool | None = False  # OQ-5 closed: no feedback (commanded only)
    max_joint_speed_deg_s: float | None = None  # A-1: servo max speed, unconfirmed
    max_joint_accel_deg_s2: float | None = None  # A-1: safe joint acceleration, unconfirmed
    max_delta_per_tick_deg: float | None = None  # A-2: superseded by PP-1 (no streaming)
    safe_pose_deg: tuple[float, ...] | None = None  # A-3: safe/home pose, unconfirmed
    jam_load_threshold: float | None = None  # A-4 / OQ-5: load units + jam threshold, unconfirmed
    gripper: GripperConfig = field(default_factory=GripperConfig)

    def limits(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        lo = require(self.joint_min_deg, "joint lower limits", "OQ-4")
        hi = require(self.joint_max_deg, "joint upper limits", "OQ-4")
        if len(lo) != NUM_JOINTS or len(hi) != NUM_JOINTS:
            raise ConfigError(f"joint limits must have {NUM_JOINTS} entries")
        if any(a >= b for a, b in zip(lo, hi, strict=True)):
            raise ConfigError("joint_min_deg must be < joint_max_deg for every joint")
        return lo, hi


@dataclass
class OverheadCameraConfig:
    """Intel RealSense D435 (bible 5.1). Stream modes are standard D435 modes."""

    width: int = 640
    height: int = 480
    fps: int = 30
    device_serial: str | None = None  # Only needed if several RealSense devices are attached
    base_T_cam_path: Path = DATA_DIR / "calib" / "base_T_cam.yaml"
    depth_median_window: int = 5  # pixels, odd
    # Bible 5.1: lock exposure and white balance. Auto runs for settle_s after start, then
    # both are locked: to the values below if set, else to values matched to the settled
    # image (exposure: same mean brightness; white balance: neutral white table). The
    # locked values are logged; copy them here to reuse them across sessions.
    lock_color: bool = True
    settle_s: float = 2.0
    exposure: int | None = None  # RealSense RGB exposure units (0.1 ms on the D435)
    gain: int | None = None  # None = keep the sensor's current gain
    white_balance_k: int | None = None  # 2800..6500
    max_exposure: int = 320  # search cap, ~32 ms: keeps 30 FPS


@dataclass
class WristCameraConfig:
    """Panoculonlabs Trinet (bible 5.2). Semantic-only: never used for metric control."""

    device_index: int | None = None  # OQ-7: UVC vs vendor SDK / device index, unconfirmed
    # OQ-7 alternative: stable V4L2 path (e.g. /dev/v4l/by-id/usb-...-video-index0); wins
    # over device_index when set.
    device_path: str | None = None
    # OQ-8 measured 2026-10-11: the Trinet's ONLY usable mode is MJPG 1920x1080 @ 30 FPS
    # (H264 is also offered: do not use it). Capture node /dev/video6, metadata /dev/video7.
    fourcc: str | None = "MJPG"
    width: int | None = 1920
    height: int | None = 1080
    fps: int | None = 30
    retrieve_hz: float = 12.0  # decode rate (grab() runs at 30 FPS, retrieve() at this rate)
    # Startup: keep retrying the open this long (a USB drop re-enumerates in up to ~20 s,
    # measured 2026-10-11). Mid-run: the Critic waits this long for a frame grabbed after
    # its request; none -> OCCLUDED (re-query policy), never an abort while moving.
    open_wait_s: float = 45.0
    fresh_frame_timeout_s: float = 5.0
    calib_path: Path = DATA_DIR / "calib" / "trinet_fisheye.yaml"  # OQ-9: produced by tools/
    dewarp_enabled: bool = True  # bible 5.2: dewarp before the VLM
    dewarp_balance: float = 0.0  # 0 = crop to valid pixels, 1 = keep full FOV
    # Optional Dewarper pre-crop (rarely needed). The Critic's gripper ROI is
    # CriticConfig.crop_box_norm (OQ-12); do not set both to the same box.
    crop_xywh: tuple[int, int, int, int] | None = None


@dataclass
class BlockDetectorConfig:
    """Live demo detector: ONE object on a white table, no YOLO (lbyr/perception/block.py).

    Tuning values, not hardware specs: check them on the real table with
    ``python tools/overhead_snapshot.py --detect`` (prints areas / chroma, saves the mask).
    A pixel is "object" when it is NOT white: Lab chroma >= ``lab_chroma_min`` or
    L <= ``lab_l_max`` (OpenCV 8-bit Lab), or with ``color_space = "hsv"``:
    S >= ``hsv_s_min`` or V <= ``hsv_v_max``.
    """

    label: str = "wood_block"  # class name the planners see
    object_id: int = 1  # stable detection id (one object)
    color_space: str = "lab"  # "lab" | "hsv"
    lab_chroma_min: float = 18.0  # wood ~25-40; white table after WB lock < 10
    lab_l_max: int = 70  # very dark pixels count as object (shadows are lighter)
    hsv_s_min: int = 60
    hsv_v_max: int = 60
    morph_kernel_px: int = 5  # open + close
    min_area_px: float = 150.0  # contour area filter (640x480)
    max_area_px: float = 8000.0  # rejects the arm / big clutter
    border_px: int = 8  # ignored frame margin when no workspace polygon is defined
    depth_window_px: int = 7  # median depth window around the centroid (odd)
    # Workspace polygon (overhead pixels). None = poses.yaml `workspace_polygon_px`
    # (tools/define_zones.py --workspace), else the whole frame minus border_px.
    workspace_polygon_px: tuple[tuple[float, float], ...] | None = None

    def validate(self) -> None:
        if self.color_space not in ("lab", "hsv"):
            raise ConfigError("detector.color_space must be 'lab' or 'hsv'")
        if not 0 < self.min_area_px < self.max_area_px:
            raise ConfigError("detector: need 0 < min_area_px < max_area_px")
        if self.depth_window_px < 1 or self.depth_window_px % 2 == 0:
            raise ConfigError("detector.depth_window_px must be a positive odd integer")


@dataclass
class CameraConfig:
    overhead: OverheadCameraConfig = field(default_factory=OverheadCameraConfig)
    wrist: WristCameraConfig = field(default_factory=WristCameraConfig)


@dataclass
class PlannerConfig:
    """Qwen2.5-VL-3B planner (bible 11.1)."""

    timeout_s: float = 3.0  # LOCKED: 3 s hard timeout
    temperature: float = 0.0  # LOCKED: temperature 0
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    model_path: Path | None = None  # local weights only; no downloads at runtime
    max_new_tokens: int = 256
    vl_dtype: str = "fp16"  # Qwen2.5-VL weights dtype: "fp16" | "bf16" (lbyr.demo uses bf16)
    # "rules" (RuleBasedPlanner, no model, no GPU) | "qwen". Default rules: Qwen is NOT loaded
    # unless backend = "qwen" (measured 11.2 s per full plan on the Orin NX, bible 9.2).
    backend: str = "rules"
    fallback_after_timeouts: int = 3  # qwen -> rules after this many timeouts in a row
    device: str = "cuda"
    max_image_pixels: int = 640 * 480  # overhead frame is downscaled above this (latency)
    warmup_timeout_s: float = 60.0  # startup warm-up generation (first CUDA call is slow)
    # "Language mode" (demo intro ONLY, never in the FSM loop): text-only prompt, compact
    # output schema. Its own timeout; every in-loop call keeps the 3 s ``timeout_s``.
    qwen_timeout_s: float = 8.0
    language_max_new_tokens: int = 64
    language_mode: bool = False  # load the language planner at startup (demo intro)
    language_backend: str = "qwen3_text"  # "qwen3_text" | "qwen_vl" (uses model_path)
    # Local Qwen3 weights (no downloads). Alternative: ~/models/qwen3-0.6b (M-2).
    language_model_path: Path = Path("~/models/qwen3-1.7b")
    language_dtype: str = "bf16"  # Qwen3 native dtype
    # Memory guard: with backend = "qwen" or language_mode, refuse to start the episode loop if
    # free system RAM (unified memory) after loading all models is below this.
    qwen_min_free_ram_mb: float = 1536.0  # 1.5 GB

    def validate(self) -> None:
        if self.temperature != 0.0:
            raise ConfigError("Planner temperature must be 0 (bible 11.1)")
        if not 0.0 < self.qwen_timeout_s <= 30.0:
            raise ConfigError("planner.qwen_timeout_s must be in (0, 30] s")
        if self.language_max_new_tokens < 1:
            raise ConfigError("planner.language_max_new_tokens must be >= 1")
        if self.language_backend not in LANGUAGE_BACKENDS:
            raise ConfigError(f"planner.language_backend must be one of {LANGUAGE_BACKENDS}")
        if self.vl_dtype not in MODEL_DTYPES:
            raise ConfigError(f"planner.vl_dtype must be one of {MODEL_DTYPES}")
        if self.language_dtype not in MODEL_DTYPES:
            raise ConfigError(f"planner.language_dtype must be one of {MODEL_DTYPES}")
        if self.qwen_min_free_ram_mb < 0.0:
            raise ConfigError("planner.qwen_min_free_ram_mb must be >= 0")
        if not 0.0 < self.timeout_s <= 3.0:
            raise ConfigError("Planner timeout must be in (0, 3] s (bible 11.1)")
        if self.backend not in PLANNER_BACKENDS:
            raise ConfigError(f"planner.backend must be one of {PLANNER_BACKENDS}")
        if self.fallback_after_timeouts < 1:
            raise ConfigError("planner.fallback_after_timeouts must be >= 1")
        if self.max_new_tokens < 1:
            raise ConfigError("planner.max_new_tokens must be >= 1")
        if self.max_image_pixels < 28 * 28:
            raise ConfigError("planner.max_image_pixels is too small")
        if self.warmup_timeout_s <= 0.0:
            raise ConfigError("planner.warmup_timeout_s must be > 0")


@dataclass
class CriticConfig:
    """Moondream2 Critic (bible 11.2)."""

    enabled: bool = True  # False = "Critic OFF" baseline condition (bible 15)
    timeout_s: float = 2.0  # explicit timeout; expected inference 0.3-0.8 s
    temperature: float = 0.0  # LOCKED: temperature 0
    model_path: Path | None = None  # local weights (FP16 or GGUF-Q4); no downloads
    lora_adapter_path: Path | None = None
    request_queue_size: int = 4
    device: str = "cuda"
    max_new_tokens: int = 8  # the answer is ONE word
    warmup_timeout_s: float = 60.0  # startup warm-up query (first CUDA call is slow)
    # "bf16" (Moondream's native dtype; the Orin supports it) | "fp16" (weights cast +
    # crop-dtype patch). M-1 resolved 2026-10-10: bf16 + 378x378 crop (0.85 s vs 1.35 s).
    dtype: str = "bf16"
    # Runtime input = dewarped wrist frame -> gripper region RESIZED (INTER_AREA) to 378x378
    # (one Moondream tile).
    crop_enabled: bool = True
    crop_size_px: int = 378
    # OQ-12: gripper bite-zone box (x0, y0, x1, y1), NORMALIZED 0..1 in the (dewarped) frame.
    # None = width-centred full-height square (1080x1080 of a 1920x1080 frame). Tune it on the
    # mounted Trinet (lbyr.demo --crop overrides it). Keep it square in PIXELS: on 16:9,
    # x1 - x0 = (y1 - y0) * 9 / 16, else the resize stretches the image.
    crop_box_norm: tuple[float, float, float, float] | None = None

    def validate(self) -> None:
        if self.temperature != 0.0:
            raise ConfigError("Critic temperature must be 0 (bible 21.4)")
        if self.dtype not in CRITIC_DTYPES:
            raise ConfigError(f"critic.dtype must be one of {CRITIC_DTYPES}")
        if self.crop_size_px < 28:
            raise ConfigError("critic.crop_size_px is too small")
        if self.crop_box_norm is not None:
            x0, y0, x1, y1 = self.crop_box_norm
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ConfigError(
                    "critic.crop_box_norm must be 0 <= x0 < x1 <= 1, 0 <= y0 < y1 <= 1"
                )
        if self.timeout_s <= 0.0:
            raise ConfigError("Critic timeout must be > 0 (bible 11.2)")
        if self.max_new_tokens < 1:
            raise ConfigError("Critic max_new_tokens must be >= 1")
        if self.warmup_timeout_s <= 0.0:
            raise ConfigError("Critic warmup_timeout_s must be > 0")


@dataclass
class MotionConfig:
    """PP-1 (2026-10-10): the bridge firmware is the trajectory generator. The host sends ONE
    ``m`` per motion segment and waits for ``DONE``; the motion thread only dispatches
    segments and polls status (``q``) at ``poll_hz``."""

    mode: str = "taught"  # "taught" (poses.yaml, demo path) | "ik" (needs OQ-4 + hand-eye)
    poses_path: Path = CALIB_DIR / "poses.yaml"
    poll_hz: float = 20.0  # status poll / segment dispatch rate
    done_timeout_factor: float = 1.5  # DONE timeout = predicted x factor + margin -> LINK
    done_timeout_margin_s: float = 0.5
    rate_hz: float = 50.0  # bible 11.4; only used by motion/trajectory.py (offline planning)
    dev_speed_cap: float = 0.30  # bible 8: development speed cap 30%
    stage_speed_cap: float = 0.50  # bible 8: stage speed cap 50%
    stage_mode: bool = False
    jam_position_error_deg: float = 10.0  # measured-vs-commanded error -> JAMMED (mock only)
    jitter_window: int = 2000  # number of recent ticks kept for jitter stats
    spin_margin_s: float = 0.0015  # final busy-wait window for deadline scheduling

    @property
    def speed_cap(self) -> float:
        return self.stage_speed_cap if self.stage_mode else self.dev_speed_cap

    @property
    def period_s(self) -> float:
        return 1.0 / self.rate_hz

    def validate(self) -> None:
        if self.dev_speed_cap > 0.30 or self.stage_speed_cap > 0.50:
            raise ConfigError("Speed caps exceed bible Section 8 (dev 30%, stage 50%)")
        if self.rate_hz <= 0 or self.poll_hz <= 0:
            raise ConfigError("rate_hz and poll_hz must be > 0")
        if self.mode not in MOTION_MODES:
            raise ConfigError(f"motion.mode must be one of {MOTION_MODES}")
        if self.done_timeout_factor < 1.0 or self.done_timeout_margin_s < 0.0:
            raise ConfigError("DONE timeout must be >= the predicted duration")

    def done_timeout_s(self, predicted_s: float) -> float:
        return predicted_s * self.done_timeout_factor + self.done_timeout_margin_s


@dataclass
class RecoveryConfig:
    max_total_recoveries: int = 2  # bible 10: RECOVER(k), k <= 2 per episode
    occluded_wrist_pitch_delta_deg: float = 10.0  # tuning, NOT from bible: cam re-position step
    critic_requery_max: int = 1  # TIMEOUT/INVALID verdicts: re-query once, then ABORT


@dataclass
class SkillConfig:
    """Task-level tuning (not hardware specs). Cartesian values in base frame, mm."""

    default_standoff_mm: float = 40.0  # bible 11.1 example
    lift_height_mm: float = 60.0
    transport_height_mm: float = 80.0
    place_release_height_mm: float = 15.0
    default_closure_pct: float = 80.0  # bible 11.1 example
    # --induce-slip (trials): closure % of the FIRST grasp only. Tuning, not a hardware value:
    # check on the bench that the block is held loosely enough to slip.
    induced_slip_closure_pct: float = 50.0


@dataclass
class LoggingConfig:
    episodes_dir: Path = DATA_DIR / "episodes"
    log_every_tick: bool = False  # True = also log every 10 Hz FSM tick
    tegrastats_interval_ms: int = 1000


@dataclass
class DemoConfig:
    """``python -m lbyr.demo`` (live demo, taught poses). Tuning values, not hardware specs."""

    instruction: str = "pick up the wooden block and place it in the place zone"
    qwen_path: Path = Path("~/models/qwen2.5-vl-3b")  # local weights only
    qwen_dtype: str = "bf16"
    # Demo intro plan ONLY (before the motion sequence, arm idle). Not an in-loop call.
    qwen_timeout_s: float = 30.0
    # Language prompting (web text box / --task): one JSON move/none answer, under
    # qwen_timeout_s. ~60 tokens for the JSON + one-sentence reason.
    task_max_new_tokens: int = 96
    moondream_path: Path = Path("~/models/moondream2")
    # Qwen stays resident (loaded once): unloading does not return memory on the Jetson
    # (RAM stayed ~14.5 GB used after "freed Qwen") and a reload costs ~11 s. Below this
    # free RAM after both models load: WARNING only, never a refusal.
    warn_free_ram_mb: float = 800.0
    regrasp_step_pct: float = 15.0  # bible 13: +15% closure per re-grasp
    # --induce-slip: gripper angle of the FIRST grasp only, zero-relative degrees (poses.yaml
    # gripper: open -40, closed 0). Operator range "clearly loose": -20 .. -25.
    induce_slip_gripper_deg: float = -22.0
    # Rehearsal / --critic scripted: verdicts returned in order (last repeats).
    scripted_verdicts: tuple[str, ...] = ("GRASPED",)
    snapshot_dir: Path = Path("runs")  # 'w' key / GET /snapshot: runs/<ts>/snap_<n>_*.jpg
    max_regrasps: int = 2  # bible 10: k <= 2
    http_host: str = "0.0.0.0"
    http_port: int = 8000
    # Wrist camera: None = first /dev/video* capture node that is not the D435 (OQ-7).
    wrist_device: str | None = None
    settle_s: float = 0.3  # wait after DONE before grabbing the wrist frame for the Critic


@dataclass
class AppConfig:
    arm: ArmConfig = field(default_factory=ArmConfig)
    cameras: CameraConfig = field(default_factory=CameraConfig)
    detector: BlockDetectorConfig = field(default_factory=BlockDetectorConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    skills: SkillConfig = field(default_factory=SkillConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    fsm_rate_hz: float = 10.0  # bible 9.3

    def validate(self) -> None:
        self.planner.validate()
        self.critic.validate()
        self.motion.validate()
        self.detector.validate()


def mock_profile() -> AppConfig:
    """SIMULATION-ONLY configuration for ``--mock`` and tests.

    These numbers are invented so the software can run without hardware. They are NOT
    ROT3U values and must never be copied into a real-hardware config.
    """
    cfg = AppConfig()
    cfg.arm = ArmConfig(
        simulated=True,
        joint_min_deg=(-150.0, -90.0, -135.0, -110.0, -150.0, 0.0),  # FAKE
        joint_max_deg=(150.0, 90.0, 135.0, 110.0, 150.0, 90.0),  # FAKE
        link_lengths_mm=(90.0, 140.0, 140.0, 80.0),  # FAKE: base height, upper arm, forearm, tool
        feedback_readable=True,  # FAKE
        max_joint_speed_deg_s=400.0,  # FAKE (fast so mock episodes finish quickly)
        max_joint_accel_deg_s2=1200.0,  # FAKE
        max_delta_per_tick_deg=10.0,  # FAKE
        safe_pose_deg=(0.0, -30.0, 90.0, 30.0, 0.0, 0.0),  # FAKE
        gripper=GripperConfig(open_deg=0.0, closed_deg=90.0),  # FAKE
    )
    cfg.cameras.wrist = WristCameraConfig(
        device_index=None, width=640, height=480, fps=30, dewarp_enabled=True
    )
    return cfg


def preflight(cfg: AppConfig) -> list[str]:
    """Everything that still blocks a real-hardware run, as operator-readable lines.

    Blocks only on what that run really needs:
    - always: data/calib/joints.yaml present, valid and ``verified: true`` (zeros checked
      against the board's ``list``, A-7); the bridge serial port (OQ-10);
    - motion.mode = "taught": data/calib/poses.yaml with every required pose / zone;
    - motion.mode = "ik": joint limits / link lengths (OQ-4), gripper mapping (OQ-6),
      safe pose (A-3);
    - Critic enabled: the Trinet device index and mode (OQ-7/8).
    Speed limits come from the firmware (``V`` line) and feedback is closed (OQ-5: none),
    so neither blocks any more. Imports are local to avoid import cycles.
    """
    from lbyr.arm.calib import load_joints
    from lbyr.arm.serial_arm import SerialArm
    from lbyr.errors import CalibrationError

    missing: list[str] = []

    def check(name: str, fn: Callable[[], object]) -> None:
        try:
            fn()
        except HardwareNotConfirmedError as e:
            missing.append(f"{name}: {e.what} ({e.question})")
        except (CalibrationError, ConfigError) as e:
            missing.append(f"{name}: {e}")

    joints = None
    try:
        joints = load_joints(cfg.arm.joints_path)
        if not joints.verified:
            missing.append(
                f"joints: {cfg.arm.joints_path} not verified: compare zero_us with the board's "
                "`list` (tools/arm_gui.py 'Sync sliders'), then set verified: true (A-7)"
            )
    except CalibrationError as e:
        missing.append(f"joints: {e}")
    if not cfg.arm.simulated:
        check("bridge", lambda: SerialArm(cfg.arm).check_confirmed())
    if cfg.motion.mode == "taught":
        if joints is not None:
            from lbyr.motion.taught import load_poses

            try:
                poses = load_poses(cfg.motion.poses_path, joints)
                missing += [f"poses: {cfg.motion.poses_path}: {m}" for m in poses.missing()]
            except CalibrationError as e:
                missing.append(f"poses: {e}")
    else:
        from lbyr.motion.ik import IkpyIK

        check("IK", lambda: IkpyIK(cfg.arm))
        check("gripper mapping", lambda: cfg.arm.gripper.closure_to_deg(0.0))
        check("safe pose", lambda: require(cfg.arm.safe_pose_deg, "safe pose", "A-3"))
    if cfg.critic.enabled:
        from lbyr.perception.wrist import TrinetWristCamera

        check("wrist camera", lambda: TrinetWristCamera(cfg.cameras.wrist))
    return missing
