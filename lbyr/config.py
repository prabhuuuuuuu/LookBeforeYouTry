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
CRITIC_DTYPES: tuple[str, ...] = ("fp16", "bf16")


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
    servo_bus: str | None = None  # OQ-1: "serial" (Feetech-style) or "pwm", unconfirmed
    servo_ids: tuple[int, ...] | None = None  # OQ-2: servo IDs, unconfirmed
    servo_baud: int | None = None  # OQ-2: servo bus baud rate, unconfirmed
    servo_voltage_v: float | None = None  # OQ-3: servo PSU voltage, unconfirmed. NEVER assume 5 V.
    serial_port: str | None = None  # OQ-10: USB port plan (/dev/ttyACM0?), unconfirmed
    host_baud: int | None = None  # Bridge firmware TBD; bible Section 6 says >= 115200
    joint_min_deg: tuple[float, ...] | None = None  # OQ-4 / bible 3.1 #4: joint limits
    joint_max_deg: tuple[float, ...] | None = None  # OQ-4 / bible 3.1 #4: joint limits
    link_lengths_mm: tuple[float, ...] | None = None  # OQ-4: URDF / measured link lengths
    feedback_readable: bool | None = None  # OQ-5: position/load feedback readable?, unconfirmed
    max_joint_speed_deg_s: float | None = None  # A-1: servo max speed, unconfirmed
    max_joint_accel_deg_s2: float | None = None  # A-1: safe joint acceleration, unconfirmed
    max_delta_per_tick_deg: float | None = None  # A-2: per-tick clamp (bridge + mock), unconfirmed
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


@dataclass
class WristCameraConfig:
    """Panoculonlabs Trinet (bible 5.2). Semantic-only: never used for metric control."""

    device_index: int | None = None  # OQ-7: UVC vs vendor SDK / device index, unconfirmed
    width: int | None = None  # OQ-8: reliable resolution on Jetson, unconfirmed (start at 720p)
    height: int | None = None  # OQ-8
    fps: int | None = None  # OQ-8
    calib_path: Path = DATA_DIR / "calib" / "trinet_fisheye.yaml"  # OQ-9: produced by tools/
    dewarp_enabled: bool = True  # bible 5.2: dewarp before the VLM
    dewarp_balance: float = 0.0  # 0 = crop to valid pixels, 1 = keep full FOV
    crop_xywh: tuple[int, int, int, int] | None = None  # OQ-12: gripper ROI, depends on mount


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
    # "rules" (RuleBasedPlanner, no model, no GPU) | "qwen". Default rules: Qwen is NOT loaded
    # unless backend = "qwen" (measured 11.2 s per full plan on the Orin NX, bible 9.2).
    backend: str = "rules"
    fallback_after_timeouts: int = 3  # qwen -> rules after this many timeouts in a row
    device: str = "cuda"
    max_image_pixels: int = 640 * 480  # overhead frame is downscaled above this (latency)
    warmup_timeout_s: float = 60.0  # startup warm-up generation (first CUDA call is slow)
    # Qwen "language mode" (demo intro ONLY, never in the FSM loop): text-only prompt, compact
    # output schema. Its own timeout; every in-loop call keeps the 3 s ``timeout_s``.
    qwen_timeout_s: float = 8.0
    language_max_new_tokens: int = 64

    def validate(self) -> None:
        if self.temperature != 0.0:
            raise ConfigError("Planner temperature must be 0 (bible 11.1)")
        if not 0.0 < self.qwen_timeout_s <= 30.0:
            raise ConfigError("planner.qwen_timeout_s must be in (0, 30] s")
        if self.language_max_new_tokens < 1:
            raise ConfigError("planner.language_max_new_tokens must be >= 1")
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
    # "fp16" (weights cast + crop-dtype patch) | "bf16" (the model's native dtype, no patch).
    # Speed experiment in tools/bench_models.py; default unchanged until a winner is picked.
    dtype: str = "fp16"

    def validate(self) -> None:
        if self.temperature != 0.0:
            raise ConfigError("Critic temperature must be 0 (bible 21.4)")
        if self.dtype not in CRITIC_DTYPES:
            raise ConfigError(f"critic.dtype must be one of {CRITIC_DTYPES}")
        if self.timeout_s <= 0.0:
            raise ConfigError("Critic timeout must be > 0 (bible 11.2)")
        if self.max_new_tokens < 1:
            raise ConfigError("Critic max_new_tokens must be >= 1")
        if self.warmup_timeout_s <= 0.0:
            raise ConfigError("Critic warmup_timeout_s must be > 0")


@dataclass
class MotionConfig:
    rate_hz: float = 50.0  # bible 11.4
    dev_speed_cap: float = 0.30  # bible 8: development speed cap 30%
    stage_speed_cap: float = 0.50  # bible 8: stage speed cap 50%
    stage_mode: bool = False
    position_tolerance_deg: float = 2.0  # tuning: "arrived" tolerance
    motion_timeout_margin_s: float = 2.0  # tuning: extra time beyond profile duration -> JAMMED
    jam_position_error_deg: float = 10.0  # tuning: commanded-vs-actual error -> JAMMED
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
        if self.rate_hz <= 0:
            raise ConfigError("rate_hz must be > 0")


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


@dataclass
class LoggingConfig:
    episodes_dir: Path = DATA_DIR / "episodes"
    log_every_tick: bool = False  # True = also log every 10 Hz FSM tick
    tegrastats_interval_ms: int = 1000


@dataclass
class AppConfig:
    arm: ArmConfig = field(default_factory=ArmConfig)
    cameras: CameraConfig = field(default_factory=CameraConfig)
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
