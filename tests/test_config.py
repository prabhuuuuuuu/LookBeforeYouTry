import pytest

from lbyr.config import AppConfig, ArmConfig, GripperConfig, mock_profile, require
from lbyr.errors import ConfigError, HardwareNotConfirmedError


def test_default_arm_config_has_no_hardware_values() -> None:
    arm = ArmConfig()
    with pytest.raises(HardwareNotConfirmedError, match="OQ-4"):
        arm.limits()
    with pytest.raises(HardwareNotConfirmedError, match="OQ-6"):
        arm.gripper.closure_to_deg(50.0)


def test_require_raises_with_question() -> None:
    with pytest.raises(HardwareNotConfirmedError) as exc:
        require(None, "servo voltage", "OQ-3")
    assert exc.value.question == "OQ-3"
    assert require(7.4, "x", "OQ-3") == 7.4


def test_gripper_mapping_clamps() -> None:
    g = GripperConfig(open_deg=10.0, closed_deg=90.0)
    assert g.closure_to_deg(0.0) == 10.0
    assert g.closure_to_deg(100.0) == 90.0
    assert g.closure_to_deg(150.0) == 90.0
    assert g.closure_to_deg(50.0) == pytest.approx(50.0)


def test_locked_rules_validated() -> None:
    cfg = AppConfig()
    cfg.validate()
    cfg.planner.temperature = 0.2
    with pytest.raises(ConfigError):
        cfg.validate()
    cfg = AppConfig()
    cfg.planner.timeout_s = 5.0
    with pytest.raises(ConfigError):
        cfg.validate()
    cfg = AppConfig()
    cfg.motion.dev_speed_cap = 0.8
    with pytest.raises(ConfigError):
        cfg.validate()


def test_mock_profile_is_marked_simulated() -> None:
    cfg = mock_profile()
    cfg.validate()
    assert cfg.arm.simulated
    assert AppConfig().arm.simulated is False


# ------------------------------------------------------------------ preflight
def _ready_calib(tmp_path):  # type: ignore[no-untyped-def]
    import yaml

    from lbyr.motion.taught import save_pose

    j = yaml.safe_load(open("data/calib/joints.yaml"))
    j["verified"] = True
    (tmp_path / "joints.yaml").write_text(yaml.safe_dump(j))
    poses = tmp_path / "poses.yaml"
    for name in ("home", "place_pre", "place", "zone_A_pre", "zone_A_grasp"):
        save_pose(poses, name, (0, 0, 0, 0, 0, 0))
    data = yaml.safe_load(poses.read_text())
    data["gripper"] = {"open_deg": 0, "closed_deg": 40}
    data["zones"] = {"A": {"polygon_px": [[0, 0], [10, 0], [10, 10]]}}
    poses.write_text(yaml.safe_dump(data))
    cfg = AppConfig()
    cfg.arm.joints_path, cfg.motion.poses_path = tmp_path / "joints.yaml", poses
    cfg.arm.serial_port = "/dev/ttyACM0"
    return cfg


def test_preflight_defaults_block_on_calibration_port_poses_and_wrist() -> None:
    from lbyr.config import preflight

    missing = preflight(AppConfig())
    text = "\n".join(missing)
    assert "joints.yaml not verified" in text and "OQ-10" in text
    assert "pose 'home'" in text and "OQ-7" in text
    assert "OQ-4" not in text and "A-1" not in text and "OQ-5" not in text  # taught mode


def test_preflight_ready_taught_setup_passes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from lbyr.config import preflight

    cfg = _ready_calib(tmp_path)
    cfg.critic.enabled = False
    assert preflight(cfg) == []
    cfg.critic.enabled = True
    assert [m for m in preflight(cfg) if "wrist camera" in m] == preflight(cfg)


def test_preflight_ik_mode_needs_link_lengths_not_poses(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from lbyr.config import preflight

    cfg = _ready_calib(tmp_path)
    cfg.critic.enabled = False
    cfg.motion.mode = "ik"
    cfg.motion.poses_path = tmp_path / "missing.yaml"
    text = "\n".join(preflight(cfg))
    assert "OQ-4" in text and "poses" not in text


def test_confirmed_values_are_set() -> None:
    arm = ArmConfig()
    assert (arm.servo_bus, arm.servo_voltage_v, arm.host_baud, arm.feedback_readable) == (
        "pwm",
        6.0,
        115200,
        False,
    )
    assert AppConfig().motion.mode == "taught"
