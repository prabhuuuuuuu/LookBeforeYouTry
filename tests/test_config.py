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
