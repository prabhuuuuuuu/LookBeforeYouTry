"""Every tools/ script must show --help without hardware or heavy imports."""

import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = sorted((Path(__file__).parents[1] / "tools").glob("*.py"))


@pytest.mark.parametrize("script", TOOLS, ids=lambda p: p.name)
def test_help(script: Path) -> None:
    out = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr
    assert "usage" in out.stdout.lower()


def test_serial_probe_refuses_without_confirm() -> None:
    script = Path(__file__).parents[1] / "tools" / "serial_probe.py"
    out = subprocess.run(
        [sys.executable, str(script), "--port", "COM_DOES_NOT_EXIST"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 2
