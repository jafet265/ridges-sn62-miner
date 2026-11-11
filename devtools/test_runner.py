"""Run a repository's Python tests safely with a timeout."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


DEFAULT_TIMEOUT = 120


def detect_test_command(repo_dir: Path) -> list[str] | None:
    """Detect a suitable Python test command."""

    root = Path(repo_dir).resolve()

    if not root.is_dir():
        raise ValueError(f"Repository directory does not exist: {root}")

    has_pytest_tests = (
        (root / "pytest.ini").exists()
        or (root / "conftest.py").exists()
        or any(root.glob("test_*.py"))
        or any(root.glob("*_test.py"))
        or any(root.glob("tests/test_*.py"))
    )

    pytest_available = importlib.util.find_spec("pytest") is not None

    if has_pytest_tests and pytest_available:
        return [sys.executable, "-m", "pytest", "-q"]

    has_unittest_tests = (
        any(root.glob("test*.py"))
        or any(root.glob("tests/test*.py"))
    )

    if has_unittest_tests:
        return [sys.executable, "-m", "unittest", "discover", "-v"]

    return None


def run_tests(
    repo_dir: Path,
    timeout: int = DEFAULT_TIMEOUT,
    command: list[str] | None = None,
) -> dict:
    """
    Run detected tests or an explicitly supplied command.

    Returns a structured result. A timeout or nonzero exit code
    is always considered a failure.
    """

    root = Path(repo_dir).resolve()

    if not root.is_dir():
        raise ValueError(f"Repository directory does not exist: {root}")

    if timeout < 1:
        raise ValueError("Test timeout must be at least one second.")

    test_command = command or detect_test_command(root)

    if not test_command:
        return {
            "status": "skipped",
            "success": False,
            "command": None,
            "exit_code": None,
            "output": "No supported Python test suite was detected.",
        }

    try:
        result = subprocess.run(
            test_command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        output = (result.stdout + "\n" + result.stderr).strip()

        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "success": result.returncode == 0,
            "command": test_command,
            "exit_code": result.returncode,
            "output": output,
        }

    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")

        return {
            "status": "timeout",
            "success": False,
            "command": test_command,
            "exit_code": None,
            "output": (
                f"Tests exceeded the {timeout}-second time limit.\n"
                f"{stdout}\n{stderr}"
            ).strip(),
        }

    except OSError as exc:
        return {
            "status": "error",
            "success": False,
            "command": test_command,
            "exit_code": None,
            "output": f"Could not start the test process: {exc}",
        }
