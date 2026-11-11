import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from devtools.test_runner import detect_test_command, run_tests


class TestRunnerTests(unittest.TestCase):

    def test_detects_pytest_suite_when_available(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "test_example.py").write_text(
                "def test_example():\n"
                "    assert 1 + 1 == 2\n",
                encoding="utf-8",
            )

            with patch(
                "devtools.test_runner.importlib.util.find_spec",
                return_value=object(),
            ):
                command = detect_test_command(root)

            self.assertEqual(
                command,
                [sys.executable, "-m", "pytest", "-q"],
            )

    def test_falls_back_to_unittest_when_pytest_unavailable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "test_example.py").write_text(
                "import unittest\n"
                "class ExampleTest(unittest.TestCase):\n"
                "    def test_example(self):\n"
                "        self.assertEqual(1 + 1, 2)\n",
                encoding="utf-8",
            )

            with patch(
                "devtools.test_runner.importlib.util.find_spec",
                return_value=None,
            ):
                command = detect_test_command(root)

            self.assertEqual(
                command,
                [sys.executable, "-m", "unittest", "discover", "-v"],
            )

    def test_runs_passing_command(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_tests(
                Path(temp),
                command=[sys.executable, "-c", "print('tests passed')"],
            )

            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["success"])
            self.assertEqual(result["exit_code"], 0)
            self.assertIn("tests passed", result["output"])

    def test_detects_failing_command(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_tests(
                Path(temp),
                command=[sys.executable, "-c", "raise SystemExit(1)"],
            )

            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["success"])
            self.assertEqual(result["exit_code"], 1)

    def test_stops_command_after_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_tests(
                Path(temp),
                timeout=1,
                command=[
                    sys.executable,
                    "-c",
                    "import time; time.sleep(5)",
                ],
            )

            self.assertEqual(result["status"], "timeout")
            self.assertFalse(result["success"])
            self.assertIn("time limit", result["output"])

    def test_skips_repository_without_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_tests(Path(temp))

            self.assertEqual(result["status"], "skipped")
            self.assertFalse(result["success"])
            self.assertIsNone(result["command"])

    def test_rejects_invalid_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = Path(temp) / "missing"

            with self.assertRaises(ValueError):
                run_tests(missing)

    def test_rejects_invalid_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                run_tests(Path(temp), timeout=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
