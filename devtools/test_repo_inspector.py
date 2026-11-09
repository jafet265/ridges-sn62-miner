import subprocess
import tempfile
import unittest
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from devtools.repo_inspector import build_task_context


def git(root, *args):
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


class RepoInspectorTests(unittest.TestCase):

    def create_repository(self, root):
        (root / "src").mkdir()
        (root / "tests").mkdir()

        (root / "README.md").write_text(
            "# Example project\n",
            encoding="utf-8",
        )

        (root / "src" / "calculator.py").write_text(
            "def add(a, b):\n    return a + b\n",
            encoding="utf-8",
        )

        (root / "tests" / "test_calculator.py").write_text(
            "def test_add():\n    assert True\n",
            encoding="utf-8",
        )

        (root / ".env").write_text(
            "API_KEY=DO_NOT_EXPOSE\n",
            encoding="utf-8",
        )

        git(root, "init", "-q")
        git(root, "config", "user.name", "Inspector Test")
        git(root, "config", "user.email", "test@example.invalid")
        git(root, "add", "-A")
        git(root, "commit", "-qm", "baseline")

    def test_collects_task_and_project_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)

            context = build_task_context(
                {
                    "problem_statement":
                        "Fix the calculator implementation."
                },
                repo_dir=root,
            )

            self.assertIn("Fix the calculator implementation.", context)
            self.assertIn("src/calculator.py", context)
            self.assertIn("def add(a, b):", context)
            self.assertIn("tests/test_calculator.py", context)
            self.assertNotIn("DO_NOT_EXPOSE", context)
            self.assertNotIn("API_KEY", context)

    def test_rejects_missing_problem_statement(self):
        with self.assertRaisesRegex(ValueError, "problem_statement"):
            build_task_context({})

    def test_respects_total_context_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)

            context = build_task_context(
                {"problem_statement": "Inspect this project."},
                repo_dir=root,
                max_total_chars=1500,
            )

            self.assertLessEqual(len(context), 1500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
