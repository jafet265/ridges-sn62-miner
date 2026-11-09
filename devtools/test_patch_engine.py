import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import generate_unified_diff


def git(cwd, *args):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(root).parts
    }


class PatchEngineTests(unittest.TestCase):

    def test_modified_deleted_and_new_files(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            workspace = base / "workspace"
            verifier = base / "verifier"

            (workspace / "src").mkdir(parents=True)

            (workspace / "src" / "existing.py").write_text(
                "def answer():\n    return 1\n",
                encoding="utf-8",
            )

            (workspace / "src" / "deleted.py").write_text(
                "OLD_FILE = True\n",
                encoding="utf-8",
            )

            git(workspace, "init", "-q")
            git(workspace, "config", "user.name", "Patch Test")
            git(workspace, "config", "user.email", "test@example.invalid")
            git(workspace, "add", "-A")
            git(workspace, "commit", "-qm", "baseline")

            # Preserve a clean copy for independent patch application.
            shutil.copytree(workspace, verifier)

            # Modify an existing file.
            (workspace / "src" / "existing.py").write_text(
                "def answer():\n    return 42\n",
                encoding="utf-8",
            )

            # Delete a tracked file.
            (workspace / "src" / "deleted.py").unlink()

            # Create a new file.
            (workspace / "src" / "new.py").write_text(
                "NEW_FILE = True\n",
                encoding="utf-8",
            )

            expected = snapshot(workspace)

            patch = generate_unified_diff(workspace)

            self.assertTrue(patch.strip())
            self.assertIn("existing.py", patch)
            self.assertIn("deleted.py", patch)
            self.assertIn("new.py", patch)

            patch_file = base / "result.patch"
            patch_file.write_text(patch, encoding="utf-8")

            # Verify and apply the patch to the clean repository.
            git(verifier, "apply", "--check", str(patch_file))
            git(verifier, "apply", str(patch_file))

            self.assertEqual(expected, snapshot(verifier))


    def test_no_changes_returns_clear_error(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)

            (workspace / "example.py").write_text(
                "VALUE = 1\\n",
                encoding="utf-8",
            )

            git(workspace, "init", "-q")
            git(workspace, "config", "user.name", "Patch Test")
            git(workspace, "config", "user.email", "test@example.invalid")
            git(workspace, "add", "-A")
            git(workspace, "commit", "-qm", "baseline")

            with self.assertRaisesRegex(
                RuntimeError, "No file changes"
            ):
                generate_unified_diff(workspace)


if __name__ == "__main__":
    unittest.main(verbosity=2)
