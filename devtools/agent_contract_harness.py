
#!/usr/bin/env python3
"""Offline smoke test for the Ridges agent_main(input) -> unified diff contract."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

TASK = "Update src/example.py so greet(name) returns a friendly greeting."


def run(command: list[str], cwd: Path, capture: bool = False) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture else subprocess.DEVNULL,
    )
    return result.stdout if capture else ""


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


def self_test_agent(_input: dict) -> str:
    """Fake agent: modifies an existing file and creates a new one."""
    root = Path.cwd()

    (root / "src" / "example.py").write_text(
        'def greet(name):\n    return f"Hello, {name}!"\n',
        encoding="utf-8",
    )

    (root / "src" / "new_module.py").write_text(
        "def answer():\n    return 42\n",
        encoding="utf-8",
    )

    # Include untracked files in the generated diff.
    run(["git", "add", "-N", "--", "."], root)

    return run(
        ["git", "diff", "--binary", "--no-ext-diff"],
        root,
        capture=True,
    )


def load_agent(path: Path):
    spec = importlib.util.spec_from_file_location(
        "ridges_agent_under_test", path
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not callable(getattr(module, "agent_main", None)):
        raise RuntimeError(
            "agent.py must define a callable agent_main(input)"
        )

    # Prevent the conventional inference helpers from contacting a provider.
    if hasattr(module, "inference"):
        module.inference = lambda *args, **kwargs: None

    if hasattr(module, "embedding"):
        module.embedding = lambda *args, **kwargs: None

    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        type=Path,
        help="Path to agent.py; omit to test the harness itself",
    )
    parser.add_argument("--task", default=TASK)
    args = parser.parse_args()

    agent_path = args.agent.resolve() if args.agent else None

    agent = (
        load_agent(agent_path)
        if agent_path
        else SimpleNamespace(agent_main=self_test_agent)
    )

    with tempfile.TemporaryDirectory(
        prefix="ridges-agent-contract-"
    ) as temp:
        root = Path(temp)
        workspace = root / "task-workspace"
        verifier = root / "clean-verifier"

        (workspace / "src").mkdir(parents=True)

        (workspace / "src" / "example.py").write_text(
            "def greet(name):\n    return name\n",
            encoding="utf-8",
        )

        # Establish the original task state.
        run(["git", "init", "-q"], workspace)
        run(["git", "config", "user.name", "Ridges Harness"], workspace)
        run([
            "git", "config", "user.email",
            "harness@example.invalid",
        ], workspace)
        run(["git", "add", "-A"], workspace)
        run(["git", "commit", "-qm", "baseline"], workspace)

        # Make an untouched copy for independent patch verification.
        shutil.copytree(workspace, verifier)

        original_cwd = Path.cwd()

        try:
            os.chdir(workspace)
            patch = agent.agent_main({
                "problem_statement": args.task
            })
        finally:
            os.chdir(original_cwd)

        if not isinstance(patch, str) or not patch.strip():
            raise RuntimeError(
                "agent_main must return a non-empty unified diff string"
            )

        patch_file = root / "agent.patch"
        patch_file.write_text(patch, encoding="utf-8")

        # Check and apply the returned patch in the clean copy.
        run(
            ["git", "apply", "--check", str(patch_file)],
            verifier,
        )
        run(["git", "apply", str(patch_file)], verifier)

        # Ensure the patch reproduces all workspace file changes,
        # including newly created files.
        if snapshot(workspace) != snapshot(verifier):
            raise RuntimeError(
                "Returned patch does not reproduce the workspace. "
                "Check modified, deleted, and NEW files."
            )

    print("PASS: agent_main returned a valid, complete patch.")

    if not agent_path:
        print("SELF-TEST: harness verified using a fake agent.")
        print("Your actual agent.py was not run.")
    else:
        print(f"AGENT TESTED: {agent_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}")
        raise SystemExit(1)
