"""Build a bounded, safe context from a Ridges task repository."""

from __future__ import annotations

import subprocess
from pathlib import Path


EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__",
    "node_modules", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "dist", "build", ".tox",
}

SENSITIVE_NAMES = {
    "credentials.json", "service-account.json",
    "id_rsa", "id_ed25519",
}

SENSITIVE_PARTS = {
    "secret", "credential", "password", "token",
}

SOURCE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".cpp", ".c", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php", ".sql",
}

USEFUL_NAMES = {
    "readme.md", "pyproject.toml", "setup.py",
    "setup.cfg", "requirements.txt", "pytest.ini",
    "tox.ini", "package.json", "cargo.toml",
    "go.mod", "makefile",
}


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Git command failed: {' '.join(args)}: "
            f"{result.stderr.strip()}"
        )

    return result.stdout


def _is_excluded(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()

    if parts & EXCLUDED_DIRS:
        return True

    if name.startswith(".env"):
        return True

    if name in SENSITIVE_NAMES:
        return True

    if any(part in name for part in SENSITIVE_PARTS):
        return True

    if path.suffix.lower() in {
        ".pem", ".key", ".p12", ".pfx", ".keystore",
    }:
        return True

    return False


def _file_priority(path: Path) -> tuple[int, str]:
    name = path.name.lower()

    if name in USEFUL_NAMES:
        return (0, str(path))

    if path.suffix.lower() in SOURCE_EXTENSIONS:
        return (1, str(path))

    if "test" in name or "spec" in name:
        return (2, str(path))

    if path.suffix.lower() in {
        ".md", ".yaml", ".yml", ".toml", ".json",
    }:
        return (3, str(path))

    return (4, str(path))


def build_task_context(
    task_input: dict,
    repo_dir: str | Path | None = None,
    max_files: int = 35,
    max_file_bytes: int = 30_000,
    max_total_chars: int = 50_000,
) -> str:
    """
    Read the task statement, Git status, project file list,
    and selected source files into one bounded text context.
    """
    if not isinstance(task_input, dict):
        raise TypeError("Task input must be a dictionary.")

    problem = task_input.get("problem_statement")

    if not isinstance(problem, str) or not problem.strip():
        raise ValueError(
            "input['problem_statement'] must be a non-empty string."
        )

    if max_files < 1 or max_total_chars < 1:
        raise ValueError("Context limits must be positive.")

    start = Path(repo_dir or Path.cwd()).resolve()

    root = Path(
        _git(start, "rev-parse", "--show-toplevel").strip()
    ).resolve()

    status = _git(root, "status", "--short").strip() or "(clean)"

    raw_paths = _git(
        root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    )

    candidates = []

    for item in raw_paths.split("\0"):
        if not item:
            continue

        relative = Path(item)

        if _is_excluded(relative):
            continue

        full_path = root / relative

        if not full_path.is_file():
            continue

        try:
            if full_path.stat().st_size > max_file_bytes:
                continue
        except OSError:
            continue

        candidates.append(relative)

    candidates.sort(key=_file_priority)

    sections = [
        "# TASK",
        problem.strip(),
        "",
        "# GIT STATUS",
        status,
        "",
        "# PROJECT FILES",
    ]

    sections.extend(str(path) for path in candidates)

    sections.extend(["", "# SELECTED FILE CONTENTS"])

    context = "\n".join(sections)

    for relative in candidates[:max_files]:
        full_path = root / relative

        try:
            data = full_path.read_bytes()
        except OSError:
            continue

        # Avoid binary files.
        if b"\0" in data:
            continue

        content = data.decode("utf-8", errors="replace")

        section = (
            f"\n\n## FILE: {relative}\n"
            f"```text\n{content}\n```"
        )

        if len(context) + len(section) > max_total_chars:
            remaining = max_total_chars - len(context)

            if remaining > 100:
                context += (
                    section[:remaining]
                    + "\n[CONTEXT TRUNCATED]"
                )

            break

        context += section

    return context
