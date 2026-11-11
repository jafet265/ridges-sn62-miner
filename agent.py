"""Ridges Agent V1: inspect a task, request a patch, apply it, return a diff."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import re
import subprocess
import shutil
import tempfile
from pathlib import Path

import requests


DEFAULT_MODEL = "qwen/qwen3-coder-next"
DEFAULT_MAX_FILES = 30
DEFAULT_MAX_FILE_BYTES = 20_000
DEFAULT_MAX_CONTEXT_CHARS = 40_000

EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__",
    "node_modules", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "dist", "build", ".tox",
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

SENSITIVE_SUFFIXES = {
    ".pem", ".key", ".p12", ".pfx", ".keystore",
}


def _git(
    root: Path,
    *args: str,
    input_text: str | None = None,
) -> str:
    """Run Git and return stdout, raising a useful error on failure."""
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        input=input_text,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"Git command failed: git {' '.join(args)}: {message}"
        )

    return result.stdout


def _repo_root(start: Path | None = None) -> Path:
    """Find the Git root for the current task."""
    location = (start or Path.cwd()).resolve()
    return Path(_git(location, "rev-parse", "--show-toplevel").strip()).resolve()


def _is_excluded(path: Path) -> bool:
    """Avoid common generated, binary, and sensitive files."""
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()

    if parts & EXCLUDED_DIRS:
        return True

    if name.startswith(".env"):
        return True

    if path.suffix.lower() in SENSITIVE_SUFFIXES:
        return True

    sensitive_words = ("secret", "credential", "password", "token")
    if any(word in name for word in sensitive_words):
        return True

    return False



def _task_terms(text: str) -> set[str]:
    """Extract useful words and identifiers from task text."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    words = set(re.findall(r"[a-z0-9]+", text.lower()))

    stop_words = {
        "the", "and", "for", "with", "from", "that", "this",
        "into", "then", "than", "are", "was", "were", "has",
        "have", "had", "its", "your", "you", "our", "not",
        "all", "any", "can", "should", "could", "would",
        "please", "fix", "update", "change", "make", "add",
        "remove", "implement", "modify", "improve", "ensure",
        "use", "using", "when", "where", "what", "which",
        "test", "tests", "spec", "specs",
    }

    return {word for word in words if len(word) >= 3 and word not in stop_words}


def _task_file_priority(
    path: Path,
    problem: str,
) -> tuple[int, int, int, str]:
    """Rank files by task relevance, then by their general usefulness."""
    task_terms = _task_terms(problem)
    path_text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(path))
    path_terms = set(re.findall(r"[a-z0-9]+", path_text.lower()))

    relevance = len(task_terms & path_terms)
    name = path.name.lower()

    if name in USEFUL_NAMES:
        general_priority = 0
    elif "test" in name or "spec" in name:
        general_priority = 2
    elif path.suffix.lower() in SOURCE_EXTENSIONS:
        general_priority = 1
    elif path.suffix.lower() in {".md", ".yaml", ".yml", ".toml", ".json"}:
        general_priority = 3
    else:
        general_priority = 4

    # Task-related files come first. Within each group, preserve
    # the existing preference for source files and useful project files.
    return (0 if relevance else 1, -relevance, general_priority, str(path))



def _analyze_change_scope(
    paths: list[Path],
    problem: str,
) -> dict:
    """Identify potentially relevant files using task and path terms."""
    task_terms = _task_terms(problem)

    related_source = []
    related_tests = []
    related_other = []

    config_files = []

    for path in paths:
        name = path.name.lower()
        path_text = re.sub(r"([a-z])([A-Z])", r"\\1 \\2", str(path))
        path_terms = set(re.findall(r"[a-z0-9]+", path_text.lower()))

        is_test = any(
            "test" in part.lower() or "spec" in part.lower()
            for part in path.parts
        )

        if name in USEFUL_NAMES or path.suffix.lower() in {
            ".yaml", ".yml", ".toml", ".json"
        }:
            config_files.append(str(path))

        if not task_terms.intersection(path_terms):
            continue

        if is_test:
            related_tests.append(str(path))
        elif path.suffix.lower() in SOURCE_EXTENSIONS:
            related_source.append(str(path))
        else:
            related_other.append(str(path))

    return {
        "matched": bool(
            related_source or related_tests or related_other
        ),
        "source_files": sorted(related_source),
        "test_files": sorted(related_tests),
        "other_files": sorted(related_other),
        "config_files": sorted(config_files),
    }


def _format_change_scope(scope: dict) -> list[str]:
    """Format scope findings as concise model context."""
    def describe(label: str, paths: list[str]) -> str:
        if not paths:
            return f"{label}: none identified"

        shown = paths[:8]
        result = f"{label}: " + ", ".join(shown)

        if len(paths) > len(shown):
            result += f" (+{len(paths) - len(shown)} more)"

        return result

    lines = [
        "Potentially relevant source files: "
        + ", ".join(scope["source_files"][:8])
        if scope["source_files"]
        else "Potentially relevant source files: none identified",

        "Potentially relevant tests: "
        + ", ".join(scope["test_files"][:8])
        if scope["test_files"]
        else "Potentially relevant tests: none identified",

        describe(
            "Other task-related files",
            scope["other_files"],
        ),

        describe(
            "Project configuration/reference files",
            scope["config_files"],
        ),
    ]

    if scope["matched"]:
        lines.append(
            "Scope basis: filename/path matches task terminology. "
            "Inspect file contents before deciding what to change."
        )
    else:
        lines.append(
            "Scope basis: no filename/path matches were found. "
            "Use the broader repository context to identify relevant files."
        )

    return lines


def build_task_context(
    task_input: dict,
    repo_dir: str | Path | None = None,
    max_files: int = DEFAULT_MAX_FILES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    """Create bounded, task-aware context from the repository."""
    if not isinstance(task_input, dict):
        raise TypeError("Task input must be a dictionary.")

    problem = task_input.get("problem_statement")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError(
            "input['problem_statement'] must be a non-empty string."
        )

    if max_files < 1 or max_file_bytes < 1 or max_total_chars < 1:
        raise ValueError("Context limits must be positive.")

    root = _repo_root(Path(repo_dir) if repo_dir else None)
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

        # Do not follow symlinks outside the task repository.
        if full_path.is_symlink() or not full_path.is_file():
            continue

        try:
            if full_path.stat().st_size > max_file_bytes:
                continue
        except OSError:
            continue

        candidates.append(relative)

    candidates.sort(key=lambda path: _task_file_priority(path, problem))

    selected = candidates[:max_files]
    omitted_by_file_limit = max(0, len(candidates) - len(selected))

    scope = _analyze_change_scope(candidates, problem)

    sections = [
        "# TASK",
        problem.strip(),
        "",
        "# CHANGE SCOPE ANALYSIS",
    ]
    sections.extend(_format_change_scope(scope))
    sections.extend([
        "",
        "# GIT STATUS",
        status,
        "",
        f"# PROJECT FILES ({len(candidates)} available)",
    ])
    sections.extend(str(path) for path in candidates)
    sections.extend([
        "",
        f"# SELECTED FILE CONTENTS ({len(selected)} selected)",
        f"# Files omitted by max_files limit: {omitted_by_file_limit}",
    ])

    context = "\n".join(sections)

    # Keep even the file listing and metadata within the total limit.
    if len(context) > max_total_chars:
        return context[:max_total_chars]

    for relative in selected:
        full_path = root / relative

        try:
            data = full_path.read_bytes()
        except OSError:
            continue

        if b"\0" in data:
            continue

        content = data.decode("utf-8", errors="replace")
        section = (
            f"\n\n## FILE: {relative}\n"
            f"```text\n{content}\n```"
        )

        remaining = max_total_chars - len(context)
        if len(section) > remaining:
            if remaining > 100:
                marker = "\n[FILE CONTENT TRUNCATED]"
                available = max(0, remaining - len(marker))
                context += section[:available] + marker
            break

        context += section

    return context

def _inference_settings() -> tuple[str, str, str]:
    """Resolve local provider settings or production OpenRouter settings."""
    provider = os.getenv("RIDGES_INFERENCE_PROVIDER", "").strip().lower()

    if provider:
        api_key = (
            os.getenv("RIDGES_INFERENCE_API_KEY", "").strip()
            or os.getenv("OPENROUTER_API_KEY", "").strip()
        )
        base_url = os.getenv("RIDGES_INFERENCE_BASE_URL", "").strip()

        if provider == "openrouter":
            base_url = base_url or "https://openrouter.ai/api/v1"
        elif provider not in {"targon", "chutes"}:
            raise RuntimeError(
                f"Unsupported inference provider: {provider}"
            )
        elif not base_url:
            raise RuntimeError(
                f"{provider} requires RIDGES_INFERENCE_BASE_URL."
            )
    else:
        # Production Ridges runtime provides OPENROUTER_API_KEY.
        provider = "openrouter"
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        base_url = "https://openrouter.ai/api/v1"

    if not api_key:
        raise RuntimeError(
            "No inference API key is configured. For local runs, configure "
            "the provider in ~/.ridges/.env.miner. Production runs receive "
            "OPENROUTER_API_KEY from Ridges."
        )

    return provider, api_key, base_url.rstrip("/")


def call_model(messages: list[dict]) -> str:
    """Call the configured OpenAI-compatible chat completions endpoint."""
    _provider, api_key, base_url = _inference_settings()

    model = os.getenv("RIDGES_MODEL", DEFAULT_MODEL).strip()
    if not model:
        raise RuntimeError("RIDGES_MODEL cannot be empty.")

    try:
        timeout = float(
            os.getenv("RIDGES_INFERENCE_TIMEOUT_SECONDS", "180")
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "RIDGES_INFERENCE_TIMEOUT_SECONDS must be a number."
        ) from exc

    if not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise RuntimeError(
            "RIDGES_INFERENCE_TIMEOUT_SECONDS must be greater than 0 "
            "and no more than 600 seconds."
        )

    max_response_bytes = 2 * 1024 * 1024
    response = None

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0.1,
                "messages": messages,
                "stream": False,
            },
            timeout=(20, timeout),
            stream=True,
        )
        response.raise_for_status()

        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > max_response_bytes:
                    raise RuntimeError(
                        "Inference response exceeds the 2 MB size limit."
                    )
            except ValueError:
                pass

        chunks = []
        total_bytes = 0

        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total_bytes += len(chunk)
            if total_bytes > max_response_bytes:
                raise RuntimeError(
                    "Inference response exceeds the 2 MB size limit."
                )
            chunks.append(chunk)

    except requests.RequestException as exc:
        status = getattr(
            getattr(exc, "response", None), "status_code", None
        )
        status_text = f" HTTP status: {status}." if status else ""
        raise RuntimeError(
            f"Inference request failed ({type(exc).__name__})."
            f"{status_text}"
        ) from exc
    finally:
        if response is not None:
            response.close()

    try:
        data = json.loads(b"".join(chunks).decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
    except (UnicodeDecodeError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Inference provider returned an unexpected response."
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The model returned an empty response.")

    return content


def _extract_unified_diff(model_output: str) -> str:
    """Extract and perform basic safety checks on a model-generated Git diff."""

    if not isinstance(model_output, str):
        raise RuntimeError("The model response must be a string.")

    output = model_output.strip()

    if not output:
        raise RuntimeError("The model returned an empty response.")

    if len(output) > 1_000_000:
        raise RuntimeError(
            "The model response exceeds the 1,000,000-character limit."
        )

    if "\x00" in output:
        raise RuntimeError("The model response contains a null byte.")

    # Find a real Git diff header at the beginning of a line.
    match = re.search(r"(?m)^diff --git .+$", output)

    if match is None:
        raise RuntimeError(
            "The model did not return a standard Git diff header."
        )

    patch = output[match.start():].strip()

    # Remove a trailing Markdown code fence.
    patch = re.sub(r"\n?```[ \t]*$", "", patch).strip()

    if not patch:
        raise RuntimeError("The model returned an empty patch.")

    # A standard Git diff header must identify both file paths.
    header = patch.splitlines()[0]
    parts = header.split(" ", 3)

    if len(parts) != 4 or parts[0] != "diff" or parts[1] != "--git":
        raise RuntimeError("The model returned a malformed Git diff header.")

    return patch + "\n"

def _validate_patch_paths(paths: list[str]) -> None:
    """Reject patch paths that could escape the repository."""
    for raw_path in paths:
        normalized = raw_path.replace("\\", "/")

        unsafe = (
            not normalized
            or normalized.startswith("/")
            or bool(re.match(r"^[A-Za-z]:", normalized))
            or ".." in normalized.split("/")
        )

        if unsafe:
            raise RuntimeError(
                f"Unsafe patch path rejected: {raw_path!r}"
            )


def _patch_changed_paths(root: Path, patch: str) -> list[str]:
    """Ask Git to parse a patch and return its changed paths."""
    result = subprocess.run(
        ["git", "apply", "--numstat", "-z", "-"],
        cwd=root,
        input=patch,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"Git could not inspect patch paths: {message}"
        )

    paths = []

    for record in result.stdout.split("\0"):
        if not record:
            continue

        fields = record.split("\t", 2)
        if len(fields) != 3:
            raise RuntimeError(
                "Git returned an unexpected patch path format."
            )

        paths.append(fields[2])

    if not paths:
        raise RuntimeError("The patch does not contain any changed files.")

    _validate_patch_paths(paths)
    return paths


def _patch_completeness_warnings(
    root: Path,
    patch: str,
    problem: str,
) -> list[str]:
    """Return advisory warnings about potentially incomplete task coverage."""
    changed_paths = set(_patch_changed_paths(root, patch))

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
        if full_path.is_symlink() or not full_path.is_file():
            continue

        candidates.append(relative)

    scope = _analyze_change_scope(candidates, problem)
    warnings = []

    relevant_sources = set(scope["source_files"])
    relevant_tests = set(scope["test_files"])

    if relevant_sources and not relevant_sources.intersection(changed_paths):
        warnings.append(
            "No potentially relevant existing source file appears in the "
            "patch. Verify that the implementation changes the correct files."
        )

    if relevant_tests and not relevant_tests.intersection(changed_paths):
        warnings.append(
            "Potentially relevant existing test files were not changed. "
            "Verify whether new or updated regression tests are needed."
        )

    return warnings



def _evaluate_patch_quality(
    root: Path,
    patch: str,
    problem: str,
    max_files: int = 20,
    max_changed_lines: int = 1000,
) -> list[str]:
    """Return conservative warnings about patch quality and potential risks."""
    changed_paths = _patch_changed_paths(root, patch)
    warnings = []

    if len(changed_paths) > max_files:
        warnings.append(
            f"Patch changes {len(changed_paths)} files, exceeding the "
            f"review threshold of {max_files}. Check for unnecessary changes."
        )

    added_lines = sum(
        1 for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    deleted_lines = sum(
        1 for line in patch.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    changed_lines = added_lines + deleted_lines

    if changed_lines > max_changed_lines:
        warnings.append(
            f"Patch changes {changed_lines} lines, exceeding the review "
            f"threshold of {max_changed_lines}. Check whether it can be "
            "made more focused."
        )

    sensitive_patterns = (
        ".env",
        "secret",
        "credential",
        "private_key",
        "id_rsa",
        ".pem",
        ".p12",
        ".pfx",
    )
    safe_env_files = {
        ".env.example",
        ".env.sample",
        ".env.template",
    }

    for path in changed_paths:
        name = Path(path).name.lower()
        if (
            (name.startswith(".env") and name not in safe_env_files)
            or any(pattern in name for pattern in sensitive_patterns[1:])
        ):
            warnings.append(
                f"Patch changes a potentially sensitive file: {path}. "
                "Verify that no credentials or private data are included."
            )

    dependency_names = {
        "pyproject.toml",
        "poetry.lock",
        "uv.lock",
        "pipfile",
        "pipfile.lock",
        "package.json",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "cargo.toml",
        "cargo.lock",
        "go.mod",
        "go.sum",
    }

    for path in changed_paths:
        name = Path(path).name.lower()
        if (
            name in dependency_names
            or name.startswith("requirements")
            and name.endswith(".txt")
        ):
            warnings.append(
                f"Patch changes a dependency manifest or lockfile: {path}. "
                "Verify that the dependency change is necessary."
            )

    removed_assertions = [
        line for line in patch.splitlines()
        if line.startswith("-")
        and not line.startswith("---")
        and re.search(
            r"\b(assert|pytest\.raises|skip|xfail)\b",
            line,
        )
    ]

    if removed_assertions:
        warnings.append(
            f"Patch removes {len(removed_assertions)} test assertion or "
            "test-control line(s). Verify that existing test protection "
            "has not been weakened."
        )

    # If at least one changed path clearly matches the task, flag other
    # changed paths with no task-term overlap for human review.
    task_terms = set(_task_terms(problem))
    if task_terms and len(changed_paths) > 1:
        matching_paths = []
        unrelated_paths = []

        for path in changed_paths:
            path_terms = set(
                re.findall(r"[a-zA-Z0-9]+", path.lower())
            )
            if path_terms.intersection(task_terms):
                matching_paths.append(path)
            else:
                unrelated_paths.append(path)

        if matching_paths and unrelated_paths:
            warnings.append(
                "Some changed files do not appear related by filename to "
                "the task: "
                + ", ".join(unrelated_paths)
                + ". Review them for unrelated changes."
            )

    warnings.extend(
        _patch_completeness_warnings(root, patch, problem)
    )

    # Keep warning output deterministic and avoid duplicate messages.
    return list(dict.fromkeys(warnings))


def _apply_patch(root: Path, patch: str) -> None:
    """Validate and apply a model-generated patch."""
    _patch_changed_paths(root, patch)
    _git(root, "apply", "--check", "-", input_text=patch)
    _git(root, "apply", "-", input_text=patch)



def _revert_patch(root: Path, patch: str) -> None:
    """Reverse the model patch after validation failure."""
    _git(root, "apply", "--reverse", "--check", "-", input_text=patch)
    _git(root, "apply", "--reverse", "-", input_text=patch)


def _detect_project_test_command(root: Path) -> list[str] | None:
    """Detect a supported Python test suite."""

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


def _ignore_test_copy(directory: str, names: list[str]) -> set[str]:
    """Exclude environments, caches, and private environment files."""
    ignored = set()

    excluded_dirs = EXCLUDED_DIRS - {".git"}
    safe_env_examples = {
        ".env.example",
        ".env.sample",
        ".env.template",
    }

    for name in names:
        item = Path(directory) / name

        if name in excluded_dirs:
            ignored.add(name)
        elif name.endswith(".pyc"):
            ignored.add(name)
        elif name == ".env":
            ignored.add(name)
        elif name.startswith(".env.") and name not in safe_env_examples:
            ignored.add(name)
        elif name == ".git" and item.is_file():
            # Avoid copying a Git worktree pointer to the original repository.
            ignored.add(name)

    return ignored


def _run_project_tests(root: Path, timeout: int = 120) -> dict:
    """Run detected Python tests in an isolated copy of the patched project."""

    command = _detect_project_test_command(root)

    if not command:
        return {
            "status": "skipped",
            "success": False,
            "command": None,
            "exit_code": None,
            "output": "No supported Python test suite was detected.",
        }

    try:
        with tempfile.TemporaryDirectory(
            prefix="ridges-validation-"
        ) as temporary_directory:
            test_root = Path(temporary_directory) / "project"

            shutil.copytree(
                root,
                test_root,
                ignore=_ignore_test_copy,
                symlinks=True,
            )

            result = subprocess.run(
                command,
                cwd=test_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

        output = (result.stdout + "\n" + result.stderr).strip()

        # unittest can exit successfully even when it discovers zero tests.
        if "unittest" in command:
            match = re.search(r"\bRan\s+(\d+)\s+tests?\b", output)

            if match is None or int(match.group(1)) == 0:
                return {
                    "status": "failed",
                    "success": False,
                    "command": command,
                    "exit_code": 1,
                    "output": (
                        output
                        + "\nValidation rejected: unittest executed zero tests."
                    ).strip(),
                }

        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "success": result.returncode == 0,
            "command": command,
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
            "command": command,
            "exit_code": None,
            "output": (
                f"Tests exceeded the {timeout}-second time limit.\n"
                f"{stdout}\n{stderr}"
            ).strip(),
        }

    except (OSError, shutil.Error) as exc:
        return {
            "status": "error",
            "success": False,
            "command": command,
            "exit_code": None,
            "output": f"Could not prepare or run the isolated test process: {exc}",
        }


def generate_unified_diff(
    repo_dir: str | Path | None = None,
) -> str:
    """Return all task changes against the original Git HEAD."""
    root = _repo_root(Path(repo_dir) if repo_dir else None)

    # Ridges establishes a baseline commit before agent execution.
    _git(root, "rev-parse", "--verify", "HEAD")

    # Make new, non-ignored files visible to git diff.
    _git(root, "add", "-N", "--", ".")

    patch = _git(
        root,
        "diff",
        "--binary",
        "--no-ext-diff",
        "HEAD",
        "--",
    )

    if not patch.strip():
        raise RuntimeError(
            "No file changes were detected. Cannot return an empty patch."
        )

    return patch


def agent_main(input: dict) -> str:
    """Ridges entry point: receive a task and return a unified diff."""
    print("[AGENT] Starting Agent V1.")

    root = _repo_root()
    context = build_task_context(input, repo_dir=root)

    system_message = """You are an expert software engineer working inside
an existing Git repository.

Solve the user's programming task by proposing a minimal, correct patch.

Rules:
- Use the supplied task instructions and repository context.
- Inspect relevant existing tests and test configuration before designing
  implementation changes.
- Treat existing tests as evidence of expected behavior. Preserve that
  behavior unless the task explicitly requires a change.
- Add or update tests for new behavior, important edge cases, and regressions
  when appropriate. Follow the repository's existing testing conventions.
- Never delete, weaken, skip, or change test assertions merely to make the
  proposed implementation pass. Change tests only when the task legitimately
  changes the expected behavior.
- If no relevant tests or test framework are available, do not invent an
  unsupported testing setup.
- Preserve existing project conventions and APIs unless the task requires
  changing them.
- Do not invent files or APIs that are not justified by the task.
- Keep the patch minimal, but include every implementation and test change
  needed to solve the task completely.
- Do not claim that tests were executed; the agent will validate the patch.
- Do not include explanations, summaries, or Markdown fences.
- Return ONLY a valid unified Git diff.
- Use standard diff headers, such as:
  diff --git a/path/to/file b/path/to/file
  --- a/path/to/file
  +++ b/path/to/file
- Include every required new, modified, or deleted file.
- Do not include secrets, credentials, or unrelated changes.
"""

    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": (
                "Solve this task in the repository.\n\n"
                "Repository context:\n"
                f"{context}\n\n"
                "Return only the unified Git diff."
            ),
        },
    ]

    print("[AGENT] Sending task and repository context to the model.")
    model_output = call_model(messages)

    patch = _extract_unified_diff(model_output)

    print("[AGENT] Evaluating patch quality.")
    warnings = _evaluate_patch_quality(
        root,
        patch,
        input["problem_statement"],
    )
    if warnings:
        for warning in warnings:
            print(f"[AGENT] WARNING: {warning}")
    else:
        print("[AGENT] No patch quality warnings detected.")

    print("[AGENT] Validating and applying the proposed patch.")
    _apply_patch(root, patch)

    # Protect every operation after patch application. Any unexpected
    # exception must trigger rollback before it escapes the agent.
    try:
        print("[AGENT] Running project tests.")
        test_result = _run_project_tests(root)
        print(f"[AGENT] Test status: {test_result['status']}")

        if test_result["status"] in {"failed", "timeout", "error"}:
            details = test_result["output"][-4000:]
            raise RuntimeError(
                "The proposed patch did not pass project test validation. "
                f"Status: {test_result['status']}\\n{details}"
            )

        if test_result["status"] == "skipped":
            print(
                "[AGENT] WARNING: No supported test suite was detected. "
                "The patch is unverified."
            )
        else:
            print("[AGENT] Project tests passed.")

        print("[AGENT] Generating the final repository diff.")
        final_patch = generate_unified_diff(root)

        print(
            f"[AGENT] Completed. Final patch size: "
            f"{len(final_patch)} characters."
        )
        return final_patch

    except Exception as original_error:
        print("[AGENT] An error occurred after patch application.")
        print("[AGENT] Attempting automatic rollback.")

        try:
            _revert_patch(root, patch)
        except Exception as rollback_error:
            raise RuntimeError(
                "Agent failed after applying the patch, and automatic "
                "rollback also failed. The repository may contain partial "
                f"changes. Original error: {original_error}. "
                f"Rollback error: {rollback_error}"
            ) from rollback_error

        raise RuntimeError(
            "Agent failed after applying the patch. "
            "Automatic rollback completed successfully. "
            f"Original error: {original_error}"
        ) from original_error
