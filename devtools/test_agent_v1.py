import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent


def git(root, *args):
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


class AgentV1Tests(unittest.TestCase):

    def create_repository(self, root):
        (root / "src").mkdir()

        (root / "src" / "calculator.py").write_text(
            "def add(a, b):\n"
            "    return a + b\n",
            encoding="utf-8",
        )

        git(root, "init", "-q")
        git(root, "config", "user.name", "Agent V1 Test")
        git(root, "config", "user.email", "agent-test@example.invalid")
        git(root, "config", "maintenance.auto", "false")
        git(root, "config", "gc.auto", "0")
        git(root, "add", "-A")
        git(root, "commit", "-qm", "baseline")

    def test_change_scope_identifies_source_tests_and_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src").mkdir()
            (root / "tests").mkdir()

            (root / "README.md").write_text(
                "# Billing project\\n", encoding="utf-8"
            )
            (root / "src" / "billing.py").write_text(
                "def calculate_total(items):\\n"
                "    return sum(items)\\n",
                encoding="utf-8",
            )
            (root / "tests" / "test_billing.py").write_text(
                "def test_calculate_total():\\n"
                "    assert True\\n",
                encoding="utf-8",
            )

            git(root, "init", "-q")
            git(root, "config", "user.name", "Scope Test")
            git(root, "config", "user.email", "scope-test@example.invalid")
            git(root, "add", "-A")
            git(root, "commit", "-qm", "baseline")

            context = agent.build_task_context(
                {
                    "problem_statement":
                        "Fix calculate_total in billing.py and update tests."
                },
                repo_dir=root,
            )

            self.assertIn("# CHANGE SCOPE ANALYSIS", context)
            self.assertIn(
                "Potentially relevant source files: src/billing.py",
                context,
            )
            self.assertIn(
                "Potentially relevant tests: tests/test_billing.py",
                context,
            )
            self.assertIn(
                "Project configuration/reference files: README.md",
                context,
            )
            self.assertIn(
                "Inspect file contents before deciding what to change.",
                context,
            )

    def test_change_scope_handles_tasks_without_path_matches(self):
        scope = agent._analyze_change_scope(
            [Path("src/calculator.py")],
            "Improve payment processing.",
        )

        self.assertFalse(scope["matched"])
        self.assertEqual(scope["source_files"], [])
        self.assertEqual(scope["test_files"], [])

    def test_task_context_prioritizes_relevant_source_and_tests(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src").mkdir()
            (root / "tests").mkdir()

            (root / "README.md").write_text(
                "# Example project\\n", encoding="utf-8"
            )
            (root / "src" / "billing.py").write_text(
                "def calculate_total(items):\\n"
                "    return sum(items)\\n",
                encoding="utf-8",
            )
            (root / "tests" / "test_billing.py").write_text(
                "def test_calculate_total():\\n"
                "    assert True\\n",
                encoding="utf-8",
            )
            (root / "src" / "unrelated.py").write_text(
                "def unrelated():\\n"
                "    return None\\n",
                encoding="utf-8",
            )

            git(root, "init", "-q")
            git(root, "config", "user.name", "Agent Context Test")
            git(root, "config", "user.email", "context-test@example.invalid")
            git(root, "add", "-A")
            git(root, "commit", "-qm", "baseline")

            context = agent.build_task_context(
                {
                    "problem_statement":
                        "Fix calculate_total in billing.py and update its tests."
                },
                repo_dir=root,
                max_files=2,
            )

            source_position = context.index("## FILE: src/billing.py")
            test_position = context.index("## FILE: tests/test_billing.py")

            self.assertLess(source_position, test_position)
            self.assertNotIn("## FILE: src/unrelated.py", context)
            self.assertIn("Files omitted by max_files limit: 2", context)


    def test_agent_prompt_includes_test_aware_instructions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)

            model_patch = (
                "diff --git a/src/calculator.py b/src/calculator.py\n"
                "--- a/src/calculator.py\n"
                "+++ b/src/calculator.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def add(a, b):\n"
                "-    return a + b\n"
                "+    return a + b + 1\n"
            )

            captured = {}
            original_cwd = Path.cwd()

            def capture_messages(messages):
                captured["messages"] = messages
                return model_patch

            skipped_tests = {
                "status": "skipped",
                "success": False,
                "command": None,
                "exit_code": None,
                "output": "No supported test suite detected.",
            }

            try:
                import os
                os.chdir(root)

                with patch.object(
                    agent, "call_model", side_effect=capture_messages
                ), patch.object(
                    agent, "_run_project_tests", return_value=skipped_tests
                ):
                    agent.agent_main({
                        "problem_statement": "Change add() to add one extra."
                    })
            finally:
                os.chdir(original_cwd)

            self.assertIn("messages", captured)
            system_prompt = captured["messages"][0]["content"]

            required_instructions = [
                "Inspect relevant existing tests and test configuration",
                "Treat existing tests as evidence of expected behavior",
                "Add or update tests for new behavior",
                "Never delete, weaken, skip, or change test assertions",
                "Follow the repository's existing testing conventions",
                "Do not claim that tests were executed",
            ]

            for instruction in required_instructions:
                with self.subTest(instruction=instruction):
                    self.assertIn(instruction, system_prompt)


    def test_agent_applies_model_patch_and_returns_final_diff(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)

            verifier = root.parent / (root.name + "-verifier")
            shutil.copytree(root, verifier)

            model_patch = (
                "diff --git a/src/calculator.py b/src/calculator.py\n"
                "--- a/src/calculator.py\n"
                "+++ b/src/calculator.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def add(a, b):\n"
                "-    return a + b\n"
                "+    return a + b + 1\n"
            )

            original_call_model = agent.call_model
            original_cwd = Path.cwd()

            try:
                agent.call_model = lambda messages: model_patch
                import os
                os.chdir(root)

                result = agent.agent_main({
                    "problem_statement": "Change add() to add one extra."
                })
            finally:
                os.chdir(original_cwd)
                agent.call_model = original_call_model
                shutil.rmtree(verifier, ignore_errors=True)

            self.assertIn("src/calculator.py", result)
            self.assertIn("+    return a + b + 1", result)
            self.assertIn(
                "return a + b + 1",
                (root / "src" / "calculator.py").read_text(encoding="utf-8"),
            )

    def run_agent_with_test_result(self, root, test_result):
        """Run Agent V1 with a mocked model and mocked test result."""
        model_patch = (
            "diff --git a/src/calculator.py b/src/calculator.py\n"
            "--- a/src/calculator.py\n"
            "+++ b/src/calculator.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n"
            "-    return a + b\n"
            "+    return a + b + 1\n"
        )

        original_cwd = Path.cwd()

        try:
            import os
            os.chdir(root)

            with patch.object(agent, "call_model", return_value=model_patch), patch.object(agent, "_run_project_tests", return_value=test_result):
                return agent.agent_main({
                    "problem_statement": "Change add() to add one extra."
                })
        finally:
            os.chdir(original_cwd)

    def test_passing_tests_allow_final_patch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)

            result = self.run_agent_with_test_result(
                root,
                {
                    "status": "passed",
                    "success": True,
                    "command": ["python", "-m", "pytest"],
                    "exit_code": 0,
                    "output": "2 passed",
                },
            )

            self.assertIn("+    return a + b + 1", result)

    def test_failing_tests_reject_and_revert_patch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)

            with self.assertRaisesRegex(RuntimeError, "did not pass"):
                self.run_agent_with_test_result(
                    root,
                    {
                        "status": "failed",
                        "success": False,
                        "command": ["python", "-m", "pytest"],
                        "exit_code": 1,
                        "output": "1 failed",
                    },
                )

            content = (root / "src" / "calculator.py").read_text(
                encoding="utf-8"
            )
            self.assertIn("return a + b", content)
            self.assertNotIn("return a + b + 1", content)

    def test_timeout_rejects_and_reverts_patch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)

            with self.assertRaisesRegex(RuntimeError, "Status: timeout"):
                self.run_agent_with_test_result(
                    root,
                    {
                        "status": "timeout",
                        "success": False,
                        "command": ["python", "-m", "pytest"],
                        "exit_code": None,
                        "output": "Tests exceeded the time limit.",
                    },
                )

            content = (root / "src" / "calculator.py").read_text(
                encoding="utf-8"
            )
            self.assertIn("return a + b", content)
            self.assertNotIn("return a + b + 1", content)

    def test_skipped_tests_return_explicitly_unverified_patch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)

            result = self.run_agent_with_test_result(
                root,
                {
                    "status": "skipped",
                    "success": False,
                    "command": None,
                    "exit_code": None,
                    "output": "No supported Python test suite was detected.",
                },
            )

            self.assertIn("+    return a + b + 1", result)


    def test_unittest_zero_tests_are_rejected(self):
        command = [sys.executable, "-m", "unittest", "discover", "-v"]
        completed = subprocess.CompletedProcess(
            command,
            0,
            "Ran 0 tests in 0.000s\n\nOK\n",
            "",
        )

        with patch.object(
            agent,
            "_detect_project_test_command",
            return_value=command,
        ), patch.object(
            agent.subprocess,
            "run",
            return_value=completed,
        ):
            result = agent._run_project_tests(Path.cwd())

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["success"])
        self.assertIn("zero tests", result["output"].lower())

    def test_unittest_with_real_test_count_can_pass(self):
        command = [sys.executable, "-m", "unittest", "discover", "-v"]
        completed = subprocess.CompletedProcess(
            command,
            0,
            "Ran 2 tests in 0.001s\n\nOK\n",
            "",
        )

        with patch.object(
            agent,
            "_detect_project_test_command",
            return_value=command,
        ), patch.object(
            agent.subprocess,
            "run",
            return_value=completed,
        ):
            result = agent._run_project_tests(Path.cwd())

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["success"])

    def test_project_tests_cannot_modify_original_repository(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)

            (root / "existing.txt").write_text(
                "original content",
                encoding="utf-8",
            )

            command = [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "Path('existing.txt').write_text('modified by tests'); "
                    "Path('generated.txt').write_text('test artifact')"
                ),
            ]

            with patch.object(
                agent,
                "_detect_project_test_command",
                return_value=command,
            ):
                result = agent._run_project_tests(root)

            self.assertEqual(result["status"], "passed")
            self.assertEqual(
                (root / "existing.txt").read_text(encoding="utf-8"),
                "original content",
            )
            self.assertFalse((root / "generated.txt").exists())

    def test_unexpected_test_exception_triggers_rollback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)
            original_cwd = Path.cwd()

            model_patch = (
                "diff --git a/src/calculator.py b/src/calculator.py\n"
                "--- a/src/calculator.py\n"
                "+++ b/src/calculator.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def add(a, b):\n"
                "-    return a + b\n"
                "+    return a + b + 1\n"
            )

            try:
                import os
                os.chdir(root)

                with patch.object(
                    agent, "call_model", return_value=model_patch
                ), patch.object(
                    agent,
                    "_run_project_tests",
                    side_effect=RuntimeError("simulated test-runner crash"),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "rollback completed successfully"
                    ):
                        agent.agent_main({
                            "problem_statement": "Change add() to add one extra."
                        })
            finally:
                os.chdir(original_cwd)

            content = (root / "src/calculator.py").read_text(encoding="utf-8")
            self.assertIn("return a + b", content)
            self.assertNotIn("return a + b + 1", content)

    def test_rollback_failure_is_reported_clearly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)
            original_cwd = Path.cwd()

            model_patch = (
                "diff --git a/src/calculator.py b/src/calculator.py\n"
                "--- a/src/calculator.py\n"
                "+++ b/src/calculator.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def add(a, b):\n"
                "-    return a + b\n"
                "+    return a + b + 1\n"
            )

            try:
                import os
                os.chdir(root)

                with patch.object(
                    agent, "call_model", return_value=model_patch
                ), patch.object(
                    agent,
                    "_run_project_tests",
                    side_effect=RuntimeError("simulated test-runner crash"),
                ), patch.object(
                    agent,
                    "_revert_patch",
                    side_effect=RuntimeError("simulated rollback failure"),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "rollback also failed"
                    ) as caught:
                        agent.agent_main({
                            "problem_statement": "Change add() to add one extra."
                        })
            finally:
                os.chdir(original_cwd)

            self.assertIn("partial changes", str(caught.exception))
            content = (root / "src/calculator.py").read_text(encoding="utf-8")
            self.assertIn("return a + b + 1", content)

    def test_agent_rejects_missing_problem_statement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)

            original_cwd = Path.cwd()
            try:
                import os
                os.chdir(root)

                with self.assertRaisesRegex(ValueError, "problem_statement"):
                    agent.agent_main({})
            finally:
                os.chdir(original_cwd)

    def test_extract_rejects_non_string_response(self):
        with self.assertRaisesRegex(RuntimeError, "must be a string"):
            agent._extract_unified_diff(None)

    def test_extract_rejects_response_without_git_header(self):
        with self.assertRaisesRegex(RuntimeError, "standard Git diff header"):
            agent._extract_unified_diff(
                "--- a/example.py\\n+++ b/example.py\\n"
            )

    def test_extract_rejects_oversized_response(self):
        with self.assertRaisesRegex(RuntimeError, "1,000,000-character"):
            agent._extract_unified_diff("x" * 1_000_001)

    def test_quality_evaluator_warns_about_sensitive_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)

            patch = (
                "diff --git a/.env b/.env\n"
                "new file mode 100644\n"
                "--- /dev/null\n"
                "+++ b/.env\n"
                "@@ -0,0 +1 @@\n"
                "+API_KEY=example\n"
            )

            warnings = agent._evaluate_patch_quality(
                root, patch, "Add calculator functionality."
            )

            self.assertTrue(
                any("sensitive file" in warning for warning in warnings)
            )

    def test_quality_evaluator_warns_about_dependency_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)

            patch = (
                "diff --git a/pyproject.toml b/pyproject.toml\n"
                "--- a/pyproject.toml\n"
                "+++ b/pyproject.toml\n"
                "@@ -1 +1,2 @@\n"
                " [project]\n"
                "+dependencies = ['example-package']\n"
            )

            warnings = agent._evaluate_patch_quality(
                root, patch, "Fix calculator addition."
            )

            self.assertTrue(
                any("dependency manifest" in warning for warning in warnings)
            )

    def test_quality_evaluator_warns_about_removed_assertions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)

            patch = (
                "diff --git a/tests/test_calculator.py "
                "b/tests/test_calculator.py\n"
                "--- a/tests/test_calculator.py\n"
                "+++ b/tests/test_calculator.py\n"
                "@@ -1,2 +1 @@\n"
                "-def test_add():\n"
                "-    assert 1 + 1 == 2\n"
                "+def test_add():\n"
            )

            warnings = agent._evaluate_patch_quality(
                root, patch, "Update calculator tests."
            )

            self.assertTrue(
                any("removes" in warning for warning in warnings)
            )

    def test_quality_evaluator_warns_about_large_patches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)

            patch = (
                "diff --git a/src/calculator.py b/src/calculator.py\n"
                "--- a/src/calculator.py\n"
                "+++ b/src/calculator.py\n"
                "@@ -1,2 +1,4 @@\n"
                " def add(a, b):\n"
                "     return a + b\n"
                "+# Additional change\n"
                "+# Another change\n"
            )

            warnings = agent._evaluate_patch_quality(
                root,
                patch,
                "Fix calculator addition.",
                max_changed_lines=1,
            )

            self.assertTrue(
                any("exceeding the review threshold" in warning
                    for warning in warnings)
            )

    def test_patch_path_validation_accepts_normal_paths(self):
        agent._validate_patch_paths([
            "src/calculator.py",
            "src/my calculator.py",
            "tests/test_calculator.py",
        ])

    def test_patch_path_validation_rejects_unsafe_paths(self):
        unsafe_paths = [
            "../outside.py",
            "src/../../outside.py",
            "/tmp/outside.py",
            "C:/outside.py",
        ]

        for path in unsafe_paths:
            with self.subTest(path=path):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Unsafe patch path",
                ):
                    agent._validate_patch_paths([path])

    def test_patch_changed_paths_uses_git_parser(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)

            patch = (
                "diff --git a/src/calculator.py b/src/calculator.py\n"
                "--- a/src/calculator.py\n"
                "+++ b/src/calculator.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def add(a, b):\n"
                "-    return a + b\n"
                "+    return a + b + 1\n"
            )

            paths = agent._patch_changed_paths(root, patch)
            self.assertEqual(paths, ["src/calculator.py"])

    def test_patch_completeness_warns_when_relevant_source_is_omitted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)

            (root / "README.md").write_text(
                "# Calculator project\\n",
                encoding="utf-8",
            )
            agent._git(root, "add", "README.md")
            agent._git(root, "commit", "-qm", "Add README")

            patch = (
                "diff --git a/README.md b/README.md\n"
                "--- a/README.md\n"
                "+++ b/README.md\n"
                "@@ -1 +1 @@\n"
                "-# Calculator project\n"
                "+# Updated calculator project\n"
            )

            warnings = agent._patch_completeness_warnings(
                root,
                patch,
                "Fix the calculator implementation.",
            )

            self.assertTrue(
                any("source file" in warning for warning in warnings)
            )

    def test_invalid_patch_does_not_modify_repository(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.create_repository(root)

            original = (root / "src/calculator.py").read_text(
                encoding="utf-8"
            )

            invalid_patch = (
                "diff --git a/src/calculator.py b/src/calculator.py\\n"
                "--- a/src/calculator.py\\n"
                "+++ b/src/calculator.py\\n"
                "@@ -1,2 +1,2 @@\\n"
                " def add(a, b):\\n"
                "-    return a + b + WRONG\\n"
                "+    return a + b + 1\\n"
            )

            with self.assertRaises(RuntimeError):
                agent._apply_patch(root, invalid_patch)

            current = (root / "src/calculator.py").read_text(
                encoding="utf-8"
            )
            self.assertEqual(current, original)

    def test_extracts_diff_from_markdown_fence(self):
        output = (
            "Here is the patch:\n"
            "```diff\n"
            "diff --git a/example.py b/example.py\n"
            "--- a/example.py\n"
            "+++ b/example.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "```\n"
        )

        patch = agent._extract_unified_diff(output)

        self.assertTrue(patch.startswith("diff --git "))
        self.assertIn("+new", patch)
        self.assertNotIn("```", patch)


    def test_call_model_accepts_valid_response(self):
        payload = (
            b'{"choices":[{"message":{"content":"'
            b'diff --git a/a.py b/a.py"}}]}'
        )

        with patch.object(
            agent, "_inference_settings",
            return_value=("openrouter", "test-key", "https://example.invalid/v1"),
        ), patch.object(agent.requests, "post") as mock_post:
            response = mock_post.return_value
            response.headers = {"Content-Length": str(len(payload))}
            response.iter_content.return_value = [payload]

            result = agent.call_model([{"role": "user", "content": "test"}])

        self.assertEqual(result, "diff --git a/a.py b/a.py")
        self.assertTrue(response.close.called)
        self.assertTrue(mock_post.called)
        self.assertTrue(mock_post.call_args.kwargs["stream"])

    def test_call_model_rejects_invalid_timeouts_before_request(self):
        invalid_values = ["abc", "0", "-1", "601", "nan", "inf"]

        with patch.object(
            agent, "_inference_settings",
            return_value=("openrouter", "test-key", "https://example.invalid/v1"),
        ), patch.object(agent.requests, "post") as mock_post:
            for value in invalid_values:
                with self.subTest(timeout=value):
                    with patch.dict(
                        "os.environ",
                        {"RIDGES_INFERENCE_TIMEOUT_SECONDS": value},
                    ):
                        with self.assertRaises(RuntimeError):
                            agent.call_model([])

            mock_post.assert_not_called()

    def test_call_model_rejects_oversized_response(self):
        with patch.object(
            agent, "_inference_settings",
            return_value=("openrouter", "test-key", "https://example.invalid/v1"),
        ), patch.object(agent.requests, "post") as mock_post:
            response = mock_post.return_value
            response.headers = {}
            response.iter_content.return_value = [b"x" * (2 * 1024 * 1024 + 1)]

            with self.assertRaisesRegex(RuntimeError, "2 MB"):
                agent.call_model([])

            response.close.assert_called_once()

    def test_call_model_rejects_malformed_json(self):
        with patch.object(
            agent, "_inference_settings",
            return_value=("openrouter", "test-key", "https://example.invalid/v1"),
        ), patch.object(agent.requests, "post") as mock_post:
            response = mock_post.return_value
            response.headers = {}
            response.iter_content.return_value = [b"not valid JSON"]

            with self.assertRaisesRegex(
                RuntimeError, "unexpected response"
            ):
                agent.call_model([])

            response.close.assert_called_once()

    def test_call_model_hides_provider_error_body(self):
        with patch.object(
            agent, "_inference_settings",
            return_value=("openrouter", "test-key", "https://example.invalid/v1"),
        ), patch.object(agent.requests, "post") as mock_post:
            response = mock_post.return_value
            response.status_code = 401
            response.raise_for_status.side_effect = (
                agent.requests.HTTPError(
                    "SENSITIVE_PROVIDER_RESPONSE",
                    response=response,
                )
            )

            with self.assertRaises(RuntimeError) as caught:
                agent.call_model([])

            self.assertIn("401", str(caught.exception))
            self.assertNotIn(
                "SENSITIVE_PROVIDER_RESPONSE", str(caught.exception)
            )
            response.close.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
