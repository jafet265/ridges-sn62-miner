import json
import os

import requests

RUN_ID = os.getenv("RUN_ID")
if not RUN_ID:
    print("[AGENT] WARNING: RUN_ID is not set")

SANDBOX_PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
if not SANDBOX_PROXY_URL:
    print("[AGENT] WARNING: SANDBOX_PROXY_URL is not set")


def inference(model, temperature, messages):
    try:
        payload = {"run_id": RUN_ID, "model": model, "temperature": temperature, "messages": messages}

        print(
            f"[AGENT] inference(): Sending inference request for model {model} (temperature {temperature}) with {len(messages)} messages"
        )
        response = requests.post(
            f"{SANDBOX_PROXY_URL}/api/inference", headers={"Content-Type": "application/json"}, data=json.dumps(payload)
        )

        if response.status_code == 200:
            result = response.text.strip('"')
            print(f"[AGENT] inference(): Inference response: {len(result)} characters")
            return result
        else:
            print(f"[AGENT] inference(): Inference failed with status {response.status_code}: {response.text}")
            return None

    except Exception as e:
        print(f"[AGENT] inference(): Inference request failed: {e}")
        return None


def embedding(input):
    try:
        payload = {"run_id": RUN_ID, "input": input}

        print("[AGENT] embedding(): Sending embedding request...")
        response = requests.post(
            f"{SANDBOX_PROXY_URL}/api/embedding", headers={"Content-Type": "application/json"}, data=json.dumps(payload)
        )

        if response.status_code == 200:
            result = response.json()
            print(f"[AGENT] embedding(): Embedding response: {len(result)} dimensions")
            return result
        else:
            print(f"[AGENT] embedding(): Embedding failed with status {response.status_code}: {response.text}")
            return None

    except Exception as e:
        print(f"[AGENT] embedding(): Embedding request failed: {e}")
        return None


def agent_main(input):
    print("[AGENT] Entered agent_main()")

    # Test inference function
    message = "What is 2+2?"
    print(f"[AGENT] <-- '{message}'")
    messages = [{"role": "user", "content": message}]
    inference_result = inference("moonshotai/Kimi-K2-Instruct", 0.5, messages)
    if inference_result:
        print(f"[AGENT] --> '{inference_result}'")

    print("[AGENT] Reading solution from /sandbox/solution.diff")
    with open("/sandbox/solution.diff", "r") as f:
        diff = f.read()

    print("[AGENT] Exiting agent_main()")

    return diff


def generate_unified_diff(repo_dir=None):
    """
    Generate a unified Git patch containing all task changes.

    Includes modified, deleted, staged, unstaged, and new non-ignored files.
    Returns a non-empty patch string or raises RuntimeError.
    """
    from pathlib import Path
    import subprocess

    root = Path(repo_dir or Path.cwd()).resolve()

    def git(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Git command failed: git {' '.join(args)}\n"
                f"{result.stderr.strip()}"
            )
        return result.stdout

    # Resolve the actual repository root.
    root = Path(git("rev-parse", "--show-toplevel").strip()).resolve()

    # Ridges establishes a baseline commit before agent execution.
    git("rev-parse", "--verify", "HEAD")

    # Register new files as intent-to-add so they appear in git diff.
    git("add", "-N", "--", ".")

    # Compare the entire working tree against the original baseline.
    patch = git("diff", "--binary", "--no-ext-diff", "HEAD", "--")

    if not patch.strip():
        raise RuntimeError(
            "No file changes were detected. Cannot return an empty patch."
        )

    return patch
