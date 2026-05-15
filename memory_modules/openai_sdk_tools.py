from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_tool_sdk() -> dict[str, Any]:
    try:
        from agents import (
            ApplyPatchTool,
            ShellCallOutcome,
            ShellCommandOutput,
            ShellResult,
            ShellTool,
        )
    except ImportError as exc:
        raise RuntimeError(
            "openai sdk tools require the openai-agents package. "
            "Install dependencies from requirements.txt or pyproject.toml."
        ) from exc
    return {
        "ApplyPatchTool": ApplyPatchTool,
        "ShellCallOutcome": ShellCallOutcome,
        "ShellCommandOutput": ShellCommandOutput,
        "ShellResult": ShellResult,
        "ShellTool": ShellTool,
    }


def make_sandbox_tools(
    *,
    sandbox_dir: Path,
    tool_calls: list[dict[str, Any]],
    tool_timeout_seconds: float,
    max_tool_output_chars: int,
) -> list[Any]:
    sdk = load_tool_sdk()
    return [
        _make_shell_tool(
            sdk=sdk,
            sandbox_dir=sandbox_dir,
            tool_calls=tool_calls,
            tool_timeout_seconds=tool_timeout_seconds,
            max_tool_output_chars=max_tool_output_chars,
        ),
        sdk["ApplyPatchTool"](
            editor=SandboxApplyPatchEditor(sandbox_dir=sandbox_dir, tool_calls=tool_calls),
            needs_approval=False,
        ),
    ]


def _make_shell_tool(
    *,
    sdk: dict[str, Any],
    sandbox_dir: Path,
    tool_calls: list[dict[str, Any]],
    tool_timeout_seconds: float,
    max_tool_output_chars: int,
) -> Any:
    def executor(request: Any) -> Any:
        outputs = [
            _run_shell_command(
                sdk=sdk,
                command=command,
                sandbox_dir=sandbox_dir,
                timeout_ms=request.data.action.timeout_ms,
                tool_calls=tool_calls,
                tool_timeout_seconds=tool_timeout_seconds,
                max_tool_output_chars=max_tool_output_chars,
            )
            for command in request.data.action.commands
        ]
        return sdk["ShellResult"](output=outputs, max_output_length=max_tool_output_chars)

    return sdk["ShellTool"](
        executor=executor,
        environment={"type": "local"},
        needs_approval=False,
    )


def _run_shell_command(
    *,
    sdk: dict[str, Any],
    command: str,
    sandbox_dir: Path,
    timeout_ms: int | None,
    tool_calls: list[dict[str, Any]],
    tool_timeout_seconds: float,
    max_tool_output_chars: int,
) -> Any:
    started = time.time()
    timeout_seconds = tool_timeout_seconds
    if timeout_ms is not None:
        timeout_seconds = min(timeout_seconds, max(timeout_ms / 1000.0, 0.001))

    try:
        env = _shell_env(sandbox_dir)
        process = subprocess.run(
            ["/bin/bash", "-lc", _prefix_path_export(command, sandbox_dir)],
            cwd=sandbox_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _subprocess_output_to_text(exc.stdout)
        stderr = _subprocess_output_to_text(exc.stderr)
        tool_response = {
            "returncode": None,
            "stdout": stdout,
            "stderr": stderr,
            "timeout_seconds": timeout_seconds,
        }
        tool_calls.append(
            {
                "tool": "shell",
                "command": command,
                "returncode": None,
                "duration_seconds": time.time() - started,
                "timed_out": True,
                "tool_response": tool_response,
            }
        )
        return sdk["ShellCommandOutput"](
            stdout=stdout,
            stderr=stderr or f"shell command timed out after {timeout_seconds}s",
            outcome=sdk["ShellCallOutcome"](type="timeout", exit_code=None),
            command=command,
            provider_data=tool_response,
        )

    stdout, stdout_truncated = _truncate_text(process.stdout, max_tool_output_chars)
    stderr, stderr_truncated = _truncate_text(process.stderr, max_tool_output_chars)
    tool_response = _truncated_output_response(
        stdout_text=process.stdout,
        stderr_text=process.stderr,
        returncode=process.returncode,
        max_chars=max_tool_output_chars,
    )
    if tool_response is not None:
        stdout = tool_response["stdout"]
        stderr_metadata = {
            key: value
            for key, value in tool_response.items()
            if key not in {"stdout", "stderr"}
        }
        stderr = tool_response["stderr"]
        metadata_text = json.dumps(stderr_metadata, ensure_ascii=True)
        stderr = f"{stderr}\n{metadata_text}" if stderr else metadata_text
        exit_code = 2
    else:
        tool_response = {
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_chars": len(process.stdout),
            "stderr_chars": len(process.stderr),
            "max_output_chars": max_tool_output_chars,
            "stdout_truncated": False,
            "stderr_truncated": False,
        }
        exit_code = process.returncode

    tool_calls.append(
        {
            "tool": "shell",
            "command": command,
            "returncode": process.returncode,
            "duration_seconds": time.time() - started,
            "timed_out": False,
            "stdout_chars": len(process.stdout),
            "stderr_chars": len(process.stderr),
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
            "tool_response": tool_response,
        }
    )
    return sdk["ShellCommandOutput"](
        stdout=stdout,
        stderr=stderr,
        outcome=sdk["ShellCallOutcome"](type="exit", exit_code=exit_code),
        command=command,
        provider_data=tool_response,
    )


class SandboxApplyPatchEditor:
    def __init__(self, *, sandbox_dir: Path, tool_calls: list[dict[str, Any]]) -> None:
        self.sandbox_dir = sandbox_dir.resolve()
        self.tool_calls = tool_calls

    def create_file(self, operation: Any) -> str:
        started = time.time()
        try:
            require(operation.diff is not None, f"missing diff for create_file: {operation.path}")
            path = self._resolve_path(operation.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_apply_diff("", operation.diff, mode="create"), encoding="utf-8")
            return self._record(operation, started, "completed", f"Created {operation.path}")
        except Exception as exc:
            self._record(operation, started, "failed", str(exc))
            raise

    def update_file(self, operation: Any) -> str:
        started = time.time()
        try:
            require(operation.diff is not None, f"missing diff for update_file: {operation.path}")
            path = self._resolve_path(operation.path)
            updated = _apply_diff(path.read_text(encoding="utf-8"), operation.diff, mode="default")
            destination = self._resolve_path(operation.move_to) if operation.move_to else path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(updated, encoding="utf-8")
            if destination != path:
                path.unlink()
            display = f"{operation.path} -> {operation.move_to}" if operation.move_to else operation.path
            return self._record(operation, started, "completed", f"Updated {display}")
        except Exception as exc:
            self._record(operation, started, "failed", str(exc))
            raise

    def delete_file(self, operation: Any) -> str:
        started = time.time()
        try:
            self._resolve_path(operation.path).unlink()
            return self._record(operation, started, "completed", f"Deleted {operation.path}")
        except Exception as exc:
            self._record(operation, started, "failed", str(exc))
            raise

    def _resolve_path(self, path: str) -> Path:
        require(isinstance(path, str) and path.strip(), "apply_patch path must be non-empty")
        raw_path = Path(path)
        candidate = raw_path if raw_path.is_absolute() else self.sandbox_dir / raw_path
        resolved = candidate.resolve()
        require(
            os.path.commonpath([str(self.sandbox_dir), str(resolved)]) == str(self.sandbox_dir),
            f"apply_patch path escapes sandbox: {path}",
        )
        return resolved

    def _record(self, operation: Any, started: float, status: str, output: str) -> str:
        self.tool_calls.append(
            {
                "tool": "apply_patch",
                "operation": operation.type,
                "path": operation.path,
                "move_to": operation.move_to,
                "duration_seconds": time.time() - started,
                "tool_response": {"status": status, "output": output},
            }
        )
        return output


def _apply_diff(original: str, diff: str, *, mode: str) -> str:
    from agents.apply_diff import apply_diff

    return apply_diff(original, diff, mode=mode)


def _shell_env(sandbox_dir: Path) -> dict[str, str]:
    shim_dir, python_bin_dir = _ensure_python_shims(sandbox_dir)
    env = os.environ.copy()
    env["PATH"] = str(shim_dir) + os.pathsep + str(python_bin_dir) + os.pathsep + env.get("PATH", "")
    return env


def _prefix_path_export(command: str, sandbox_dir: Path) -> str:
    shim_dir, python_bin_dir = _ensure_python_shims(sandbox_dir)
    path_prefix = str(shim_dir) + os.pathsep + str(python_bin_dir)
    return "export PATH=" + shlex.quote(path_prefix) + ':$PATH\n' + command


def _ensure_python_shims(sandbox_dir: Path) -> tuple[Path, Path]:
    shim_dir = sandbox_dir / ".openai_sdk_runner_bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    target = Path(sys.executable).resolve()
    for name in ("python", "python3"):
        shim_path = shim_dir / name
        if not shim_path.exists():
            shim_path.symlink_to(target)
    return shim_dir.resolve(), target.parent


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + f"\n[truncated to {max_chars} chars]", True


def _subprocess_output_to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _truncated_output_response(
    *,
    stdout_text: str,
    stderr_text: str,
    returncode: int | None,
    max_chars: int,
) -> dict[str, Any] | None:
    stdout_truncated = len(stdout_text) > max_chars
    stderr_truncated = len(stderr_text) > max_chars
    if not stdout_truncated and not stderr_truncated:
        return None

    stream_details: list[str] = []
    if stdout_truncated:
        stream_details.append(f"stdout was {len(stdout_text)} chars")
    if stderr_truncated:
        stream_details.append(f"stderr was {len(stderr_text)} chars")
    stdout, _ = _truncate_text(stdout_text, max_chars)
    stderr, _ = _truncate_text(stderr_text, max_chars)
    return {
        "returncode": 2,
        "error": "OUTPUT_TRUNCATED",
        "message": (
            f"Output was too large ({'; '.join(stream_details)}), cap is {max_chars}. "
            "Rerun with a narrower command: sed range, grep pattern, "
            "inspect_trajectory.py --state/--span/--match."
        ),
        "stdout": stdout,
        "stderr": stderr,
        "original_returncode": returncode,
        "stdout_chars": len(stdout_text),
        "stderr_chars": len(stderr_text),
        "max_output_chars": max_chars,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }
