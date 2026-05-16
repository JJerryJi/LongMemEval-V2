from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from agents import (
    ApplyPatchTool,
    ShellCallOutcome,
    ShellCommandOutput,
    ShellCommandRequest,
    ShellResult,
    ShellTool,
    apply_diff,
)
from agents.editor import ApplyPatchOperation, ApplyPatchResult


TRUNCATION_HINT = (
    "Rerun with a narrower command: sed range, grep pattern, "
    "inspect_trajectory.py --state/--span/--match."
)


def make_sandbox_tools(
    *,
    sandbox_dir: Path,
    tool_calls: list[dict[str, Any]],
    tool_timeout_seconds: float,
    max_tool_output_chars: int,
) -> list[Any]:
    return [
        ShellTool(
            executor=ShellExecutor(
                sandbox_dir=sandbox_dir,
                tool_calls=tool_calls,
                tool_timeout_seconds=tool_timeout_seconds,
                max_tool_output_chars=max_tool_output_chars,
            ),
            environment={"type": "local"},
            needs_approval=False,
        ),
        ApplyPatchTool(
            editor=WorkspaceEditor(sandbox_dir=sandbox_dir, tool_calls=tool_calls),
            needs_approval=False,
        ),
    ]


class ShellExecutor:
    """Executes shell commands; approvals are disabled on ShellTool."""

    def __init__(
        self,
        *,
        sandbox_dir: Path,
        tool_calls: list[dict[str, Any]],
        tool_timeout_seconds: float,
        max_tool_output_chars: int,
    ) -> None:
        self.cwd = sandbox_dir.resolve()
        self.env = os.environ.copy()
        shim_dir = self.cwd / ".openai_sdk_runner_bin"
        shim_dir.mkdir(parents=True, exist_ok=True)
        python_bin = Path(sys.executable).resolve()
        for name in ("python", "python3"):
            shim = shim_dir / name
            if not shim.exists():
                shim.symlink_to(python_bin)
        self.env["PATH"] = (
            str(shim_dir.resolve())
            + os.pathsep
            + str(python_bin.parent)
            + os.pathsep
            + self.env.get("PATH", "")
        )
        self.tool_calls = tool_calls
        self.tool_timeout_seconds = tool_timeout_seconds
        self.max_tool_output_chars = max_tool_output_chars

    async def __call__(self, request: ShellCommandRequest) -> ShellResult:
        action = request.data.action
        timeout = self._timeout(action.timeout_ms)
        outputs: list[ShellCommandOutput] = []

        for command in action.commands:
            started = time.time()
            proc = await asyncio.create_subprocess_shell(
                command,
                executable="/bin/bash",
                cwd=self.cwd,
                env=self.env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            timed_out = False
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                timed_out = True
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                stdout_bytes, stderr_bytes = await proc.communicate()

            original_stdout = _decode_output(stdout_bytes)
            original_stderr = _decode_output(stderr_bytes)
            duration_seconds = time.time() - started

            if timed_out:
                tool_response = {
                    "returncode": None,
                    "stdout": original_stdout,
                    "stderr": original_stderr,
                    "timeout_seconds": timeout,
                }
                self._record(
                    command=command,
                    returncode=None,
                    duration_seconds=duration_seconds,
                    timed_out=True,
                    stdout_text=original_stdout,
                    stderr_text=original_stderr,
                    stdout_truncated=False,
                    stderr_truncated=False,
                    tool_response=tool_response,
                )
                outputs.append(
                    ShellCommandOutput(
                        command=command,
                        stdout=original_stdout,
                        stderr=original_stderr or f"shell command timed out after {timeout}s",
                        outcome=ShellCallOutcome(type="timeout", exit_code=None),
                        provider_data=tool_response,
                    )
                )
                break

            stdout_truncated = len(original_stdout) > self.max_tool_output_chars
            stderr_truncated = len(original_stderr) > self.max_tool_output_chars
            stdout = _trim(original_stdout, self.max_tool_output_chars)
            stderr = _trim(original_stderr, self.max_tool_output_chars)

            if stdout_truncated or stderr_truncated:
                stream_details: list[str] = []
                if stdout_truncated:
                    stream_details.append(f"stdout was {len(original_stdout)} chars")
                if stderr_truncated:
                    stream_details.append(f"stderr was {len(original_stderr)} chars")
                message = (
                    f"Output was too large ({'; '.join(stream_details)}), "
                    f"cap is {self.max_tool_output_chars}. {TRUNCATION_HINT}"
                )
                tool_response = {
                    "returncode": 2,
                    "error": "OUTPUT_TRUNCATED",
                    "message": message,
                    "stdout": stdout,
                    "stderr": stderr,
                    "original_returncode": proc.returncode,
                    "stdout_chars": len(original_stdout),
                    "stderr_chars": len(original_stderr),
                    "max_output_chars": self.max_tool_output_chars,
                    "stdout_truncated": stdout_truncated,
                    "stderr_truncated": stderr_truncated,
                }
                metadata = {
                    key: value
                    for key, value in tool_response.items()
                    if key not in {"stdout", "stderr"}
                }
                stderr = "\n".join(
                    part for part in (stderr, message, json.dumps(metadata)) if part
                )
                exit_code = 2
            else:
                tool_response = {
                    "returncode": proc.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "stdout_chars": len(original_stdout),
                    "stderr_chars": len(original_stderr),
                    "max_output_chars": self.max_tool_output_chars,
                    "stdout_truncated": False,
                    "stderr_truncated": False,
                }
                exit_code = proc.returncode

            self._record(
                command=command,
                returncode=exit_code,
                duration_seconds=duration_seconds,
                timed_out=False,
                stdout_text=original_stdout,
                stderr_text=original_stderr,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
                tool_response=tool_response,
            )
            outputs.append(
                ShellCommandOutput(
                    command=command,
                    stdout=stdout,
                    stderr=stderr,
                    outcome=ShellCallOutcome(type="exit", exit_code=exit_code),
                    provider_data=tool_response,
                )
            )

        return ShellResult(
            output=outputs,
            max_output_length=self.max_tool_output_chars,
            provider_data={"working_directory": str(self.cwd)},
        )

    def _timeout(self, timeout_ms: int | None) -> float:
        if timeout_ms is None:
            return self.tool_timeout_seconds
        return min(self.tool_timeout_seconds, max(timeout_ms / 1000.0, 0.001))

    def _record(
        self,
        *,
        command: str,
        returncode: int | None,
        duration_seconds: float,
        timed_out: bool,
        stdout_text: str,
        stderr_text: str,
        stdout_truncated: bool,
        stderr_truncated: bool,
        tool_response: dict[str, Any],
    ) -> None:
        self.tool_calls.append(
            {
                "tool": "shell",
                "command": command,
                "returncode": returncode,
                "duration_seconds": duration_seconds,
                "timed_out": timed_out,
                "stdout_chars": len(stdout_text),
                "stderr_chars": len(stderr_text),
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "tool_response": tool_response,
            }
        )


class WorkspaceEditor:
    """Applies apply_patch operations inside the prepared benchmark sandbox."""

    def __init__(self, *, sandbox_dir: Path, tool_calls: list[dict[str, Any]]) -> None:
        self._root = sandbox_dir.resolve()
        self.tool_calls = tool_calls

    def create_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        started = time.time()
        try:
            if operation.diff is None:
                raise RuntimeError(f"missing diff for create_file: {operation.path}")
            target = self._resolve(operation.path, ensure_parent=True)
            content = apply_diff("", operation.diff, mode="create")
            target.write_text(content, encoding="utf-8")
            return self._result(
                operation,
                started,
                "completed",
                f"Created {self._relative(target)}",
            )
        except Exception as exc:
            self._record(operation, started, "failed", str(exc))
            raise

    def update_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        started = time.time()
        try:
            if operation.diff is None:
                raise RuntimeError(f"missing diff for update_file: {operation.path}")
            target = self._resolve(operation.path)
            updated = apply_diff(target.read_text(encoding="utf-8"), operation.diff)
            destination = self._resolve(operation.move_to) if operation.move_to else target
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(updated, encoding="utf-8")
            if destination != target:
                target.unlink()
                output = (
                    f"Updated {self._relative(target)}\n"
                    f"Moved {self._relative(target)} to {self._relative(destination)}"
                )
            else:
                output = f"Updated {self._relative(target)}"
            return self._result(operation, started, "completed", output)
        except Exception as exc:
            self._record(operation, started, "failed", str(exc))
            raise

    def delete_file(self, operation: ApplyPatchOperation) -> ApplyPatchResult:
        started = time.time()
        try:
            target = self._resolve(operation.path)
            target.unlink(missing_ok=True)
            return self._result(
                operation,
                started,
                "completed",
                f"Deleted {self._relative(target)}",
            )
        except Exception as exc:
            self._record(operation, started, "failed", str(exc))
            raise

    def _resolve(self, value: str, ensure_parent: bool = False) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError("apply_patch path must be non-empty")
        candidate = Path(value)
        target = candidate if candidate.is_absolute() else self._root / candidate
        target = target.resolve()
        if os.path.commonpath([str(self._root), str(target)]) != str(self._root):
            raise RuntimeError(f"apply_patch path escapes sandbox: {value}")
        if ensure_parent:
            target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _relative(self, path: Path) -> str:
        return path.relative_to(self._root).as_posix()

    def _result(
        self,
        operation: ApplyPatchOperation,
        started: float,
        status: str,
        output: str,
    ) -> ApplyPatchResult:
        self._record(operation, started, status, output)
        return ApplyPatchResult(status=status, output=output)

    def _record(
        self,
        operation: ApplyPatchOperation,
        started: float,
        status: str,
        output: str,
    ) -> None:
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


def _decode_output(value: bytes | bytearray | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return bytes(value).decode("utf-8", errors="replace")


def _trim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[truncated to {max_chars} chars]"
