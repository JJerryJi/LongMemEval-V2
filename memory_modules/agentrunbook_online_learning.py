from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROCESS_POLL_INTERVAL_SECONDS = 0.25
TERMINATION_GRACE_SECONDS = 5.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def relative_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(relative_target)


def parse_codex_json_events(raw_stdout: str) -> tuple[list[dict[str, Any]], dict[str, int] | None]:
    events: list[dict[str, Any]] = []
    usage: dict[str, int] | None = None
    for line in raw_stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
            if payload.get("type") == "turn.completed" and isinstance(payload.get("usage"), dict):
                usage = payload["usage"]
    return events, usage


def terminate_process_group(process: subprocess.Popen[str], *, reason: str) -> tuple[str, str]:
    if process.poll() is not None:
        return process.communicate()
    print(
        f"[online-learning] terminating pid={process.pid} reason={reason}",
        file=sys.stderr,
        flush=True,
    )
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        return process.communicate(timeout=TERMINATION_GRACE_SECONDS)
    except ProcessLookupError:
        return process.communicate()
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            return process.communicate()
        return process.communicate()


ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "agentrunbook_online_learning"
CONSOLIDATION_INSTRUCTION_PATH = ASSET_ROOT / "CONSOLIDATE_STRATEGY.md"
STRATEGY_SKELETON_PATH = ASSET_ROOT / "LEARNED_RETRIEVAL_STRATEGY_SKELETON.md"
STRATEGY_FILENAME = "LEARNED_RETRIEVAL_STRATEGY.md"

QUERY_INSTRUCTION_APPENDIX = """

## Learned Retrieval Strategy

Before broad trajectory exploration, briefly read `LEARNED_RETRIEVAL_STRATEGY.md` if it exists. It is an online strategy file learned from previous queries in this run.

- First check `Past Queries`. If part of the prior retrieved spans appears reusable, consider reusing it directly and avoid broad search.
- Treat learned notes as leads, not answers. Verify exact page type, actor/view, section boundary, field name, and pre-action versus post-action state before reusing a note.
- Use `Strategies` as retrieval shortcuts and exactness gotchas.
- If a learned note conflicts with current evidence, current evidence wins.
- If a learned note only points to a nearby workflow, keep searching or report uncertainty; do not convert the nearby workflow into a positive answer.
- Do not edit `LEARNED_RETRIEVAL_STRATEGY.md`.

## Online-Learning Evidence Gate

Before writing the output, classify the evidence for the exact requested target:
- `directly_supported`: the cited state directly shows the requested field, control, section, workflow step, page type, or answer.
- `contradicts_premise`: the cited state directly shows the named field/control/workflow/page does not exist or the prompt's wording is wrong.
- `insufficient`: only nearby or partial evidence was found.
- `near_match_only`: the evidence is from a similar but different page, actor/view, field, time, or workflow.

Only provide a positive answer hint for `directly_supported` evidence. For `contradicts_premise`, lead with the contradiction and tell the downstream reader to abstain from the prompt's premise. For `insufficient` or `near_match_only`, preserve uncertainty instead of converting the nearest workflow into an answer.

Check exact scope before reusing evidence: page type, actor/view, section boundary, field name, pre-action versus post-action state, and whether the question asks for a control inside a named section versus a nearby control outside that section.

Do not answer with a nearby valid workflow when the question asks for a nonexistent label, missing tab, missing direct link, missing textbox, missing upload control, missing price filter, or missing dedicated module. In these cases, the useful memory is the negative evidence.
If a field value appears only after a user action in the span, do not describe it as prepopulated. Separate initial state from post-action state.
If a link/control is outside the section named in the question, do not present it as if it were inside that section. For example, a link in a separate sidebar block is not a direct link in `Toolbox`.

The downstream reader depends on your framing. If the evidence is negative or only a near match, make that the first sentence of `## Support Analysis`.
"""


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_size(path: Path) -> int | None:
    if not path.exists():
        return None
    return path.stat().st_size


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_string_list(value: object, *, field_name: str) -> list[str]:
    require(isinstance(value, list), f"{field_name} must be a list")
    out: list[str] = []
    for idx, item in enumerate(value):
        require(
            isinstance(item, str) and item.strip(),
            f"{field_name}[{idx}] must be a non-empty string",
        )
        out.append(item.strip())
    return out


def _ensure_positive_float(value: object, *, field_name: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0.0,
        f"{field_name} must be a positive number",
    )
    return float(value)


def _ensure_string(value: object, *, field_name: str) -> str:
    require(isinstance(value, str) and value.strip(), f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class AgentRunbookOnlineLearningConfig:
    enabled: bool
    binary: str = "codex"
    model: str = "gpt-5.4-mini"
    reasoning_effort: str = "medium"
    timeout_seconds: float = 1800.0
    extra_config: list[str] = field(default_factory=list)
    extra_args: list[str] = field(default_factory=list)
    strategy_memory_dir: Path | None = None

    @classmethod
    def from_params(cls, params_obj: object) -> "AgentRunbookOnlineLearningConfig":
        if params_obj is None:
            return cls(enabled=False)
        require(
            isinstance(params_obj, dict),
            "agentrunbook_c_openai_sdk online_learning_params must be an object",
        )
        params = dict(params_obj)
        enabled = params.get("enabled", False)
        require(
            isinstance(enabled, bool),
            "online_learning_params.enabled must be a boolean",
        )
        if not enabled:
            return cls(enabled=False)

        strategy_memory_dir_obj = params.get("strategy_memory_dir")
        strategy_memory_dir = (
            Path(strategy_memory_dir_obj).expanduser().resolve()
            if isinstance(strategy_memory_dir_obj, str) and strategy_memory_dir_obj.strip()
            else None
        )
        return cls(
            enabled=True,
            binary=_ensure_string(params.get("binary", "codex"), field_name="online_learning_params.binary"),
            model=_ensure_string(params.get("model", "gpt-5.4-mini"), field_name="online_learning_params.model"),
            reasoning_effort=_ensure_string(
                params.get("reasoning_effort", "medium"),
                field_name="online_learning_params.reasoning_effort",
            ),
            timeout_seconds=_ensure_positive_float(
                params.get("timeout_seconds", 1800.0),
                field_name="online_learning_params.timeout_seconds",
            ),
            extra_config=_ensure_string_list(
                params.get("extra_config", []),
                field_name="online_learning_params.extra_config",
            ),
            extra_args=_ensure_string_list(
                params.get("extra_args", []),
                field_name="online_learning_params.extra_args",
            ),
            strategy_memory_dir=strategy_memory_dir,
        )

    def to_params(self) -> dict[str, object]:
        out: dict[str, object] = {"enabled": self.enabled}
        if not self.enabled:
            return out
        out.update(
            {
                "binary": self.binary,
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "timeout_seconds": self.timeout_seconds,
                "extra_config": list(self.extra_config),
                "extra_args": list(self.extra_args),
                "strategy_memory_dir": (
                    str(self.strategy_memory_dir) if self.strategy_memory_dir is not None else None
                ),
            }
        )
        return out


class AgentRunbookOnlineLearning:
    def __init__(self, config: AgentRunbookOnlineLearningConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._attempt_metadata: dict[str, dict[str, Any]] = {}
        self._latest_successful_attempt_by_question: dict[str, Path] = {}
        if config.enabled:
            require(
                CONSOLIDATION_INSTRUCTION_PATH.exists(),
                f"missing online-learning consolidation instruction: {CONSOLIDATION_INSTRUCTION_PATH}",
            )
            require(
                STRATEGY_SKELETON_PATH.exists(),
                f"missing online-learning strategy skeleton: {STRATEGY_SKELETON_PATH}",
            )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def strategy_memory_dir(
        self,
        *,
        query_trace_dir: Path | None,
        workspace_dir: Path | None,
    ) -> Path:
        if self.config.strategy_memory_dir is not None:
            return self.config.strategy_memory_dir
        if query_trace_dir is not None:
            return query_trace_dir.parent / "strategy_memory"
        require(
            workspace_dir is not None,
            "online learning requires workspace_dir or query_trace_dir for strategy memory",
        )
        return workspace_dir / "strategy_memory"

    def strategy_file(
        self,
        *,
        query_trace_dir: Path | None,
        workspace_dir: Path | None,
    ) -> Path:
        return self.strategy_memory_dir(
            query_trace_dir=query_trace_dir,
            workspace_dir=workspace_dir,
        ) / STRATEGY_FILENAME

    def ensure_strategy_file(
        self,
        *,
        query_trace_dir: Path | None,
        workspace_dir: Path | None,
    ) -> Path:
        strategy_file = self.strategy_file(
            query_trace_dir=query_trace_dir,
            workspace_dir=workspace_dir,
        )
        strategy_file.parent.mkdir(parents=True, exist_ok=True)
        if not strategy_file.exists():
            strategy_file.write_text(
                STRATEGY_SKELETON_PATH.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        return strategy_file

    def snapshot_strategy_file(self, strategy_file: Path, snapshot_path: Path) -> None:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        if strategy_file.exists():
            shutil.copy2(strategy_file, snapshot_path)
        else:
            snapshot_path.write_text("", encoding="utf-8")

    def expose_to_sandbox(
        self,
        *,
        sandbox_dir: Path,
        query_trace_dir: Path | None,
        workspace_dir: Path | None,
    ) -> None:
        if not self.enabled:
            return None

        attempt_dir = sandbox_dir.parent
        sandbox_strategy_path = sandbox_dir / STRATEGY_FILENAME
        before_snapshot_path = attempt_dir / "strategy_before.md"
        metadata: dict[str, Any] = {
            "strategy_enabled": False,
            "strategy_file": None,
            "sandbox_strategy_path": str(sandbox_strategy_path),
            "strategy_link_mode": "unavailable",
            "before_snapshot_path": str(before_snapshot_path),
            "before_size_bytes": None,
            "before_sha256": None,
            "setup_error": None,
        }

        try:
            with self._lock:
                strategy_file = self.ensure_strategy_file(
                    query_trace_dir=query_trace_dir,
                    workspace_dir=workspace_dir,
                )
                self.snapshot_strategy_file(strategy_file, before_snapshot_path)
                metadata.update(
                    {
                        "strategy_enabled": True,
                        "strategy_file": str(strategy_file),
                        "before_size_bytes": _file_size(strategy_file),
                        "before_sha256": _file_sha256(strategy_file),
                    }
                )
                try:
                    relative_symlink(strategy_file, sandbox_strategy_path)
                    metadata["strategy_link_mode"] = "symlink"
                except Exception as exc:
                    shutil.copy2(strategy_file, sandbox_strategy_path)
                    metadata["strategy_link_mode"] = "copy"
                    metadata["symlink_error"] = str(exc)
        except Exception as exc:
            metadata["setup_error"] = str(exc)

        self._attempt_metadata[str(attempt_dir.resolve())] = metadata
        return None

    def latest_attempt_dir(self, *, query_trace_dir: Path | None, question_id: str) -> Path | None:
        if query_trace_dir is None:
            return None
        question_trace_dir = query_trace_dir / question_id
        if not question_trace_dir.exists():
            return None
        attempt_dirs = sorted(
            path
            for path in question_trace_dir.iterdir()
            if path.is_dir() and path.name.startswith("attempt_")
        )
        return attempt_dirs[-1] if attempt_dirs else None

    def finalize_attempt(
        self,
        *,
        query_trace_dir: Path | None,
        question_id: str,
        attempt_result: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return None
        attempt_dir = self.latest_attempt_dir(
            query_trace_dir=query_trace_dir,
            question_id=question_id,
        )
        if attempt_dir is None:
            return None
        summary_path = attempt_dir / "summary.json"
        metadata = self._attempt_metadata.pop(str(attempt_dir.resolve()), {})
        if not metadata:
            return None

        strategy_file_value = metadata.get("strategy_file")
        if not isinstance(strategy_file_value, str) or not strategy_file_value:
            self.update_summary_with_attempt(summary_path, metadata)
            return None

        strategy_file = Path(strategy_file_value)
        before_snapshot_path = Path(str(metadata["before_snapshot_path"]))
        after_snapshot_path = attempt_dir / "strategy_after.md"
        restored = False
        restore_reason: str | None = None
        query_phase_changed_strategy = False

        with self._lock:
            try:
                self.snapshot_strategy_file(strategy_file, after_snapshot_path)
            except Exception as exc:
                metadata["after_snapshot_error"] = str(exc)

            before_size = _file_size(before_snapshot_path) or 0
            after_size = _file_size(strategy_file) or 0
            after_sha256 = _file_sha256(strategy_file)
            query_phase_changed_strategy = metadata.get("before_sha256") != after_sha256
            should_restore = False
            if not attempt_result.get("success"):
                should_restore = before_snapshot_path.exists()
                restore_reason = "attempt_failed"
            elif query_phase_changed_strategy:
                should_restore = before_snapshot_path.exists()
                restore_reason = "query_phase_strategy_edit"
            elif before_size > 0 and after_size == 0:
                should_restore = before_snapshot_path.exists()
                restore_reason = "empty_after_nonempty_before"

            if should_restore:
                shutil.copy2(before_snapshot_path, strategy_file)
                restored = True

        metadata.update(
            {
                "after_snapshot_path": str(after_snapshot_path),
                "after_size_bytes": _file_size(after_snapshot_path),
                "after_sha256": _file_sha256(after_snapshot_path),
                "final_size_bytes": _file_size(strategy_file),
                "final_sha256": _file_sha256(strategy_file),
                "changed": metadata.get("before_sha256") != _file_sha256(after_snapshot_path),
                "query_phase_changed_strategy": query_phase_changed_strategy,
                "restored_from_before": restored,
                "restore_reason": restore_reason,
            }
        )
        self.update_summary_with_attempt(summary_path, metadata)
        if attempt_result.get("success"):
            self._latest_successful_attempt_by_question[question_id] = attempt_dir
        return None

    def update_summary_with_attempt(self, summary_path: Path, metadata: dict[str, Any]) -> None:
        summary_payload = self._summary_payload(summary_path)
        summary_payload["strategy_memory"] = metadata
        save_json(summary_path, summary_payload)

    def update_summary_with_consolidation(self, summary_path: Path, metadata: dict[str, Any]) -> None:
        summary_payload = self._summary_payload(summary_path)
        summary_payload["strategy_consolidation"] = metadata
        save_json(summary_path, summary_payload)

    def _summary_payload(self, summary_path: Path) -> dict[str, Any]:
        if summary_path.exists():
            try:
                summary_payload = _json_load(summary_path)
            except Exception:
                summary_payload = {}
        else:
            summary_payload = {}
        if not isinstance(summary_payload, dict):
            summary_payload = {}
        return summary_payload

    def build_consolidation_command(
        self,
        *,
        attempt_dir: Path,
        last_message_path: Path,
    ) -> list[str]:
        command = [
            self.config.binary,
            "exec",
            "-C",
            str(attempt_dir),
            "--skip-git-repo-check",
            "--ephemeral",
            "--dangerously-bypass-approvals-and-sandbox",
            "--json",
            "-o",
            str(last_message_path),
            "-m",
            self.config.model,
            "-c",
            f"model_reasoning_effort={json.dumps(self.config.reasoning_effort)}",
        ]
        for item in self.config.extra_config:
            command.extend(["-c", item])
        command.extend(self.config.extra_args)
        command.append(
            "Read CONSOLIDATE_STRATEGY.md and update LEARNED_RETRIEVAL_STRATEGY.md. "
            "Do not modify sandbox/memory_module_output.json."
        )
        return command

    def expose_consolidation_strategy_file(
        self,
        *,
        strategy_file: Path,
        attempt_dir: Path,
    ) -> tuple[Path, str, str | None]:
        link_path = attempt_dir / STRATEGY_FILENAME
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        try:
            relative_symlink(strategy_file, link_path)
            return link_path, "symlink", None
        except Exception as exc:
            shutil.copy2(strategy_file, link_path)
            return link_path, "copy", str(exc)

    def run_consolidation(
        self,
        *,
        question_id: str,
        attempt_dir: Path,
        query_trace_dir: Path | None,
        workspace_dir: Path | None,
        is_cancelled: Any | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "skipped", "reason": "strategy_memory_disabled"}
        if is_cancelled is not None and is_cancelled():
            raise KeyboardInterrupt("online-learning consolidation cancelled")

        summary_path = attempt_dir / "summary.json"
        consolidation_summary_path = attempt_dir / "strategy_consolidation_summary.json"
        instruction_path = attempt_dir / "CONSOLIDATE_STRATEGY.md"
        stdout_path = attempt_dir / "strategy_consolidation_stdout.log"
        stderr_path = attempt_dir / "strategy_consolidation_stderr.log"
        events_path = attempt_dir / "strategy_consolidation_events.json"
        last_message_path = attempt_dir / "strategy_consolidation_last_message.txt"
        before_snapshot_path = attempt_dir / "strategy_consolidation_before.md"
        after_snapshot_path = attempt_dir / "strategy_consolidation_after.md"

        metadata: dict[str, Any] = {
            "question_id": question_id,
            "attempt_dir": str(attempt_dir),
            "summary_path": str(summary_path),
            "status": "not_started",
            "started_at_utc": None,
            "completed_at_utc": None,
            "duration_seconds": None,
            "strategy_file": None,
            "strategy_link_path": None,
            "strategy_link_mode": None,
            "strategy_link_error": None,
            "before_snapshot_path": str(before_snapshot_path),
            "before_size_bytes": None,
            "before_sha256": None,
            "after_snapshot_path": str(after_snapshot_path),
            "after_size_bytes": None,
            "after_sha256": None,
            "final_size_bytes": None,
            "final_sha256": None,
            "changed": False,
            "restored_from_before": False,
            "restore_reason": None,
            "returncode": None,
            "timed_out": False,
            "usage": None,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "events_path": None,
            "last_message_path": str(last_message_path),
            "instruction_path": str(instruction_path),
            "consolidation_summary_path": str(consolidation_summary_path),
        }

        try:
            strategy_file = self.ensure_strategy_file(
                query_trace_dir=query_trace_dir,
                workspace_dir=workspace_dir,
            )
            metadata["strategy_file"] = str(strategy_file)
            with self._lock:
                self.snapshot_strategy_file(strategy_file, before_snapshot_path)
                metadata["before_size_bytes"] = _file_size(strategy_file)
                metadata["before_sha256"] = _file_sha256(strategy_file)
                link_path, link_mode, link_error = self.expose_consolidation_strategy_file(
                    strategy_file=strategy_file,
                    attempt_dir=attempt_dir,
                )
                metadata["strategy_link_path"] = str(link_path)
                metadata["strategy_link_mode"] = link_mode
                metadata["strategy_link_error"] = link_error
            instruction_path.write_text(
                CONSOLIDATION_INSTRUCTION_PATH.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            command = self.build_consolidation_command(
                attempt_dir=attempt_dir,
                last_message_path=last_message_path,
            )
            metadata["command"] = command
            started_at_ts = time.time()
            metadata["started_at_utc"] = datetime.fromtimestamp(
                started_at_ts,
                timezone.utc,
            ).isoformat()
            timed_out = False
            stdout_text = ""
            stderr_text = ""
            returncode: int | None = None
            process: subprocess.Popen[str] | None = None
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=(os.name == "posix"),
                )
                while True:
                    if is_cancelled is not None and is_cancelled():
                        raise KeyboardInterrupt("online-learning consolidation cancelled")
                    elapsed_seconds = time.time() - started_at_ts
                    remaining_seconds = self.config.timeout_seconds - elapsed_seconds
                    if remaining_seconds <= 0:
                        timed_out = True
                        stdout_text, stderr_text = terminate_process_group(
                            process,
                            reason="timeout",
                        )
                        break
                    try:
                        stdout_text, stderr_text = process.communicate(
                            timeout=min(PROCESS_POLL_INTERVAL_SECONDS, remaining_seconds),
                        )
                        break
                    except subprocess.TimeoutExpired:
                        continue
                returncode = process.returncode
            except KeyboardInterrupt:
                if process is not None:
                    stdout_text, stderr_text = terminate_process_group(
                        process,
                        reason="keyboard_interrupt",
                    )
                    returncode = process.returncode
                raise

            duration_seconds = time.time() - started_at_ts
            stdout_path.write_text(stdout_text, encoding="utf-8")
            stderr_path.write_text(stderr_text, encoding="utf-8")
            events, usage = parse_codex_json_events(stdout_text)
            if events:
                save_json(events_path, events)
                metadata["events_path"] = str(events_path)

            with self._lock:
                link_path_value = metadata.get("strategy_link_path")
                link_path = Path(link_path_value) if isinstance(link_path_value, str) else None
                if (
                    metadata.get("strategy_link_mode") == "copy"
                    and returncode == 0
                    and not timed_out
                    and link_path is not None
                    and link_path.exists()
                ):
                    shutil.copy2(link_path, strategy_file)

                self.snapshot_strategy_file(strategy_file, after_snapshot_path)
                after_size = _file_size(strategy_file) or 0
                before_size = _file_size(before_snapshot_path) or 0
                failed = timed_out or returncode != 0
                restore_reason = None
                if failed:
                    restore_reason = "consolidation_failed"
                elif before_size > 0 and after_size == 0:
                    restore_reason = "empty_after_nonempty_before"
                if restore_reason is not None and before_snapshot_path.exists():
                    shutil.copy2(before_snapshot_path, strategy_file)
                    metadata["restored_from_before"] = True
                    metadata["restore_reason"] = restore_reason

            metadata.update(
                {
                    "status": "finished" if returncode == 0 and not timed_out else "failed",
                    "completed_at_utc": utc_now_iso(),
                    "duration_seconds": duration_seconds,
                    "returncode": returncode,
                    "timed_out": timed_out,
                    "usage": usage,
                    "after_size_bytes": _file_size(after_snapshot_path),
                    "after_sha256": _file_sha256(after_snapshot_path),
                    "final_size_bytes": _file_size(strategy_file),
                    "final_sha256": _file_sha256(strategy_file),
                    "changed": metadata.get("before_sha256") != _file_sha256(after_snapshot_path),
                }
            )
        except Exception as exc:
            metadata.update(
                {
                    "status": "internal_error",
                    "completed_at_utc": utc_now_iso(),
                    "error": str(exc),
                }
            )

        save_json(consolidation_summary_path, metadata)
        self.update_summary_with_consolidation(summary_path, metadata)
        return metadata

    def attempt_for_post_query(
        self,
        *,
        query_trace_dir: Path | None,
        question_id: str,
    ) -> Path | None:
        return self._latest_successful_attempt_by_question.get(question_id) or self.latest_attempt_dir(
            query_trace_dir=query_trace_dir,
            question_id=question_id,
        )

    def copy_strategy_to(self, *, output_dir: Path, query_trace_dir: Path | None, workspace_dir: Path | None) -> None:
        if not self.enabled:
            return None
        try:
            strategy_file = self.strategy_file(
                query_trace_dir=query_trace_dir,
                workspace_dir=workspace_dir,
            )
        except Exception:
            return None
        if strategy_file.exists():
            target_dir = output_dir / "strategy_memory"
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(strategy_file, target_dir / STRATEGY_FILENAME)
        return None
