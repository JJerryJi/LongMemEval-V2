from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .oai_agents_sdk import OaiAgentsSDKRunResult, OaiAgentsSDKRunner


ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "agentrunbook_online_learning"
CONSOLIDATION_INSTRUCTION_PATH = ASSET_ROOT / "CONSOLIDATE_STRATEGY.md"
STRATEGY_SKELETON_PATH = ASSET_ROOT / "LEARNED_RETRIEVAL_STRATEGY_SKELETON.md"
STRATEGY_FILENAME = "LEARNED_RETRIEVAL_STRATEGY.md"
ONLINE_LEARNING_SYSTEM_INSTRUCTIONS = """
You are a file system learning agent. You inspect local files, derive reusable operational insights, and consolidate them into working tips that help future agents complete related tasks faster and more accurately.

# Personality

You are a deeply pragmatic, effective software engineer. You take engineering quality seriously, and collaboration comes through as direct, factual statements. You communicate efficiently, keeping the user clearly informed about ongoing actions without unnecessary detail.

## Values
You are guided by these core values:
- Clarity: You communicate reasoning explicitly and concretely, so decisions and tradeoffs are easy to evaluate upfront.
- Pragmatism: You keep the end goal and momentum in mind, focusing on what will actually work and move things forward.
- Rigor: You expect technical arguments to be coherent and defensible, and you surface gaps or weak assumptions politely.

## Interaction Style
You communicate concisely and respectfully, focusing on the task at hand. You prioritize actionable guidance, clearly stating assumptions, environment prerequisites, and next steps. Unless explicitly asked, avoid excessively verbose explanations.

You avoid cheerleading, motivational language, artificial reassurance, and fluff. You do not fill space with words; communicate what is necessary for collaboration.

# General
As an expert file system learning agent, your primary focus is to inspect local artifacts, understand what they prove, and convert only reliable patterns into concise reusable guidance. Build context by examining local files first without making assumptions or jumping to conclusions.

- Start with targeted discovery: read the request, inspect compact indexes, summaries, manifests, or instruction files first, then open only the files and spans needed to verify the evidence.
- When searching for text or files, prefer `rg` or `rg --files`. Prefer scoped searches, sed ranges, and focused helper scripts over broad dumps.
- Avoid `find` for first-pass discovery. When expected files are not visible, check for symlinked directories and use direct paths or symlink-aware commands before concluding files are absent.

## Critical Analysis
Treat local artifacts as evidence, not authority. Before preserving a lesson, decide whether the supporting evidence is direct, contradictory, incomplete, or only a near match.

- Do not turn a near match into a positive conclusion.
- Preserve uncertainty when evidence is incomplete.
- Prefer negative exactness guards over unsupported reusable shortcuts.
- Keep distinctions clear: page type, actor or view, section boundary, field name, initial state versus post-action state, and visible evidence versus inference.
- Treat previous learned notes as retrieval leads, not answer authority. A learned note can suggest where to inspect first, but it cannot by itself prove the current question.
- Every learned note you keep must have narrow applicability and a clear reuse guard. If you cannot state when not to reuse the note, the lesson is too broad.
- Do not preserve final-answer text, option letters, accepted labels, or question-specific conclusions as reusable lessons unless they are only framed as evidence-locating hints with exact scope.
- For contradiction or premise-false lessons, require explicit page-boundary evidence before writing the guard. Keep the guard tied to the exact page, section, actor/view, and control shown in the span.
- If files disagree, current local evidence and the narrow task instruction take precedence.

## Consolidation Style
Write tips that are operational, reusable, and scoped. A good tip tells a future agent when it applies, what to inspect first, what uncertainty label applies, and what mistake to avoid. Avoid final-answer memorization, overgeneralization, and broad lessons that are not supported by the local evidence.

## Editing Constraints
Use apply_patch for manual file edits. Do not use shell redirection, heredocs, or Python scripts to write manual edits when apply_patch is sufficient. Make the smallest faithful edit needed for the task.

Do not load or run local or Hugging Face vision-language/image encoder models.

## Autonomy and Persistence
Persist until the task is fully handled within the current turn whenever feasible. Do not stop at analysis or partial fixes; carry changes through implementation, verification, and a clear explanation of outcomes unless the user explicitly pauses or redirects you.
""".strip()

QUERY_INSTRUCTION_APPENDIX = """

## Learned Retrieval Strategy

Before broad trajectory exploration, briefly read `LEARNED_RETRIEVAL_STRATEGY.md` if it exists. It is an online strategy file learned from previous queries in this run.

- The strategy file uses two retrieval sections: `Past Queries` and `Strategies`.
- Every row in those sections must carry one of four evidence statuses: `directly_supported`, `contradicts_premise`, `near_match_only`, or `insufficient`.
- First check `Past Queries` for relevant prior evidence leads. If an entry appears reusable, inspect its cited span first; reuse it only when exact scope still matches.
- Use `Strategies` as search shortcuts and exactness gotchas, not as answer authority.
- Treat learned notes as leads, not answers. Verify exact page type, actor/view, entity, section boundary, field/control name, and pre-action versus post-action state before reusing a note.
- Apply the note's narrow applicability condition. If the note has no clear applicability condition, or the current question differs on entity, page, role, section, control, workflow stage, or time/state, do not reuse it as support.
- Apply the note's reuse guard. If the guard might apply, keep searching or classify the evidence as `near_match_only` or `insufficient`; do not force a positive answer.
- Ignore answer-like wording in learned notes except as a pointer to the cited trajectory/state. Re-derive the support from the current span.
- Only a `directly_supported` row can be reused as positive support, and only after exact scope verification. Use `contradicts_premise`, `near_match_only`, and `insufficient` rows to avoid bad transfers or guide search boundaries.
- If a learned note conflicts with current evidence, current evidence wins.
- If a learned note only points to a nearby workflow, keep searching or report uncertainty; do not convert the nearby workflow into a positive answer.
- Do not edit `LEARNED_RETRIEVAL_STRATEGY.md`.

## Online-Learning Evidence Gate

This online-learning appendix extends the base Output Requirement. When online
learning is enabled, `memory_module_output.json` must include these additional
top-level fields along with `memory_markdown` and `trajectory_spans`.

Before writing the output, classify the evidence for the exact requested target:
- Classify conservatively. The label is a confidence gate for the downstream reader, not a reward for finding a related span. Do not default to `directly_supported`.
- `contradicts_premise`: the cited current span directly shows the named field/control/workflow/page does not exist or the prompt's wording is wrong on the exact page/scope in question.
- `near_match_only`: the evidence is from a similar but different page, actor/view, entity, field, time, section, or workflow.
- `directly_supported`: choose only when a cited current span directly shows the requested field, control, section, workflow step, page type, or answer with the same entity, actor/view, page/surface, section, and pre/post-action state as the question.
- `insufficient`: no direct contradiction was found, but the available evidence is missing, incomplete, or uncertain.
- Closed-set absence rule: if the current scoped page, form, list, dialog, dropdown, tab set, button group, or related-record popup shows the relevant closed set of options/fields/controls, and the requested target is absent from that closed set, classify the evidence as `contradicts_premise`, not `near_match_only` or `insufficient`.
- Use `near_match_only` only when the best evidence comes from a different page, entity, actor/view, workflow, or state and therefore cannot prove absence on the current requested target.
- If you are unsure whether a span exactly matches the question, choose `near_match_only` or `insufficient`, not `directly_supported`.

Write that classification into `memory_module_output.json`:
- `evidence_status`: one of the four labels above.
- `evidence_status_reason`: a brief reason naming the exact match, contradiction, near match, or missing evidence.
- `answer_policy`: `answer_normally` for `directly_supported`, `state_premise_false` for `contradicts_premise`, `say_exact_target_not_found` for `near_match_only`, and `abstain_unknown` for `insufficient`.

Only provide a positive answer hint for `directly_supported` evidence, and only after validating the current span rather than relying on a learned note. For `contradicts_premise`, lead with the contradiction and tell the downstream reader to abstain from the prompt's premise. For premise-flaw questions, do not answer `UNKNOWN` when exact scoped evidence shows the requested item is absent; lead with the false premise and name what the scoped evidence actually shows. For `insufficient` or `near_match_only`, preserve uncertainty instead of converting the nearest workflow into an answer.

Check exact scope before reusing evidence: page type, actor/view, entity, section boundary, field/control name, pre-action versus post-action state, and whether the question asks for a control inside a named section versus a nearby control outside that section.

Do not answer with a nearby valid workflow when the question asks for a nonexistent label, missing tab, missing direct link, missing textbox, missing upload control, missing price filter, or missing dedicated module. In these cases, the useful memory is the negative evidence.
If a field value appears only after a user action in the span, do not describe it as prepopulated. Separate initial state from post-action state.
If a link/control is outside the section named in the question, do not present it as if it were inside that section. For example, a link in a separate sidebar block is not a direct link in `Toolbox`.

The downstream reader depends on your framing. If the evidence is negative or only a near match, make that the first sentence of `## Support Analysis`.
"""


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


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _failed_apply_patch_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for call in tool_calls:
        if call.get("tool") != "apply_patch":
            continue
        response = call.get("tool_response")
        if isinstance(response, dict) and response.get("status") == "failed":
            failed.append(call)
    return failed


def _strategy_validation_errors(strategy_path: Path) -> list[str]:
    errors: list[str] = []
    if not strategy_path.exists():
        return ["strategy_file_missing"]
    text = strategy_path.read_text(encoding="utf-8")
    if not text.strip():
        errors.append("strategy_file_empty")
    required_sections = [
        "# Learned Retrieval Strategy",
        "## Past Queries",
        "## Strategies",
    ]
    for section in required_sections:
        if section not in text:
            errors.append(f"missing_section:{section}")
    required_table_headers = [
        "| Looking for | Evidence status | Evidence found | Applicability and guard | Fast path |",
        "| When | Evidence status | Try first | Guard |",
    ]
    for header in required_table_headers:
        if header not in text:
            errors.append(f"missing_table_header:{header}")
    return errors


@dataclass(frozen=True)
class AgentRunbookOnlineLearningConfig:
    enabled: bool
    strategy_memory_dir: Path | None = None

    @classmethod
    def from_params(cls, params_obj: object) -> "AgentRunbookOnlineLearningConfig":
        if params_obj is None:
            return cls(enabled=False)
        require(
            isinstance(params_obj, dict),
            "agentrunbook_c_v2 online_learning_params must be an object",
        )
        params = dict(params_obj)
        enabled = params.get("enabled", False)
        require(isinstance(enabled, bool), "online_learning_params.enabled must be a boolean")
        if not enabled:
            return cls(enabled=False)

        strategy_memory_dir_obj = params.get("strategy_memory_dir")
        strategy_memory_dir = (
            Path(strategy_memory_dir_obj).expanduser().resolve()
            if isinstance(strategy_memory_dir_obj, str) and strategy_memory_dir_obj.strip()
            else None
        )
        return cls(enabled=True, strategy_memory_dir=strategy_memory_dir)

    def to_params(self) -> dict[str, object]:
        out: dict[str, object] = {"enabled": self.enabled}
        if self.enabled:
            out["strategy_memory_dir"] = (
                str(self.strategy_memory_dir) if self.strategy_memory_dir is not None else None
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
            shutil.copy2(STRATEGY_SKELETON_PATH, strategy_file)
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
            "strategy_exposure_mode": "copy",
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
                shutil.copy2(strategy_file, sandbox_strategy_path)
                sandbox_strategy_path.chmod(0o444)
                metadata.update(
                    {
                        "strategy_enabled": True,
                        "strategy_file": str(strategy_file),
                        "before_size_bytes": _file_size(strategy_file),
                        "before_sha256": _file_sha256(strategy_file),
                    }
                )
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

        sandbox_strategy_path = Path(str(metadata["sandbox_strategy_path"]))
        before_snapshot_path = Path(str(metadata["before_snapshot_path"]))
        after_snapshot_path = attempt_dir / "strategy_after.md"
        restored = False
        restore_reason: str | None = None

        with self._lock:
            try:
                self.snapshot_strategy_file(sandbox_strategy_path, after_snapshot_path)
            except Exception as exc:
                metadata["after_snapshot_error"] = str(exc)

            before_size = _file_size(before_snapshot_path) or 0
            after_size = _file_size(after_snapshot_path) or 0
            before_sha256 = _file_sha256(before_snapshot_path)
            after_sha256 = _file_sha256(after_snapshot_path)
            query_phase_changed_strategy = before_sha256 != after_sha256
            if not attempt_result.get("success"):
                restore_reason = "attempt_failed"
            elif before_size > 0 and after_size == 0:
                restore_reason = "empty_after_nonempty_before"
            elif query_phase_changed_strategy:
                restore_reason = "query_phase_strategy_edit"

            if restore_reason is not None and before_snapshot_path.exists():
                sandbox_strategy_path.chmod(0o644) if sandbox_strategy_path.exists() else None
                shutil.copy2(before_snapshot_path, sandbox_strategy_path)
                sandbox_strategy_path.chmod(0o444)
                restored = True

        strategy_file_value = metadata.get("strategy_file")
        strategy_file = Path(strategy_file_value) if isinstance(strategy_file_value, str) else None
        metadata.update(
            {
                "after_snapshot_path": str(after_snapshot_path),
                "after_size_bytes": _file_size(after_snapshot_path),
                "after_sha256": _file_sha256(after_snapshot_path),
                "shared_final_size_bytes": _file_size(strategy_file) if strategy_file else None,
                "shared_final_sha256": _file_sha256(strategy_file) if strategy_file else None,
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
        runner: OaiAgentsSDKRunner,
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
        exposed_strategy_path = attempt_dir / STRATEGY_FILENAME
        sandbox_output_path = attempt_dir / "sandbox" / "memory_module_output.json"
        sandbox_strategy_path = attempt_dir / "sandbox" / STRATEGY_FILENAME
        asset_strategy_path = ASSET_ROOT / STRATEGY_FILENAME

        metadata: dict[str, Any] = {
            "question_id": question_id,
            "attempt_dir": str(attempt_dir),
            "summary_path": str(summary_path),
            "status": "not_started",
            "started_at_utc": None,
            "completed_at_utc": None,
            "duration_seconds": None,
            "strategy_file": None,
            "strategy_link_path": str(exposed_strategy_path),
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
            "copied_back": False,
            "restored_from_before": False,
            "restore_reason": None,
            "timed_out": False,
            "runner_error_detail": None,
            "tool_call_count": 0,
            "apply_patch_failure_count": 0,
            "tool_failure_count": 0,
            "validation_errors": [],
            "sandbox_output_before_sha256": _file_sha256(sandbox_output_path),
            "sandbox_output_after_sha256": None,
            "sandbox_strategy_before_sha256": _file_sha256(sandbox_strategy_path),
            "sandbox_strategy_after_sha256": None,
            "asset_strategy_before_sha256": _file_sha256(asset_strategy_path),
            "asset_strategy_after_sha256": None,
            "usage": None,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "events_path": str(events_path),
            "last_message_path": str(last_message_path),
            "instruction_path": str(instruction_path),
            "consolidation_summary_path": str(consolidation_summary_path),
        }

        strategy_file: Path | None = None
        run_result = OaiAgentsSDKRunResult()
        started_at_ts = time.time()
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

            instruction_text = CONSOLIDATION_INSTRUCTION_PATH.read_text(encoding="utf-8")
            instruction_path.write_text(instruction_text, encoding="utf-8")
            metadata["system_instruction_sources"] = [
                "ONLINE_LEARNING_SYSTEM_INSTRUCTIONS",
            ]
            metadata["instruction_delivery"] = "attempt_file"
            metadata["started_at_utc"] = datetime.fromtimestamp(
                started_at_ts,
                timezone.utc,
            ).isoformat()
            run_result = runner.run(
                sandbox_dir=attempt_dir,
                user_prompt=(
                    "Read CONSOLIDATE_STRATEGY.md and update "
                    "LEARNED_RETRIEVAL_STRATEGY.md. Do not modify "
                    "sandbox/memory_module_output.json."
                ),
                system_instructions=ONLINE_LEARNING_SYSTEM_INSTRUCTIONS,
                allowed_apply_patch_paths={strategy_file},
            )
            last_message_path.write_text(run_result.final_output, encoding="utf-8")
            stdout_payload = {
                "runner": "openai_agents_sdk",
                "final_output": run_result.final_output,
                "tool_calls": run_result.tool_calls,
                "usage": run_result.usage,
            }
            stdout_path.write_text(
                json.dumps(stdout_payload, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            save_json(events_path, stdout_payload)
            stderr_path.write_text(run_result.error_traceback, encoding="utf-8")

            failed_apply_patch_calls = _failed_apply_patch_calls(run_result.tool_calls)
            with self._lock:
                validation_errors: list[str] = []
                run_failed = run_result.error_detail is not None
                restore_reason = None
                link_path_value = metadata.get("strategy_link_path")
                link_path = (
                    Path(link_path_value)
                    if isinstance(link_path_value, str) and link_path_value
                    else None
                )
                candidate_strategy_path = (
                    link_path
                    if metadata.get("strategy_link_mode") == "copy"
                    and link_path is not None
                    and link_path.exists()
                    else strategy_file
                )

                if run_failed:
                    restore_reason = (
                        "consolidation_timeout"
                        if run_result.timed_out
                        else "consolidation_failed"
                    )
                else:
                    before_size = _file_size(before_snapshot_path) or 0
                    after_size = _file_size(candidate_strategy_path) or 0
                    if before_size > 0 and after_size == 0:
                        validation_errors.append("empty_after_nonempty_before")
                    validation_errors.extend(_strategy_validation_errors(candidate_strategy_path))

                    metadata["sandbox_output_after_sha256"] = _file_sha256(sandbox_output_path)
                    metadata["sandbox_strategy_after_sha256"] = _file_sha256(sandbox_strategy_path)
                    metadata["asset_strategy_after_sha256"] = _file_sha256(asset_strategy_path)
                    if (
                        metadata["sandbox_output_before_sha256"]
                        != metadata["sandbox_output_after_sha256"]
                    ):
                        validation_errors.append("sandbox_memory_module_output_changed")
                    if (
                        metadata["sandbox_strategy_before_sha256"]
                        != metadata["sandbox_strategy_after_sha256"]
                    ):
                        validation_errors.append("sandbox_strategy_changed")
                    if (
                        metadata["asset_strategy_before_sha256"]
                        != metadata["asset_strategy_after_sha256"]
                    ):
                        validation_errors.append("asset_strategy_changed")
                    if validation_errors:
                        restore_reason = "validation_failed"

                metadata["validation_errors"] = validation_errors
                metadata["apply_patch_failure_count"] = len(failed_apply_patch_calls)
                metadata["tool_failure_count"] = len(failed_apply_patch_calls)
                if restore_reason is not None:
                    shutil.copy2(before_snapshot_path, strategy_file)
                    metadata["restored_from_before"] = True
                    metadata["restore_reason"] = restore_reason
                elif metadata.get("strategy_link_mode") == "copy":
                    shutil.copy2(candidate_strategy_path, strategy_file)
                    metadata["copied_back"] = True

                self.snapshot_strategy_file(strategy_file, after_snapshot_path)

            if metadata["restore_reason"] is not None:
                status = "failed"
            elif failed_apply_patch_calls:
                status = "finished_with_editor_retries"
            else:
                status = "finished"

            metadata.update(
                {
                    "status": status,
                    "completed_at_utc": utc_now_iso(),
                    "duration_seconds": time.time() - started_at_ts,
                    "timed_out": run_result.timed_out,
                    "runner_error_detail": run_result.error_detail,
                    "tool_call_count": len(run_result.tool_calls),
                    "usage": run_result.usage,
                    "after_size_bytes": _file_size(after_snapshot_path),
                    "after_sha256": _file_sha256(after_snapshot_path),
                    "final_size_bytes": _file_size(strategy_file),
                    "final_sha256": _file_sha256(strategy_file),
                    "changed": metadata.get("before_sha256") != _file_sha256(after_snapshot_path),
                }
            )
        except Exception as exc:
            if strategy_file is not None and before_snapshot_path.exists():
                try:
                    shutil.copy2(before_snapshot_path, strategy_file)
                    metadata["restored_from_before"] = True
                    metadata["restore_reason"] = "internal_error"
                except Exception as restore_exc:
                    metadata["restore_error"] = str(restore_exc)
            metadata.update(
                {
                    "status": "internal_error",
                    "completed_at_utc": utc_now_iso(),
                    "duration_seconds": time.time() - started_at_ts,
                    "error": str(exc),
                    "timed_out": run_result.timed_out,
                    "runner_error_detail": run_result.error_detail,
                    "tool_call_count": len(run_result.tool_calls),
                    "usage": run_result.usage,
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

    def copy_strategy_to(
        self,
        *,
        output_dir: Path,
        query_trace_dir: Path | None,
        workspace_dir: Path | None,
    ) -> None:
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
