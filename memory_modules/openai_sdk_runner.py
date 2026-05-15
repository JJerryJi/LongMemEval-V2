from __future__ import annotations

import asyncio
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .openai_sdk_tools import make_sandbox_tools


CODEX_SYSTEM_INSTRUCTIONS = (
    "You are a coding agent running in a local sandbox. "
    "Be precise, safe, and task-focused. "
    "Use the shell tool to inspect and run local commands in the current sandbox. "
    "Use apply_patch for local file edits when it is a better fit than shell redirection. "
    "If tool output is truncated, rerun a narrower command such as a sed range, grep pattern, "
    "or a focused helper-script invocation. "
    "Prefer targeted reads over broad dumps, respect the current working directory, and avoid "
    "destructive, administrative, or network actions unless the task explicitly requires them. "
    "When the task asks you to write an output file, create or overwrite exactly that file before "
    "returning your final response."
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


@dataclass(frozen=True)
class OpenAISDKRunnerConfig:
    model: str
    reasoning_effort: str
    timeout_seconds: float
    max_turns: int
    api_key_env: str
    tool_timeout_seconds: float
    max_tool_output_chars: int
    agent_name: str = "OpenAISDKRunner"


@dataclass
class OpenAISDKRunResult:
    final_output: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    error_detail: str | None = None
    error_traceback: str = ""
    timed_out: bool = False


def load_agents_sdk() -> dict[str, Any]:
    try:
        from agents import (
            Agent,
            ModelSettings,
            RunConfig,
            Runner,
            ToolExecutionConfig,
        )
        from openai.types.shared import Reasoning
    except ImportError as exc:
        raise RuntimeError(
            "openai sdk runner requires the openai-agents package. "
            "Install dependencies from requirements.txt or pyproject.toml."
        ) from exc
    return {
        "Agent": Agent,
        "ModelSettings": ModelSettings,
        "Reasoning": Reasoning,
        "RunConfig": RunConfig,
        "Runner": Runner,
        "ToolExecutionConfig": ToolExecutionConfig,
    }


def ensure_string(value: object, *, field_name: str) -> str:
    require(isinstance(value, str) and value.strip(), f"{field_name} must be a non-empty string")
    return value.strip()


def ensure_positive_float(value: object, *, field_name: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0.0,
        f"{field_name} must be a positive number",
    )
    return float(value)


def ensure_positive_int(value: object, *, field_name: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{field_name} must be a positive integer",
    )
    return int(value)


def build_agent_input(
    *,
    user_prompt: str,
) -> str:
    return ensure_string(user_prompt, field_name="user_prompt")


class OpenAISDKRunner:
    def __init__(self, config: OpenAISDKRunnerConfig) -> None:
        self.config = config
        require(
            os.getenv(config.api_key_env),
            f"Missing OpenAI API key via env {config.api_key_env}",
        )
        load_agents_sdk()

    def run(
        self,
        *,
        sandbox_dir: Path,
        user_prompt: str,
    ) -> OpenAISDKRunResult:
        tool_calls: list[dict[str, Any]] = []
        try:
            result = self._run_agent(
                sandbox_dir=sandbox_dir,
                user_prompt=user_prompt,
                tool_calls=tool_calls,
            )
            return OpenAISDKRunResult(
                final_output=str(getattr(result, "final_output", "")),
                tool_calls=tool_calls,
            )
        except asyncio.TimeoutError as exc:
            return OpenAISDKRunResult(
                tool_calls=tool_calls,
                error_detail=f"OpenAI Agents SDK run timed out after {self.config.timeout_seconds}s",
                error_traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                timed_out=True,
            )
        except Exception as exc:
            return OpenAISDKRunResult(
                tool_calls=tool_calls,
                error_detail=str(exc),
                error_traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )

    def _run_agent(
        self,
        *,
        sandbox_dir: Path,
        user_prompt: str,
        tool_calls: list[dict[str, Any]],
    ) -> Any:
        sdk = load_agents_sdk()
        Agent = sdk["Agent"]
        ModelSettings = sdk["ModelSettings"]
        Reasoning = sdk["Reasoning"]
        RunConfig = sdk["RunConfig"]
        Runner = sdk["Runner"]
        ToolExecutionConfig = sdk["ToolExecutionConfig"]

        agent = Agent(
            name=self.config.agent_name,
            instructions=CODEX_SYSTEM_INSTRUCTIONS,
            model=self.config.model,
            model_settings=ModelSettings(
                reasoning=Reasoning(effort=self.config.reasoning_effort),
                parallel_tool_calls=False,
            ),
            tools=make_sandbox_tools(
                sandbox_dir=sandbox_dir,
                tool_calls=tool_calls,
                tool_timeout_seconds=self.config.tool_timeout_seconds,
                max_tool_output_chars=self.config.max_tool_output_chars,
            ),
        )
        agent_input = build_agent_input(
            user_prompt=user_prompt,
        )
        run_config = RunConfig(
            tool_execution=ToolExecutionConfig(max_function_tool_concurrency=1),
        )

        async def run_with_timeout() -> Any:
            return await asyncio.wait_for(
                Runner.run(
                    agent,
                    agent_input,
                    max_turns=self.config.max_turns,
                    run_config=run_config,
                ),
                timeout=self.config.timeout_seconds,
            )

        return asyncio.run(run_with_timeout())
