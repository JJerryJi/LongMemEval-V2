from __future__ import annotations

import asyncio
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

from .openai_sdk_tools import make_sandbox_tools


TIMEOUT_CLEANUP_GRACE_SECONDS = 5.0


# CODEX_SYSTEM_INSTRUCTIONS = (
#     "You are a local filesystem memory retrieval agent running in a sandbox. "
#     "Your job is to inspect local files and return the most relevant evidence for the request. "
#     "Use the shell tool to inspect files, search text, and run local helper commands in the current sandbox. "
#     "Use apply_patch only when the task explicitly requires creating or editing files. "
#     "Start with targeted discovery: read the request, inspect compact indexes, summaries, or manifests first, "
#     "then open only the files and spans needed to verify the evidence. "
#     "Prefer scoped rg searches, sed ranges, and focused helper-script invocations over broad dumps. "
#     "Do not load or run local or Hugging Face vision-language/image encoder models. "
# )

CODEX_SYSTEM_INSTRUCTIONS = '''
You are a file system agent. You and the user share the same workspace and collaborate to achieve the user's goals.

# Personality

You are a deeply pragmatic, effective software engineer. You take engineering quality seriously, and collaboration comes through as direct, factual statements. You communicate efficiently, keeping the user clearly informed about ongoing actions without unnecessary detail.

## Values
You are guided by these core values:
- Clarity: You communicate reasoning explicitly and concretely, so decisions and tradeoffs are easy to evaluate upfront.
- Pragmatism: You keep the end goal and momentum in mind, focusing on what will actually work and move things forward to achieve the user's goal.
- Rigor: You expect technical arguments to be coherent and defensible, and you surface gaps or weak assumptions politely with emphasis on creating clarity and moving the task forward.

## Interaction Style
You communicate concisely and respectfully, focusing on the task at hand. You always prioritize actionable guidance, clearly stating assumptions, environment prerequisites, and next steps. Unless explicitly asked, you avoid excessively verbose explanations about your work.

You avoid cheerleading, motivational language, or artificial reassurance, or any kind of fluff. You don't comment on user requests, positively or negatively, unless there is reason for escalation. You don't feel like you need to fill the space with words, you stay concise and communicate what is necessary for user collaboration - not more, not less.

## Escalation
You may challenge the user to raise their technical bar, but you never patronize or dismiss their concerns. When presenting an alternative approach or solution to the user, you explain the reasoning behind the approach, so your thoughts are demonstrably correct. You maintain a pragmatic mindset when discussing these tradeoffs, and so are willing to work with the user after concerns have been noted.

# General
As an expert file system agent, your primary focus is executing commands and helping the user complete their task in the current environment. You build context by examining the files first without making assumptions or jumping to conclusions.
- Start with targeted discovery: read the request, inspect compact indexes, summaries, or manifests first, then open only the files and spans needed to verify the evidence.
- When searching for text or files, prefer using `rg` or `rg --files` respectively because `rg` is much faster than alternatives like `grep`. However, prefer scoped rg searches, sed ranges, and focused helper script invocations (if any) over broad dumps.

## Editing constraints

- Always use apply_patch for manual code edits. Do not use cat or any other commands when creating or editing files. Formatting commands or bulk edits don't need to be done with apply_patch.
- Do not use Python to read/write files when a simple shell command or apply_patch would suffice.
- Do not load or run local or Hugging Face vision-language/image encoder models.

## Autonomy and persistence
Persist until the task is fully handled end-to-end within the current turn whenever feasible: do not stop at analysis or partial fixes; carry changes through implementation, verification, and a clear explanation of outcomes unless the user explicitly pauses or redirects you.
'''


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class OpenAISDKOuterTimeoutError(TimeoutError):
    pass


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
    responses_transport: str = "websocket"
    api_connect_timeout_seconds: float = 15.0
    api_read_timeout_seconds: float = 300.0
    api_write_timeout_seconds: float = 300.0
    api_pool_timeout_seconds: float = 300.0
    api_max_retries: int = 0


@dataclass
class OpenAISDKRunResult:
    final_output: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    error_detail: str | None = None
    error_traceback: str = ""
    timed_out: bool = False


def load_agents_sdk() -> dict[str, Any]:
    try:
        from agents import (
            Agent,
            ModelSettings,
            OpenAIProvider,
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
        "OpenAIProvider": OpenAIProvider,
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


def ensure_non_negative_int(value: object, *, field_name: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{field_name} must be a non-negative integer",
    )
    return int(value)


def build_agent_input(
    *,
    user_prompt: str,
) -> str:
    return ensure_string(user_prompt, field_name="user_prompt")


def _usage_to_dict(usage: object) -> dict[str, int]:
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "requests": int(getattr(usage, "requests", 0) or 0),
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "cached_input_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "reasoning_output_tokens": int(getattr(output_details, "reasoning_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _empty_usage_totals() -> dict[str, int]:
    return {
        "requests": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0,
    }


def summarize_run_usage(result: object) -> dict[str, Any]:
    totals = _empty_usage_totals()
    raw_responses = getattr(result, "raw_responses", []) or []
    raw_response_usage: list[dict[str, Any]] = []
    for response in raw_responses:
        usage = getattr(response, "usage", None)
        if usage is None:
            continue
        usage_dict = _usage_to_dict(usage)
        for key in totals:
            totals[key] += usage_dict[key]
        raw_response_usage.append(
            {
                "response_id": getattr(response, "response_id", None),
                "request_id": getattr(response, "request_id", None),
                **usage_dict,
            }
        )

    return {
        **totals,
        "raw_response_count": len(raw_responses),
        "raw_responses_with_usage": len(raw_response_usage),
        "raw_response_usage": raw_response_usage,
    }


class OpenAISDKRunner:
    def __init__(self, config: OpenAISDKRunnerConfig) -> None:
        self.config = config
        require(
            config.responses_transport == "websocket",
            "OpenAISDKRunner only supports responses_transport='websocket'",
        )
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
                usage=summarize_run_usage(result),
            )
        except OpenAISDKOuterTimeoutError as exc:
            return OpenAISDKRunResult(
                tool_calls=tool_calls,
                error_detail=f"OpenAI Agents SDK run timed out after {self.config.timeout_seconds}s",
                error_traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                timed_out=True,
            )
        except (TimeoutError, httpx.TimeoutException) as exc:
            detail = str(exc) or exc.__class__.__name__
            return OpenAISDKRunResult(
                tool_calls=tool_calls,
                error_detail=f"OpenAI Agents SDK run timed out: {detail}",
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
        OpenAIProvider = sdk["OpenAIProvider"]
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

        async def run_with_timeout() -> Any:
            api_key = os.getenv(self.config.api_key_env)
            client = AsyncOpenAI(
                api_key=api_key,
                # The Agents SDK websocket transport builds the handshake from
                # client.default_headers, not the OpenAI client's auth_headers.
                default_headers={"Authorization": f"Bearer {api_key}"},
                timeout=httpx.Timeout(
                    connect=self.config.api_connect_timeout_seconds,
                    read=self.config.api_read_timeout_seconds,
                    write=self.config.api_write_timeout_seconds,
                    pool=self.config.api_pool_timeout_seconds,
                ),
                max_retries=self.config.api_max_retries,
            )
            provider = OpenAIProvider(
                openai_client=client,
                use_responses_websocket=True,
            )
            try:
                run_config = RunConfig(
                    model_provider=provider,
                    tool_execution=ToolExecutionConfig(max_function_tool_concurrency=1),
                )
                return await Runner.run(
                    agent,
                    agent_input,
                    max_turns=self.config.max_turns,
                    run_config=run_config,
                )
            finally:
                try:
                    await provider.aclose()
                finally:
                    await client.close()

        async def run_with_outer_timeout() -> Any:
            task = asyncio.create_task(run_with_timeout())
            done, _pending = await asyncio.wait({task}, timeout=self.config.timeout_seconds)
            if task in done:
                return await task
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=TIMEOUT_CLEANUP_GRACE_SECONDS)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
            raise OpenAISDKOuterTimeoutError(
                f"OpenAI Agents SDK run timed out after {self.config.timeout_seconds}s"
            )

        return asyncio.run(run_with_outer_timeout())
