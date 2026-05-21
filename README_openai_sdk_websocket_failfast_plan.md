# OpenAI SDK WebSocket Fail-Fast Plan

## Goal

Change the AgentRunbook-C OpenAI SDK backend so model communication uses the OpenAI Agents SDK Responses WebSocket transport instead of the default HTTP Responses transport. The change is intended to remove the hidden 600s non-streaming HTTP read-timeout behavior from this backend and make long post-tool stalls fail around Codex-like 300s stream idle timing.

This is not a feature flag. WebSocket transport should become the backend behavior for this runner.

## Target Behavior

- Use Responses WebSocket transport.
- Use explicit OpenAI client timeout settings:
  - connect: 15s
  - read/receive: 300s
  - write/send: 300s
  - pool/request-lock wait: 300s
- Disable OpenAI Python provider-managed retries with `max_retries=0`.
- Fail fast. Do not add HTTP fallback.
- Keep existing local tool behavior unchanged:
  - same sandbox tool implementation
  - same per-tool timeout
  - same max tool output chars
  - same max turns
  - same prompt/system instructions unless separately requested

## Expected Latency Impact

This targets the post-tool model/API wait, not local shell runtime.

It should reduce or cap the observed 600s tail if the tail is caused by the current non-streaming HTTP `responses.create(...)` call waiting on the OpenAI Python client's default 600s read timeout or hidden retries.

It may not eliminate all slow cases. If the server accepts a huge post-tool context but emits no WebSocket event for 300s, the attempt should now fail around 300s rather than silently waiting around 600s or longer.

## Affected Files

### `memory_modules/openai_sdk_runner.py`

Primary implementation file.

Planned changes:

- Import `httpx` and `AsyncOpenAI`.
- Load `OpenAIProvider` from the Agents SDK in `load_agents_sdk()`.
- Extend `OpenAISDKRunnerConfig` with explicit API transport settings:
  - `api_connect_timeout_seconds: float = 15.0`
  - `api_read_timeout_seconds: float = 300.0`
  - `api_write_timeout_seconds: float = 300.0`
  - `api_pool_timeout_seconds: float = 300.0`
  - `api_max_retries: int = 0`
  - `responses_transport: str = "websocket"` or no field if hardcoded is preferred
- In `_run_agent(...)`, build:

```python
client = AsyncOpenAI(
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
```

- Pass the provider into `RunConfig`:

```python
run_config = RunConfig(
    model_provider=provider,
    tool_execution=ToolExecutionConfig(max_function_tool_concurrency=1),
)
```

- Do not add HTTP fallback.
- Keep the existing outer `asyncio.wait_for(..., timeout=self.config.timeout_seconds)` run timeout.

### `memory_modules/agentrunbook_c_openai_sdk.py`

Artifact provenance and config reporting file.

Planned changes:

- Pass the new default timeout/retry values into `OpenAISDKRunnerConfig` if they are explicit config fields.
- Add transport/runtime provenance to `memory_config` under `query_openai_sdk_params`, for example:
  - `responses_transport: "websocket"`
  - `api_connect_timeout_seconds: 15.0`
  - `api_read_timeout_seconds: 300.0`
  - `api_write_timeout_seconds: 300.0`
  - `api_pool_timeout_seconds: 300.0`
  - `api_max_retries: 0`
  - `fail_fast: true`
- Add the same fields to `_runner_summary_fields()` so every `stdout.log` proves which transport behavior was used.

## Validation Plan

1. Compile-check the modified modules:

```bash
python -m py_compile memory_modules/openai_sdk_runner.py memory_modules/agentrunbook_c_openai_sdk.py
```

2. Run a tiny SDK smoke query to verify WebSocket connectivity and no-tool model response.

3. Rerun a few known slow question IDs from the previous tail analysis, such as:

- `7e32e4a2`
- `6c285a23`
- `f02bec54`
- `c3dbec17`

4. Compare the new traces against the current HTTP run:

- total attempt duration
- shell tool duration
- error/timeout type
- whether any attempt clusters near 300s instead of 600s
- whether `stdout.log` includes `responses_transport: "websocket"` and `api_max_retries: 0`

## Success Criteria

- The SDK backend no longer uses the default HTTP Responses path for model calls.
- Hidden OpenAI Python retries are disabled for this runner.
- Long no-event model waits fail at the configured 300s read/receive timeout.
- Tool execution behavior and model prompt behavior are otherwise unchanged.
