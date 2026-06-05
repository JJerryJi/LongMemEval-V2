# Task Overview

You are acting as the strategy-memory consolidation agent.

You are running in one query attempt directory. The retrieval query has already
finished. Your job is to update `LEARNED_RETRIEVAL_STRATEGY.md` in a succinct way 
so future memory queries can retrieve evidence faster and more exactly.

Do not answer the user question. Do not modify `sandbox/memory_module_output.json`.


# Available Files

Useful files in the current directory:

- `sandbox/question.json`: the query that was just handled.
- `sandbox/memory_module_output.json`: the retrieval result returned to the
  downstream reader.
- `summary.json`: retrieval metadata, selected spans, valid/invalid spans, and
  memory markdown.
- `events.json`: JSON event stream from the retrieval Codex run, if present.
- `last_message.txt`: final retrieval-agent message, if present.
- `sandbox/trajectories/`: the haystack used by the retrieval query.
- `LEARNED_RETRIEVAL_STRATEGY.md`: the shared online strategy file to update.

Read only the files you need. Prefer `summary.json` and
`sandbox/memory_module_output.json` first. Use `events.json`, `last_message.txt`,
or cited trajectory spans only when they help infer a reusable retrieval lesson.


# Update Policy

Update only `LEARNED_RETRIEVAL_STRATEGY.md`.

The strategy file should have two main sections:

- `## Past Queries`: reusable prior query evidence. Each entry should say what
  the prior query was looking for, where the related evidence fell, when it can
  be reused, and the fast path for reuse.
- `## Strategies`: reusable search tactics, shortcuts, exactness guards, and
  gotchas that make future memory search faster or more exact.

Keep the file readable and compact. Prefer one top-level `## Past Queries`
section and one top-level `## Strategies` section. Merge duplicate or stale
entries when doing so improves readability.


# What To Add

Add or revise a `Past Queries` entry if the completed retrieval found evidence
that future queries can plausibly reuse.

Good `Past Queries` entries look like:

```markdown
| Looking for | Evidence found | Fast path |
|---|---|---|
| <query target> | trajectory `<id>`, states <start>-<end>, <what the span showed> | inspect this span first; if it matches the current target, reuse it without broad search |
```

Add or revise a `Strategies` entry if the completed retrieval revealed a
general shortcut, route, exactness guard, or gotcha.

Good `Strategies` entries look like:

```markdown
| When | Try first | Gotcha |
|---|---|---|
| <query shape or retrieval situation> | <fast search route, summary terms, helper command pattern, or span inspection tactic> | <near-match trap, unsupported transfer, or exactness check to avoid mistakes> |
```

Tables are preferred. Move fast and do not spend too much time over-exploring.


# Final Check

Before finishing:

- Make sure `LEARNED_RETRIEVAL_STRATEGY.md` still has clear `Past Queries` and
  `Strategies` content.
- Make sure the update helps future retrieval, not final answer memorization.
- Do not create any required output JSON; editing the strategy markdown is the
  only required output.
