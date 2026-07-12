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
- `stdout.log`: the OpenAI Agents SDK retrieval trace JSON for this attempt. In
  completed SDK-runner attempts, it can include runner/model/turn-limit
  metadata, `final_output`, aggregate `usage`, token-throughput estimates, and a
  chronological `tool_calls` list. Each tool call records the shell command,
  return code, duration, timeout/truncation flags, stdout/stderr character
  counts, and the captured `tool_response` stdout/stderr. Use this file, when
  present and parseable, to understand what the SDK retrieval agent actually
  inspected, whether it missed symlinked `trajectories/`, whether command output
  was truncated, and whether the final written memory output was based on direct
  evidence or a near match. If this file is missing or malformed, fall back to
  `summary.json`, `last_message.txt`, and `sandbox/memory_module_output.json`
  instead of inferring evidence from a broken trace.
- `last_message.txt`: final retrieval-agent message, if present.
- `sandbox/trajectories/`: the haystack used by the retrieval query.
- `LEARNED_RETRIEVAL_STRATEGY.md`: the exposed strategy file to update. In the
  normal run path this file is a symlink to the shared run-level strategy
  memory, so updating this exact path updates future attempts.
- `sandbox/LEARNED_RETRIEVAL_STRATEGY.md`: the read-only strategy snapshot that
  the retrieval agent saw during the query phase. Do not edit it.

Read only the files you need. Prefer `summary.json` and
`sandbox/memory_module_output.json` first. Use `stdout.log`, `last_message.txt`,
or cited trajectory spans only when they help infer a reusable retrieval lesson.


# Evidence Status Taxonomy

Every learned note must include one of these evidence statuses:

- `directly_supported`: selected spans directly prove the exact requested target
  under the same entity, actor/view, page/surface, section, and pre/post-action
  state as the question.
- `contradicts_premise`: selected spans directly prove the named field, control,
  section, workflow, or page premise is absent or wrong.
- `near_match_only`: the retrieval relies on a similar page, workflow, control,
  entity, actor/view, section, or state, but does not directly match the current
  question.
- `insufficient`: the retrieval says evidence is missing, uncertain,
  incomplete, empty-span, or no local trajectory evidence was found, without
  direct contradictory evidence.

Before editing the strategy file, classify the completed retrieval result. First
read `evidence_status`, `evidence_status_reason`, and `answer_policy` from
`sandbox/memory_module_output.json` or `summary.json`, but treat those fields as
claims to verify, not as proof. Check the cited spans and memory markdown before
adding a reusable note. If a `directly_supported` label is backed only by a
nearby workflow, different entity/view, missing span, or vague reason, treat it
as `near_match_only` or `insufficient` for consolidation and do not add a
positive reusable evidence entry.

Closed-set absence rule: if the cited scoped page, form, list, dialog, dropdown,
tab set, button group, or related-record popup shows the relevant closed set of
options/fields/controls, and the requested target is absent from that closed set,
treat the result as `contradicts_premise`, not `near_match_only` or
`insufficient`. Treat it as `near_match_only` only when the best evidence comes
from a different page, entity, actor/view, workflow, or state and therefore
cannot prove absence on the current requested target.

A learned note must preserve the evidence uncertainty label. Do not upgrade a
`contradicts_premise`, `near_match_only`, or `insufficient` result into a
positive answer shortcut. A prior note should help a future agent decide where
to inspect or what not to reuse; it should not let the future agent skip exact
evidence validation.

If the status is `insufficient` or `near_match_only`, usually do not edit the
strategy file. If there is a reusable lesson, add only conservative search
boundary or exactness guidance with the matching status.


# Update Policy

Update only `./LEARNED_RETRIEVAL_STRATEGY.md` in the current attempt directory.
Do not edit `sandbox/LEARNED_RETRIEVAL_STRATEGY.md`, do not search the repo for
another strategy file, and do not edit
`memory_modules/assets/agentrunbook_online_learning/LEARNED_RETRIEVAL_STRATEGY.md`
or the skeleton asset.

Use apply_patch for the strategy edit. If an apply_patch update fails, reread the
exact current section and retry with a narrower patch against
`./LEARNED_RETRIEVAL_STRATEGY.md`.

The strategy file should keep exactly these two retrieval sections:

- `## Past Queries`
- `## Strategies`

Do not create status sections such as `## directly_supported`. Status belongs in
the `Evidence status` column of the two tables.

Keep the file readable and compact. Merge duplicate or stale entries when doing
so improves readability.


# What To Add

Add or revise a `Past Queries` entry when the completed retrieval produced a
query-specific evidence location, contradiction, near-match warning, or search
boundary that future queries can plausibly reuse. The entry must be faithful to
the cited span, including whether the span is pre-action or post-action, which
page/view it is on, and whether a link/control is inside the named section or
merely nearby.

Add or revise a `Strategies` entry when the completed retrieval revealed a
general search route, shortcut, exactness guard, or gotcha that applies beyond
one specific query target.

Every learned note must include:

- an `Evidence status` cell with one of the four exact labels;
- a narrow applicability condition naming the page/surface, actor/view, entity,
  field/control/section, and pre-action versus post-action status when relevant;
- a `do not reuse if...` guard naming the nearest likely false transfer;
- cited trajectory/state provenance when the note makes a concrete evidence
  claim.

Only `directly_supported` rows may act as positive evidence leads, and only after
exact scope verification. Do not add a positive `directly_supported` entry when
the retrieval result is wrong, empty, unsupported, based on a nearby workflow, or
says the premise is uncertain. Do not turn a closest available workflow into a
reusable answer for a missing label/control/module. Do not preserve final answer
text, option letters, or accepted-answer wording as reusable guidance unless it
is rewritten as a retrieval hint tied to exact evidence and guarded against
transfer.

Good entries look like:

```markdown
## Past Queries

| Looking for | Evidence status | Evidence found | Applicability and guard | Fast path |
|---|---|---|---|---|
| <query target> | directly_supported | trajectory `<id>`, states <start>-<end>, <what the span showed> | Applies only when <exact page/surface, actor/view, entity, field/control, pre/post state> match. Do not reuse if <nearby false transfer>. | inspect this span first; if the applicability condition matches, verify the current span and then reuse it |
| <missing exact label/control/module> | contradicts_premise | trajectory `<id>`, states <start>-<end>, showed <negative fact on exact page/scope> | Applies only when <exact scope> matches. Do not reuse if the current page/scope differs. | state the premise is false; do not answer with <nearby label> |
| <nearby but different workflow> | near_match_only | trajectory `<id>`, states <start>-<end>, showed <different scope> | Do not reuse if entity/page/control does not exactly match. | keep searching for the exact target or classify as near_match_only |
| <query target> | insufficient | <files/spans/searches inspected> | <what was not verified> | <specific next route or abstain condition> |

## Strategies

| When | Evidence status | Try first | Guard |
|---|---|---|---|
| <query shape or retrieval situation> | <one of the four statuses> | <fast search route, summary terms, helper command pattern, or span inspection tactic> | <near-match trap, unsupported transfer, exactness check, or do-not-reuse condition> |
```

Move fast and do not spend too much time over-exploring.


# Final Check

Before finishing:

- Make sure `LEARNED_RETRIEVAL_STRATEGY.md` has exactly the two retrieval
  sections listed above.
- Make sure every new row includes an `Evidence status` cell with one of:
  `directly_supported`, `contradicts_premise`, `near_match_only`, or
  `insufficient`.
- Make sure you changed only `./LEARNED_RETRIEVAL_STRATEGY.md`; leave
  `sandbox/memory_module_output.json`, `sandbox/LEARNED_RETRIEVAL_STRATEGY.md`,
  and repo asset strategy files unchanged.
- Make sure the update helps future retrieval, not final answer memorization.
- Verify that every new positive `directly_supported` entry cites a non-empty
  span that directly proves the note.
- Verify that every new entry has a narrow applicability condition and a
  do-not-reuse guard. If the guard would be vague, skip the entry or rewrite it
  under a more conservative status.
- Skip the update if the only available lesson would preserve a wrong answer, a
  post-action value as a prefilled value, a nearby control as an in-section
  control, or a personal/account phone number as customer-service support.
- Do not create any required output JSON; editing the strategy markdown is the
  only required output.
