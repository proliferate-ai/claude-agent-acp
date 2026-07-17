# AnyHarness v0.59 behavior port

This adapter branch ports the still-relevant behavior from the historical
`proliferate-ai/claude-agent-acp` PR #25 head
`c7eeb3816f1d447c35d00f0203ae9c72cd087ace` onto v0.59 after PR #28, whose merge
is `baefcb0d5acdb91a47689e703c518ca288814e09`. The old implementation is intent
only; none of its architecture was cherry-picked.

## Retained behavior

- `_anyharness/activity/list` returns exactly `{ processes, subagents }`.
- Process and subagent changes use zero-text `agent_message_chunk` notifications
  tagged with schema v1 `process_upserted` or `subagent_upserted` metadata.
- Background Bash output and Claude child transcripts advertise the current
  canonical `{ kind: "tail_file", path }` feed shape. A subagent's feed stays
  on its native child JSONL after terminal task notifications; their
  `output_file` is a result artifact, not the live transcript.
- Task lifecycle events update stable in-memory rosters, including terminal
  entries. A bounded tool-use cache correlates Bash commands and late output
  paths without growing for the session lifetime. Subagent start also consults
  the latest background-task level because Claude may emit that level before
  the start bookend. PostToolUse hooks from children cannot replace the parent
  transcript tailer with a child JSONL path.
- Loop fires are attributed only from Claude's current native cron transcript
  row (`type: "user"`, `isMeta: true`, `promptSource: "sdk"`, valid
  `timestamp`) and require one exact active native prompt created no later than
  that row. Pending provisional loops and stale transcript history are
  ineligible. A non-recurring loop is removed after its one authoritative fire.
- A loop set requested during a live or queued turn returns and emits the same
  provisional id, defers its native command until a true idle boundary, and
  aliases that id to the CronCreate id when the native tool result arrives.
  Clear requests follow the alias and also defer safely. Injection completion
  and main-thread CronCreate observation are correlated by the active injection
  UUID after excluding hooks with a child `agent_id`, so GoalPort traffic or an
  unrelated same-prompt child cron cannot consume the wrong request. A command
  that finishes without CronCreate retires its pending and provisional state
  before a same-prompt retry can be matched. A clear racing CronCreate keeps
  both the provisional and reconciled native ids retired while native deletion
  queues.

These wire shapes are reconciled against the current Proliferate consumer, not
the historical PR metadata.

## Deliberately not ported

- #25's obsolete ActivityPort metadata is omitted: `activity/list` does not
  repeat goal/loop state, feeds use the current `kind` discriminator instead of
  `transport`, and process/subagent rows do not expose historical `updatedAtMs`
  or prompt fields. The current flat usage siblings are retained; the current
  consumer shapes are authoritative.
- The old per-prompt idle pump and child-event demultiplexer are superseded by
  v0.59's persistent consumer and PR #28's native-subagent transcript lanes,
  bounded cleanup, permission attribution, and background turn hold.
- Synthetic child transcript copies are omitted. Activity exposes the real
  Claude subagent JSONL path while PR #28 continues to stream native child
  output with `parentToolCallId` metadata.
- Session-cron reconciliation and resume seeding are omitted because current
  Claude cron jobs are session-process state, not a durable transcript roster.
- The historical response-only provisional loop is replaced because the
  current runtime confirms loop creation by observing an upsert for the exact
  id returned from `loop/set`.
- SDK user-message replay and the previous first-loop fallback are not used for
  fire attribution; both are ambiguous when prompts overlap or an ordinary
  autonomous turn occurs.
- Old goal deferral, GoalPort/LoopPort contract changes, package renames,
  dependency changes, and release-version changes are omitted. Existing v0.59
  contracts and package version `0.59.0-proliferate.1` remain intact.
