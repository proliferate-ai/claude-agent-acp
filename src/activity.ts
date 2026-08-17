import { RequestError } from "@agentclientprotocol/sdk";

/**
 * Read-only activity rosters (background processes + subagents) re-implemented
 * on the canonical claude-agent-acp 0.66.0 base (bgwork RF).
 *
 * THE RUNTIME IS THE AUTHORITY for this wire contract: field names/shapes here
 * mirror `anyharness/crates/anyharness-lib/src/domains/activity/wire.rs`
 * (`ActivityProcessWire` / `ActivitySubagentWire`, camelCase on the wire) and
 * `session_observer.rs` (the `_meta.anyharness` chunk-tagging envelope). Do not
 * rename a field here without updating that Rust contract first — a rename
 * silently degrades an ingest (see the runtime's own `#[serde(default)]` /
 * `alias` comments for examples of exactly this failure mode).
 *
 * These rosters are strictly read-only from the client's perspective: there is
 * no write path, only this emission.
 */

// ---------------------------------------------------------------------------
// Wire contract (mirrors wire.rs's ActivityProcessWire / ActivitySubagentWire)
// ---------------------------------------------------------------------------

export const ACTIVITY_SCHEMA_VERSION = 1 as const;

export const PROCESS_UPSERTED_TRANSCRIPT_EVENT = "process_upserted" as const;
export const SUBAGENT_UPSERTED_TRANSCRIPT_EVENT = "subagent_upserted" as const;

/** The attach-time roster reconcile pull. ACP strips the leading `_` before
 *  dispatch; the client (this harness) advertises/handles the underscored
 *  form, matching `_anyharness/rewindFiles`'s existing precedent. */
export const ACTIVITY_LIST_EXT_METHOD = "_anyharness/activity/list";

/** Membrane-side byte transport for a roster element's content stream. The
 *  runtime swaps this for an opaque FeedRef before anything leaves the
 *  runtime boundary — the client never learns the path. claude-agent-acp only
 *  ever emits `tail_file`; `deserialize_feed_transport` on the runtime side
 *  accepts either `transport` (this fork's discriminator key) or `kind`
 *  (codex's), so `transport` is the correct key to send here. */
export type ActivityFeed = { transport: "tail_file"; path: string };

export type ActivityProcessStatus = "running" | "exited";
export type ActivitySubagentStatus = "running" | "completed" | "failed";

export type ActivityProcessWire = {
  id: string;
  command: string;
  cwd: string | null;
  status: ActivityProcessStatus;
  exitCode: number | null;
  pid: number | null;
  startedAtMs: number;
  endedAtMs: number | null;
  feed: ActivityFeed | null;
};

export type ActivitySubagentWire = {
  id: string;
  agentType: string | null;
  description: string | null;
  model: string | null;
  background: boolean;
  status: ActivitySubagentStatus;
  summary: string | null;
  tokensUsed: number | null;
  toolCalls: number | null;
  durationSeconds: number | null;
  feed: ActivityFeed | null;
};

/** `_meta.anyharness` on a tagged zero-text `agent_message_chunk`, per
 *  `session_observer.rs`'s `ActivityChunkAnyHarnessMeta` (camelCase). */
export type ActivityChunkMeta = {
  anyharness:
    | {
        schemaVersion: typeof ACTIVITY_SCHEMA_VERSION;
        transcriptEvent: typeof PROCESS_UPSERTED_TRANSCRIPT_EVENT;
        process: ActivityProcessWire;
      }
    | {
        schemaVersion: typeof ACTIVITY_SCHEMA_VERSION;
        transcriptEvent: typeof SUBAGENT_UPSERTED_TRANSCRIPT_EVENT;
        subagent: ActivitySubagentWire;
      };
};

// ---------------------------------------------------------------------------
// Internal per-session state
// ---------------------------------------------------------------------------

export type ActivityProcessRecord = ActivityProcessWire & { updatedAtMs: number };
export type ActivitySubagentRecord = ActivitySubagentWire & { updatedAtMs: number };

export type ActivityState = {
  /** Keyed by the SDK `task_id` (== `backgroundTaskId` on the spawning Bash
   *  tool's structured result). */
  processes: Map<string, ActivityProcessRecord>;
  /** Keyed by the SDK `task_id` (== `agentId` on the spawning Agent/Task
   *  tool's structured result — `task_started.task_id` IS the agent id the
   *  SDK's own task registry uses, per the adjoining `liveBackgroundTasks`
   *  doc in acp-agent.ts). */
  subagents: Map<string, ActivitySubagentRecord>;
  /** Spawning tool_use id -> task id, recorded at `task_started`. Lets an
   *  errored Agent/Task tool_result that carries NO structured `AgentOutput`
   *  (a hard SDK-level failure) still resolve which roster entry to mark
   *  failed, since its own `tool_use_id` is all such a result carries. */
  subagentToolUseIndex: Map<string, string>;
};

export function newActivityState(): ActivityState {
  return { processes: new Map(), subagents: new Map(), subagentToolUseIndex: new Map() };
}

export function processWire(record: ActivityProcessRecord): ActivityProcessWire {
  return {
    id: record.id,
    command: record.command,
    cwd: record.cwd,
    status: record.status,
    exitCode: record.exitCode,
    pid: record.pid,
    startedAtMs: record.startedAtMs,
    endedAtMs: record.endedAtMs,
    feed: record.feed,
  };
}

export function subagentWire(record: ActivitySubagentRecord): ActivitySubagentWire {
  return {
    id: record.id,
    agentType: record.agentType,
    description: record.description,
    model: record.model,
    background: record.background,
    status: record.status,
    summary: record.summary,
    tokensUsed: record.tokensUsed,
    toolCalls: record.toolCalls,
    durationSeconds: record.durationSeconds,
    feed: record.feed,
  };
}

/** Merge-only-defined-fields upsert: an `undefined` field in `patch` leaves the
 *  existing value untouched (so a later, partial event — e.g. `task_progress`
 *  carrying no `feed` — can never blank out a field an earlier event set). */
function upsertRecord<V extends { updatedAtMs: number }>(
  map: Map<string, V>,
  key: string,
  init: () => V,
  patch: Partial<V>,
): V {
  const existing = map.get(key);
  const merged: V = existing ? { ...existing } : init();
  for (const field of Object.keys(patch) as (keyof V)[]) {
    const value = patch[field];
    if (value !== undefined) {
      merged[field] = value as V[keyof V];
    }
  }
  merged.updatedAtMs = Date.now();
  map.set(key, merged);
  return merged;
}

export function upsertProcess(
  state: ActivityState,
  id: string,
  patch: Partial<ActivityProcessRecord>,
): ActivityProcessRecord {
  return upsertRecord(
    state.processes,
    id,
    () => ({
      id,
      command: "",
      cwd: null,
      status: "running",
      exitCode: null,
      pid: null,
      startedAtMs: Date.now(),
      endedAtMs: null,
      feed: null,
      updatedAtMs: Date.now(),
    }),
    patch,
  );
}

export function upsertSubagent(
  state: ActivityState,
  id: string,
  patch: Partial<ActivitySubagentRecord>,
): ActivitySubagentRecord {
  return upsertRecord(
    state.subagents,
    id,
    () => ({
      id,
      agentType: null,
      description: null,
      model: null,
      background: true,
      status: "running",
      summary: null,
      tokensUsed: null,
      toolCalls: null,
      durationSeconds: null,
      feed: null,
      updatedAtMs: Date.now(),
    }),
    patch,
  );
}

// ---------------------------------------------------------------------------
// Structured tool_use_result narrowing (mirrors tools.ts's private
// `structuredResult`, duplicated here — not exported there — to keep this
// module's footprint additive rather than reaching into tools.ts's internals).
// ---------------------------------------------------------------------------

export function structuredResult<T extends object>(toolUseResult: unknown): T | undefined {
  return toolUseResult !== null &&
    typeof toolUseResult === "object" &&
    !Array.isArray(toolUseResult)
    ? (toolUseResult as T)
    : undefined;
}

// ---------------------------------------------------------------------------
// Background-bash output-file materialization
// ---------------------------------------------------------------------------

/**
 * Parses the live output-file path out of a backgrounded Bash tool_result's
 * RAW text content. Live pitfall carried over from the old fork's fix 3fba2f5
 * (still present on this base, for the same underlying reason): the
 * structured `BashOutput` (message-level tool_use_result) for a backgrounded
 * command has EMPTY stdout/stderr — only `backgroundTaskId` is populated —
 * because the human-readable "Command running in background with ID: <id>.
 * Output is being written to: <path>" notice is a RAW-TEXT-ONLY artifact of
 * the tool_result (see tools.ts's `structuredBash.backgroundTaskId ===
 * undefined` gate, which documents the identical split for its own
 * rendering). There is no structured field carrying this path anywhere in the
 * SDK's task lifecycle messages (`task_started`/`task_progress` never carry
 * one; only the terminal `task_notification.output_file` does, well after the
 * process is no longer live) — so this text parse is the ONLY way to get a
 * live-tail feed at launch time, not a legacy fallback.
 */
export function parseBackgroundOutputFile(toolResultContent: unknown): string | null {
  const text = extractText(toolResultContent);
  if (!text) {
    return null;
  }
  const patterns = [
    // "…Output is being written to: <path>" / "…being logged to <path>".
    /(?:written|writing|logged|logging)\s+to:?\s*(\S+)/i,
    // "logs to /path", "logging to /path".
    /logs?\s+to:?\s*(\S+)/i,
    // Legacy forms: "output → /path", "output: /path", "output file: /path".
    /output(?:\s+file)?\s*(?:→|->|:)\s*(\S+)/i,
  ];
  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match?.[1]) {
      return match[1].replace(/[.,)]+$/, "");
    }
  }
  return null;
}

function extractText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((block) =>
        block && typeof block === "object" && typeof (block as { text?: unknown }).text === "string"
          ? (block as { text: string }).text
          : "",
      )
      .join("\n");
  }
  if (content && typeof content === "object") {
    const nested = (content as { content?: unknown }).content;
    if (typeof nested === "string") {
      return nested;
    }
    if (Array.isArray(nested)) {
      return extractText(nested);
    }
  }
  return "";
}

// ---------------------------------------------------------------------------
// `_anyharness/activity/list` ext method (attach-time roster reconcile pull)
// ---------------------------------------------------------------------------

export type ActivityListRequest = { sessionId: string };
export type ActivityListResponse = {
  processes: ActivityProcessWire[];
  subagents: ActivitySubagentWire[];
};

export function parseActivityListRequest(params: unknown): ActivityListRequest {
  if (!params || typeof params !== "object") {
    throw RequestError.invalidParams(undefined, "activity/list params must be an object");
  }
  const { sessionId } = params as Record<string, unknown>;
  if (typeof sessionId !== "string" || sessionId.length === 0) {
    throw RequestError.invalidParams(
      undefined,
      "activity/list params require a non-empty sessionId",
    );
  }
  return { sessionId };
}
