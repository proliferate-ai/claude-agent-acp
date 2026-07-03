// GoalPort / LoopPort support for anyharness (wire contract v1, pinned 2026-07-02).
//
// This module holds the normalized wire shapes (GoalWire / LoopWire), the
// per-session goal/loop state, and the transcript tailer used to observe
// native goal evaluations (`attachment.goal_status` rows in the session
// transcript .jsonl).

import * as fs from "node:fs";
import * as path from "node:path";
import { Logger } from "./acp-agent.js";

export const ANYHARNESS_SCHEMA_VERSION = 1;

/** Capability blob advertised on InitializeResponse._meta.anyharness. */
export const ANYHARNESS_CAPABILITIES = {
  schemaVersion: ANYHARNESS_SCHEMA_VERSION,
  goals: { supported: true, native: true },
  loops: { supported: true, native: true },
} as const;

export type AnyharnessTranscriptEvent =
  | "goal_updated"
  | "goal_cleared"
  | "goal_met"
  | "loop_upserted"
  | "loop_fired"
  | "loop_removed"
  | "process_upserted"
  | "subagent_upserted";

export type GoalWireStatus = "active" | "paused" | "blocked" | "met" | "failed" | "cleared";

export type GoalWire = {
  objective: string;
  status: GoalWireStatus;
  nativeStatus: string;
  tokenBudget: number | null;
  tokensUsed: number | null;
  timeUsedSeconds: number | null;
  metReason: string | null;
  iterations: number | null;
  native: boolean;
  updatedAtMs: number;
};

export type LoopSchedule = { kind: "interval" | "cron"; expr: string };

export type LoopWire = {
  loopId: string;
  prompt: string;
  schedule: LoopSchedule;
  recurring: boolean;
  status: "active" | "cleared";
  native: boolean;
  lastFiredAtMs: number | null;
  fireCount: number;
  updatedAtMs: number;
};

export type GoalState = {
  objective: string;
  status: GoalWireStatus;
  nativeStatus: string;
  metReason: string | null;
  iterations: number | null;
  tokensUsed: number | null;
  timeUsedSeconds: number | null;
  updatedAtMs: number;
};

export type LoopState = {
  loopId: string;
  prompt: string;
  schedule: LoopSchedule;
  recurring: boolean;
  status: "active" | "cleared";
  lastFiredAtMs: number | null;
  fireCount: number;
  updatedAtMs: number;
};

export type PendingLoopSet = {
  prompt: string;
  schedule: LoopSchedule;
  recurring: boolean;
  requestedAtMs: number;
  resolve: (loop: LoopState) => void;
};

// ---------------------------------------------------------------------------
// Read-only rosters (background processes + subagents).
//
// These mirror the Claude task machinery (task_started/task_progress/
// task_notification system events). They are never writable; each carries a
// FeedTransport describing the membrane→runtime byte transport (the runtime
// swaps it for an opaque FeedRef before it leaves the boundary — the client
// never learns the path).
// ---------------------------------------------------------------------------

/** Membrane-side byte transport for a roster element's live content stream. */
export type FeedTransport = { transport: "tail_file"; path: string };

export type ProcessStatus = "running" | "exited";

/** Normalized wire record for a background process (Claude `local_bash` task). */
export type ProcessWire = {
  id: string; // claude task_id
  command: string;
  cwd: string | null;
  status: ProcessStatus;
  exitCode: number | null; // claude does not report an exit code — null
  pid: number | null; // claude does not report a pid — null
  startedAtMs: number;
  endedAtMs: number | null;
  feed: FeedTransport | null; // tail_file(output_file) once known
  updatedAtMs: number;
};

/** Internal process record: the wire shape + the correlating tool_use id. */
export type ProcessState = ProcessWire & {
  /** tool_use id of the spawning Bash call (correlates the output-file tool_result). */
  toolUseId: string | null;
};

export type SubagentStatus = "running" | "completed" | "failed";

/** Normalized wire record for a subagent (Claude `local_agent` task). */
export type SubagentWire = {
  id: string; // claude task_id (doubles as SendMessage agent id)
  agentType: string | null; // subagent_type
  description: string | null;
  prompt: string | null;
  model: string | null;
  background: boolean;
  status: SubagentStatus;
  summary: string | null;
  // Usage as FLAT sibling fields (from task_progress / task_notification),
  // matching anyharness's ActivitySubagentWire contract. Nesting these under a
  // `usage` object (or naming/uniting them differently) makes the runtime read
  // them as absent, so the roster UI shows no usage. Seconds, not milliseconds.
  tokensUsed: number | null;
  toolCalls: number | null;
  durationSeconds: number | null;
  feed: FeedTransport | null; // tail_file(<parent>/subagents/agent-<id>.jsonl)
  updatedAtMs: number;
};

export type SubagentState = SubagentWire;

/** Strips the internal correlation field before a process record leaves the membrane. */
export function processWireFromState(state: ProcessState): ProcessWire {
  return {
    id: state.id,
    command: state.command,
    cwd: state.cwd,
    status: state.status,
    exitCode: state.exitCode,
    pid: state.pid,
    startedAtMs: state.startedAtMs,
    endedAtMs: state.endedAtMs,
    feed: state.feed,
    updatedAtMs: state.updatedAtMs,
  };
}

/**
 * An injection (a `/goal …` / `/loop …` user message we push) that is held
 * until the session reaches a turn boundary. `/goal` and `/loop` are local
 * commands that silently degrade to never-executing queued prompts when they
 * arrive while a turn is streaming, so we queue them and inject at idle. The
 * deferred goal/loop set methods return a provisional response immediately and
 * carry NO fork-side confirmation wait — the mirror reconciles from the later
 * transcript sentinel (goal) or CronCreate (loop) notification.
 */
export type DeferredInjection = {
  uuid: string;
  text: string;
};

/** Per-session goal/loop bookkeeping, attached to the ACP Session. */
export type AnyharnessSessionState = {
  goal: GoalState | null;
  /** Watchers invoked with every goal_status transcript row (used by goal/set + goal/clear to await native confirmation). */
  goalRowWatchers: ((row: GoalStatusRow) => void)[];
  loops: Map<string, LoopState>;
  /** loop/set calls whose CronCreate tool_use hasn't been observed yet. */
  pendingLoopSets: PendingLoopSet[];
  /** Watchers resolved whenever a loop transitions to cleared (used by loop/clear). */
  loopClearWatchers: (() => void)[];
  /** Read-only roster: background processes (local_bash tasks), keyed by task id. */
  processes: Map<string, ProcessState>;
  /** Read-only roster: subagents (local_agent tasks), keyed by task id. */
  subagents: Map<string, SubagentState>;
  /** tool_use id → captured Bash command / background output file, correlating task_started to its spawning tool call. */
  taskToolUse: Map<string, { command?: string; outputFile?: string }>;
  /** uuids of user messages we injected (goal/loop instructions) — the pump must not treat their turns as cron wakes. */
  injectedUuids: Set<string>;
  /** Injections queued while a turn is streaming; flushed at the next idle boundary. */
  deferredInjections: DeferredInjection[];
  transcriptPath: string | null;
  tailer: TranscriptTailer | null;
  /** Whether the tailer should read the transcript from byte 0 (fresh session) or from EOF (resume/fork). */
  tailFromStart: boolean;
  pumpRunning: boolean;
  /** Resolver that makes an idle-blocked pump yield the message stream to a queued prompt(). */
  pumpInterrupt: (() => void) | null;
  /**
   * True while a drain is actively inside a turn (between its first message and
   * the idle boundary). Deferred injections wait for this to fall so a `/goal`
   * local command never degrades to a queued mid-turn prompt.
   */
  turnActive: boolean;
};

export function newAnyharnessSessionState(tailFromStart: boolean): AnyharnessSessionState {
  return {
    goal: null,
    goalRowWatchers: [],
    loops: new Map(),
    pendingLoopSets: [],
    loopClearWatchers: [],
    processes: new Map(),
    subagents: new Map(),
    taskToolUse: new Map(),
    injectedUuids: new Set(),
    deferredInjections: [],
    transcriptPath: null,
    tailer: null,
    tailFromStart,
    pumpRunning: false,
    pumpInterrupt: null,
    turnActive: false,
  };
}

export function goalWireFromState(state: GoalState): GoalWire {
  return {
    objective: state.objective,
    status: state.status,
    nativeStatus: state.nativeStatus,
    tokenBudget: null, // claude's native /goal has no token budget
    tokensUsed: state.tokensUsed,
    timeUsedSeconds: state.timeUsedSeconds,
    metReason: state.metReason,
    iterations: state.iterations,
    native: true,
    updatedAtMs: state.updatedAtMs,
  };
}

export function loopWireFromState(state: LoopState): LoopWire {
  return {
    loopId: state.loopId,
    prompt: state.prompt,
    schedule: state.schedule,
    recurring: state.recurring,
    status: state.status,
    native: true,
    lastFiredAtMs: state.lastFiredAtMs,
    fireCount: state.fireCount,
    updatedAtMs: state.updatedAtMs,
  };
}

export function activeLoops(state: AnyharnessSessionState): LoopState[] {
  return [...state.loops.values()].filter((loop) => loop.status === "active");
}

/**
 * Attributes a spontaneous cron-wake turn to a specific armed loop by the
 * user-prompt the wake replayed. A native cron wake re-injects the loop's exact
 * prompt as a user message (and re-emits system:init); that replay is the ONLY
 * reliable fire signal. A bare spontaneous assistant turn — a goal continuation,
 * a background-task wake, or an ambiguous multi-loop wake — carries no such
 * message and must never be credited as a fire.
 *
 * Prefers an exact prompt match; falls back to a containment match only when it
 * is unambiguous (exactly one loop), so prompts that overlap as substrings
 * across loops can't credit the wrong loop. Returns undefined when zero or more
 * than one loop matches — better to under-count than to move the wrong loop's
 * fire bookkeeping.
 */
export function matchLoopForWake(loops: LoopState[], userText: string): LoopState | undefined {
  const exact = loops.filter((loop) => loop.prompt === userText);
  if (exact.length === 1) {
    return exact[0];
  }
  if (exact.length > 1) {
    return undefined; // multiple loops share this exact prompt — ambiguous
  }
  const contained = loops.filter(
    (loop) => userText.includes(loop.prompt) || loop.prompt.includes(userText),
  );
  return contained.length === 1 ? contained[0] : undefined;
}

/**
 * A loop id we synthesized ourselves (CronCreate returned no id, or loop/set
 * timed out) rather than a real harness cron id. These are replaced by a real
 * id when a session_crons snapshot reveals one for the same prompt.
 */
export function isSyntheticLoopId(loopId: string): boolean {
  return loopId.startsWith("provisional-") || loopId.startsWith("cron-");
}

/**
 * Coerces a tool_response (string | content-blocks | object) into a searchable
 * string. Claude's cron tools return prose results, so ids/paths must be parsed
 * out of text rather than read from structured fields.
 */
function toolResponseText(toolResponse: unknown): string {
  if (typeof toolResponse === "string") {
    return toolResponse;
  }
  if (Array.isArray(toolResponse)) {
    return toolResponse
      .map((block) =>
        typeof block === "object" && block !== null && typeof (block as any).text === "string"
          ? (block as any).text
          : "",
      )
      .join("\n");
  }
  if (typeof toolResponse === "object" && toolResponse !== null) {
    const content = (toolResponse as { content?: unknown }).content;
    if (typeof content === "string") {
      return content;
    }
    if (Array.isArray(content)) {
      return toolResponseText(content);
    }
    try {
      return JSON.stringify(toolResponse);
    } catch {
      return "";
    }
  }
  return "";
}

/**
 * Parses the cron job id out of a CronCreate/ScheduleWakeup result string.
 * Live-verified against Claude Code 2.1.199: the result is prose, e.g.
 * "Scheduled recurring job dad38e14 (Every minute). Session-only …" — the id
 * is NOT a structured field, so extractCronId can't see it. Falls back to null.
 */
export function parseCronIdFromResult(toolResponse: unknown): string | null {
  const text = toolResponseText(toolResponse);
  if (!text) {
    return null;
  }
  const match = text.match(/\bjob\s+([A-Za-z0-9_-]{4,})/i);
  return match?.[1] ?? null;
}

/** Best-effort extraction of a cron job id from CronCreate/CronDelete/session_crons IO. */
export function extractCronId(value: unknown): string | undefined {
  if (typeof value !== "object" || value === null) {
    return undefined;
  }
  const record = value as Record<string, unknown>;
  for (const key of ["id", "jobId", "cronId", "job_id", "cron_id"]) {
    const candidate = record[key];
    if (typeof candidate === "string" && candidate) {
      return candidate;
    }
    if (typeof candidate === "number") {
      return String(candidate);
    }
  }
  for (const key of ["job", "cron", "result", "output", "structuredContent"]) {
    const found = extractCronId(record[key]);
    if (found) {
      return found;
    }
  }
  return undefined;
}

/** A single armed cron as it appears in a hook payload's `session_crons` snapshot. */
export type SessionCronSnapshotEntry = {
  id?: string | number;
  jobId?: string | number;
  cron?: string;
  schedule?: string;
  prompt?: string;
  recurring?: boolean;
  lastFiredAt?: string | number;
  fireCount?: number;
};

export type SessionCronReconcile = {
  upserted: LoopState[];
  removed: string[];
};

/**
 * Reconciles the loop mirror against a `session_crons` snapshot from a hook
 * payload. session_crons is a free, authoritative reconciliation snapshot the
 * harness hands us on every hook firing (harness-runtime-mechanics §3): it lets
 * the mirror heal drift (a cron deleted or armed by a bare TUI against the same
 * session) without a model-costing CronList.
 *
 * Rules:
 * - Every cron in the snapshot is upserted as an active loop under its real id,
 *   preserving fire bookkeeping when the id (or, for a synthesized loop, the
 *   prompt) already matched.
 * - An active loop with a *real* id absent from the snapshot was deleted
 *   externally — mark it removed. Synthetic-id loops are left untouched (their
 *   real cron may simply not be in this snapshot yet).
 */
export function reconcileSessionCrons(
  state: AnyharnessSessionState,
  crons: unknown,
  now: number,
): SessionCronReconcile {
  const result: SessionCronReconcile = { upserted: [], removed: [] };
  if (!Array.isArray(crons)) {
    return result;
  }
  const snapshotIds = new Set<string>();

  for (const raw of crons) {
    const id = extractCronId(raw);
    if (!id) {
      continue; // can't reconcile a cron we can't identify
    }
    snapshotIds.add(id);
    const entry = (raw ?? {}) as SessionCronSnapshotEntry;
    const prompt = typeof entry.prompt === "string" ? entry.prompt : "";
    const expr =
      typeof entry.cron === "string"
        ? entry.cron
        : typeof entry.schedule === "string"
          ? entry.schedule
          : "";
    const recurring = entry.recurring !== false;

    const existing = state.loops.get(id);
    if (existing && existing.status === "active") {
      // Update in place only when a field actually changed, so a steady-state
      // snapshot emits nothing.
      const changed =
        (prompt && existing.prompt !== prompt) ||
        (expr && existing.schedule.expr !== expr) ||
        existing.recurring !== recurring;
      if (changed) {
        existing.prompt = prompt || existing.prompt;
        existing.schedule = { kind: "cron", expr: expr || existing.schedule.expr };
        existing.recurring = recurring;
        existing.updatedAtMs = now;
        result.upserted.push(existing);
      }
      continue;
    }

    // A synthesized loop for the same prompt is the real cron under a placeholder
    // id — upgrade it to the real id, carrying its fire bookkeeping.
    const synth = activeLoops(state).find(
      (loop) => isSyntheticLoopId(loop.loopId) && prompt && loop.prompt === prompt,
    );
    const loop: LoopState = {
      loopId: id,
      prompt: prompt || synth?.prompt || "",
      schedule: { kind: "cron", expr: expr || synth?.schedule.expr || "" },
      recurring,
      status: "active",
      lastFiredAtMs: synth?.lastFiredAtMs ?? null,
      fireCount: synth?.fireCount ?? 0,
      updatedAtMs: now,
    };
    if (synth) {
      state.loops.delete(synth.loopId);
      result.removed.push(synth.loopId);
    }
    state.loops.set(id, loop);
    result.upserted.push(loop);
  }

  for (const loop of activeLoops(state)) {
    if (!isSyntheticLoopId(loop.loopId) && !snapshotIds.has(loop.loopId)) {
      loop.status = "cleared";
      loop.updatedAtMs = now;
      result.removed.push(loop.loopId);
    }
  }

  return result;
}

/**
 * The per-subagent transcript path for a `local_agent` task. Claude persists it
 * at `<project>/<parentSessionId>/subagents/agent-<taskId>.jsonl` (the task's
 * output_file is a symlink to it) — deriving it lets a live nested-transcript
 * feed open immediately at task_started, before any completion event.
 */
export function subagentFeedPath(
  parentTranscriptPath: string,
  sessionId: string,
  taskId: string,
): string {
  const projectDir = path.dirname(parentTranscriptPath);
  return path.join(projectDir, sessionId, "subagents", `agent-${taskId}.jsonl`);
}

/**
 * Parses the output-file path out of a background-bash tool_result body, so the
 * process feed can tail it live rather than waiting for the completion
 * notification. Live-verified against Claude Code 2.1.199, whose real
 * background-Bash result is:
 *   "Command running in background with ID: <id>. Output is being written to: <path>"
 * The old `/output\s*(?:→|->|:)/` pattern did NOT match "Output is being written
 * to:" (there is no colon/arrow directly after "output"), so the output file was
 * never captured and the process roster element shipped with `feed: null` (gate
 * B "carries a live-tail FeedRef" FAIL). We now also match the "written to:" /
 * "logs to" phrasings while keeping the legacy "output → <path>" forms.
 */
export function parseBackgroundOutputFile(toolResult: unknown): string | null {
  const text =
    typeof toolResult === "string"
      ? toolResult
      : Array.isArray(toolResult)
        ? toolResult
            .map((block) =>
              typeof block === "object" && block !== null && typeof (block as any).text === "string"
                ? (block as any).text
                : "",
            )
            .join("\n")
        : typeof toolResult === "object" && toolResult !== null
          ? typeof (toolResult as any).content === "string"
            ? (toolResult as any).content
            : JSON.stringify(toolResult)
          : "";
  if (!text) {
    return null;
  }
  const patterns = [
    // Claude Code background Bash: "…Output is being written to: <path>".
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

/** Shape of a `goal_status` attachment persisted in the session transcript .jsonl. */
export type GoalStatusRow = {
  type: "goal_status";
  met?: boolean;
  sentinel?: boolean;
  failed?: boolean;
  condition?: string;
  reason?: string;
  iterations?: number;
  durationMs?: number;
  tokens?: number;
};

function isGoalStatusRow(value: unknown): value is GoalStatusRow {
  return (
    typeof value === "object" &&
    value !== null &&
    (value as { type?: unknown }).type === "goal_status"
  );
}

export type GoalRowKind = "armed" | "cleared" | "met" | "failed" | "progress";

/**
 * Classifies a goal_status transcript row (live-verified against Claude Code
 * 2.1.198): sentinel rows are zero-evaluation bookkeeping markers — arm
 * writes {met:false, sentinel:true, condition}, "/goal clear" writes
 * {met:true, sentinel:true, condition}. Evaluator rows carry no sentinel:
 * {met, condition, reason} (met:true auto-clears the native goal).
 */
export function classifyGoalStatus(row: GoalStatusRow): GoalRowKind {
  if (row.sentinel) {
    return row.met ? "cleared" : "armed";
  }
  if (row.met) {
    return "met";
  }
  if (row.failed) {
    return "failed";
  }
  return "progress";
}

/**
 * Liberal extraction of a goal_status attachment from one parsed transcript
 * row. Transcript rows are `{type: "attachment", attachment: {...}}`, but we
 * also probe a couple of plausible nestings to stay robust across CLI
 * versions.
 */
export function extractGoalStatus(row: unknown): GoalStatusRow | null {
  if (typeof row !== "object" || row === null) {
    return null;
  }
  const candidate = row as {
    type?: unknown;
    attachment?: unknown;
    message?: { attachment?: unknown };
  };
  if (isGoalStatusRow(candidate.attachment)) {
    return candidate.attachment;
  }
  if (candidate.message && isGoalStatusRow(candidate.message.attachment)) {
    return candidate.message.attachment;
  }
  if (isGoalStatusRow(candidate)) {
    return candidate;
  }
  return null;
}

/**
 * Fallback transcript location when no hook has reported transcript_path yet:
 * <configDir>/projects/<munged cwd>/<sessionId>.jsonl
 */
export function computeTranscriptPath(configDir: string, cwd: string, sessionId: string): string {
  const munged = cwd.replace(/[^a-zA-Z0-9]/g, "-");
  return path.join(configDir, "projects", munged, `${sessionId}.jsonl`);
}

/**
 * Reads the last goal_status row already present in a transcript. Used to
 * seed the goal mirror on resume/fork (native goals survive `--resume` but
 * the tailer only reads appended rows for those sessions).
 */
export function readLastGoalStatus(filePath: string): GoalStatusRow | null {
  let text: string;
  try {
    text = fs.readFileSync(filePath, "utf8");
  } catch {
    return null;
  }
  let last: GoalStatusRow | null = null;
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || !trimmed.includes('"goal_status"')) {
      continue;
    }
    let row: unknown;
    try {
      row = JSON.parse(trimmed);
    } catch {
      continue;
    }
    const goalStatus = extractGoalStatus(row);
    if (goalStatus) {
      last = goalStatus;
    }
  }
  return last;
}

const TAIL_DEBOUNCE_MS = 50;
const TAIL_POLL_INTERVAL_MS = 750;

/**
 * Tails a session transcript .jsonl, parsing only appended complete lines.
 * Uses fs.watch on the containing directory (modeled on
 * SettingsManager.setupWatchers) plus a polling fallback since the file may
 * not exist yet when the tailer starts.
 */
export class TranscriptTailer {
  private offset = 0;
  private partialLine = "";
  private watcher: fs.FSWatcher | null = null;
  private pollTimer: ReturnType<typeof setInterval> | null = null;
  private debounceTimer: ReturnType<typeof setTimeout> | null = null;
  private disposed = false;
  private reading = false;

  constructor(
    private readonly filePath: string,
    private readonly onRow: (row: unknown) => void,
    private readonly logger: Logger,
    private readonly options: { fromStart?: boolean } = {},
  ) {}

  start(): void {
    try {
      this.offset =
        this.options.fromStart || !fs.existsSync(this.filePath)
          ? 0
          : fs.statSync(this.filePath).size;
    } catch {
      this.offset = 0;
    }

    try {
      const dir = path.dirname(this.filePath);
      const filename = path.basename(this.filePath);
      if (fs.existsSync(dir)) {
        this.watcher = fs.watch(dir, (_eventType, changedFilename) => {
          if (changedFilename === filename) {
            this.scheduleRead();
          }
        });
        this.watcher.on("error", (error) => {
          this.logger.error(`[anyharness] transcript watcher error for ${this.filePath}:`, error);
        });
      }
    } catch (error) {
      this.logger.error(`[anyharness] failed to watch transcript ${this.filePath}:`, error);
    }

    // Polling fallback: covers the file (or its directory) not existing yet
    // and platforms where directory watch misses in-place appends.
    this.pollTimer = setInterval(() => this.readAppended(), TAIL_POLL_INTERVAL_MS);
    this.pollTimer.unref?.();

    this.scheduleRead();
  }

  private scheduleRead(): void {
    if (this.disposed) {
      return;
    }
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }
    this.debounceTimer = setTimeout(() => {
      this.debounceTimer = null;
      this.readAppended();
    }, TAIL_DEBOUNCE_MS);
  }

  private readAppended(): void {
    if (this.disposed || this.reading) {
      return;
    }
    this.reading = true;
    try {
      if (!fs.existsSync(this.filePath)) {
        return;
      }
      const size = fs.statSync(this.filePath).size;
      if (size < this.offset) {
        // Truncated/rotated — restart from the beginning of the new content.
        this.offset = 0;
        this.partialLine = "";
      }
      if (size === this.offset) {
        return;
      }
      const fd = fs.openSync(this.filePath, "r");
      let text: string;
      try {
        const length = size - this.offset;
        const buffer = Buffer.alloc(length);
        const bytesRead = fs.readSync(fd, buffer, 0, length, this.offset);
        this.offset += bytesRead;
        text = this.partialLine + buffer.toString("utf8", 0, bytesRead);
      } finally {
        fs.closeSync(fd);
      }
      const lines = text.split("\n");
      this.partialLine = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) {
          continue;
        }
        let row: unknown;
        try {
          row = JSON.parse(trimmed);
        } catch {
          continue; // not JSON — skip
        }
        try {
          this.onRow(row);
        } catch (error) {
          this.logger.error(`[anyharness] transcript row handler failed:`, error);
        }
      }
    } catch (error) {
      this.logger.error(`[anyharness] transcript tail read failed for ${this.filePath}:`, error);
    } finally {
      this.reading = false;
    }
  }

  dispose(): void {
    this.disposed = true;
    if (this.watcher) {
      this.watcher.close();
      this.watcher = null;
    }
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
  }
}
