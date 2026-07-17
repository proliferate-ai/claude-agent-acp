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
  "goal_updated" | "goal_cleared" | "goal_met" | "loop_updated" | "loop_fired" | "loop_cleared";

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

/** Per-session goal/loop bookkeeping, attached to the ACP Session. */
export type AnyharnessSessionState = {
  goal: GoalState | null;
  /** Watchers invoked with every goal_status transcript row. */
  goalRowWatchers: ((row: GoalStatusRow) => void)[];
  loops: Map<string, LoopState>;
  /** loop/set calls whose CronCreate tool_use hasn't been observed yet. */
  pendingLoopSets: PendingLoopSet[];
  /** Watchers resolved whenever a loop transitions to cleared. */
  loopClearWatchers: (() => void)[];
  /** UUIDs of user messages injected by the GoalPort/LoopPort bridge. */
  injectedUuids: Set<string>;
  transcriptPath: string | null;
  tailer: TranscriptTailer | null;
  /** Fresh sessions tail from byte 0; resumed/forked sessions start at EOF. */
  tailFromStart: boolean;
};

export function newAnyharnessSessionState(tailFromStart: boolean): AnyharnessSessionState {
  return {
    goal: null,
    goalRowWatchers: [],
    loops: new Map(),
    pendingLoopSets: [],
    loopClearWatchers: [],
    injectedUuids: new Set(),
    transcriptPath: null,
    tailer: null,
    tailFromStart,
  };
}

export function goalWireFromState(state: GoalState): GoalWire {
  return {
    objective: state.objective,
    status: state.status,
    nativeStatus: state.nativeStatus,
    tokenBudget: null,
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

/** Shape of a `goal_status` attachment persisted in the session transcript. */
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

/** Classify native sentinel and evaluator rows from Claude's transcript. */
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

/** Liberally extract a goal_status attachment across known transcript shapes. */
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

/** Fallback transcript path when no hook has reported transcript_path yet. */
export function computeTranscriptPath(configDir: string, cwd: string, sessionId: string): string {
  const munged = cwd.replace(/[^a-zA-Z0-9]/g, "-");
  return path.join(configDir, "projects", munged, `${sessionId}.jsonl`);
}

/** Read the last goal_status row already present in a transcript. */
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
    try {
      const goalStatus = extractGoalStatus(JSON.parse(trimmed));
      if (goalStatus) {
        last = goalStatus;
      }
    } catch {
      // Ignore malformed/incomplete historical lines.
    }
  }
  return last;
}

const TAIL_DEBOUNCE_MS = 50;
const TAIL_POLL_INTERVAL_MS = 750;

/** Tail a transcript JSONL file, parsing only appended complete lines. */
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
        try {
          this.onRow(JSON.parse(trimmed));
        } catch (error) {
          this.logger.error("[anyharness] transcript row handler failed:", error);
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
