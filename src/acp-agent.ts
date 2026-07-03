import {
  Agent,
  AgentSideConnection,
  AuthenticateRequest,
  AuthMethod,
  AvailableCommand,
  CancelNotification,
  ClientCapabilities,
  ForkSessionRequest,
  ForkSessionResponse,
  InitializeRequest,
  InitializeResponse,
  ListSessionsRequest,
  ListSessionsResponse,
  LoadSessionRequest,
  LoadSessionResponse,
  ndJsonStream,
  NewSessionRequest,
  NewSessionResponse,
  PromptRequest,
  PromptResponse,
  ReadTextFileRequest,
  ReadTextFileResponse,
  RequestError,
  ResumeSessionRequest,
  ResumeSessionResponse,
  SessionConfigOption,
  SessionModelState,
  SessionModeState,
  SessionNotification,
  SetSessionConfigOptionRequest,
  SetSessionConfigOptionResponse,
  SetSessionModelRequest,
  SetSessionModelResponse,
  SetSessionModeRequest,
  SetSessionModeResponse,
  CloseSessionRequest,
  CloseSessionResponse,
  TerminalHandle,
  TerminalOutputResponse,
  WriteTextFileRequest,
  WriteTextFileResponse,
  StopReason,
} from "@agentclientprotocol/sdk";
import {
  CanUseTool,
  getSessionMessages,
  listSessions,
  McpServerConfig,
  ModelInfo,
  Options,
  PermissionMode,
  Query,
  query,
  SDKMessage,
  SDKPartialAssistantMessage,
  SDKResultMessage,
  Settings,
  SDKUserMessage,
  SlashCommand,
} from "@anthropic-ai/claude-agent-sdk";
import { ContentBlockParam } from "@anthropic-ai/sdk/resources";
import { BetaContentBlock, BetaRawContentBlockDelta } from "@anthropic-ai/sdk/resources/beta.mjs";
import { randomUUID } from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import packageJson from "../package.json" with { type: "json" };
import {
  activeLoops,
  ANYHARNESS_CAPABILITIES,
  ANYHARNESS_SCHEMA_VERSION,
  AnyharnessSessionState,
  AnyharnessTranscriptEvent,
  classifyGoalStatus,
  computeTranscriptPath,
  extractCronFirePrompt,
  extractCronId,
  extractGoalStatus,
  GoalStatusRow,
  GoalWire,
  goalWireFromState,
  isSyntheticLoopId,
  LoopSchedule,
  LoopState,
  LoopWire,
  loopWireFromState,
  matchLoopForWake,
  newAnyharnessSessionState,
  parseBackgroundOutputFile,
  parseCronIdFromResult,
  ProcessState,
  ProcessWire,
  processWireFromState,
  readLastGoalStatus,
  reconcileSessionCrons,
  subagentFeedPath,
  SubagentState,
  SubagentWire,
  TranscriptTailer,
} from "./anyharness.js";
import { SettingsManager } from "./settings.js";
import {
  ClaudePlanEntry,
  createPostToolUseHook,
  planEntries,
  registerHookCallback,
  toolInfoFromToolUse,
  toolUpdateFromEditToolResponse,
  toolUpdateFromToolResult,
} from "./tools.js";
import { nodeToWebReadable, nodeToWebWritable, Pushable, sleep, unreachable } from "./utils.js";

export const CLAUDE_CONFIG_DIR =
  process.env.CLAUDE_CONFIG_DIR ?? path.join(os.homedir(), ".claude");

const MAX_TITLE_LENGTH = 256;

function sanitizeTitle(text: string): string {
  // Replace newlines and collapse whitespace
  const sanitized = text
    .replace(/[\r\n]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (sanitized.length <= MAX_TITLE_LENGTH) {
    return sanitized;
  }
  return sanitized.slice(0, MAX_TITLE_LENGTH - 1) + "…";
}

/**
 * Logger interface for customizing logging output
 */
export interface Logger {
  log: (...args: any[]) => void;
  error: (...args: any[]) => void;
}

type AccumulatedUsage = {
  inputTokens: number;
  outputTokens: number;
  cachedReadTokens: number;
  cachedWriteTokens: number;
};

type SupportedEffortLevel = "low" | "medium" | "high" | "max";
type LiveEffortLevel = Exclude<SupportedEffortLevel, "max">;

type ModelCapabilities = {
  supportsEffort: boolean;
  supportedEffortLevels: SupportedEffortLevel[];
  supportsAdaptiveThinking: boolean;
  supportsFastMode: boolean;
};

type ModelCapabilitiesById = Record<string, ModelCapabilities>;

type AvailableModelsResult = {
  state: SessionModelState;
  capabilitiesById: ModelCapabilitiesById;
};

type MutableQuery = Query & {
  applyFlagSettings(settings: Partial<Settings>): Promise<void>;
  getSettings(): Promise<Settings>;
};

type Session = {
  query: MutableQuery;
  input: Pushable<SDKUserMessage>;
  cancelled: boolean;
  cwd: string;
  settingsManager: SettingsManager;
  accumulatedUsage: AccumulatedUsage;
  modes: SessionModeState;
  models: SessionModelState;
  modelCapabilitiesById: ModelCapabilitiesById;
  liveSettings: Settings;
  configOptions: SessionConfigOption[];
  promptRunning: boolean;
  pendingMessages: Map<string, { resolve: (cancelled: boolean) => void; order: number }>;
  nextPendingOrder: number;
  abortController: AbortController;
  /** Goal/loop state + transcript tailer for the anyharness GoalPort/LoopPort extension. */
  anyharness: AnyharnessSessionState;
  /**
   * In-flight query.next() shared between the prompt drain and the idle
   * pump so an interrupted pump never loses a pulled message.
   */
  pendingQueryNext?: Promise<IteratorResult<SDKMessage, void>> | null;
};

type LegacySessionStateChangedMessage = {
  type: "system";
  subtype: "session_state_changed";
  state: string;
  session_id: string;
};

type LegacyApiRetryMessage = {
  type: "system";
  subtype: "api_retry";
  session_id: string;
};

function isLegacySessionStateChangedMessage(message: {
  type: "system";
  subtype: string;
}): message is LegacySessionStateChangedMessage {
  return message.subtype === "session_state_changed";
}

function isLegacyApiRetryMessage(message: {
  type: "system";
  subtype: string;
}): message is LegacyApiRetryMessage {
  return message.subtype === "api_retry";
}

type BackgroundTerminal =
  | {
      handle: TerminalHandle;
      status: "started";
      lastOutput: TerminalOutputResponse | null;
    }
  | {
      status: "aborted" | "exited" | "killed" | "timedOut";
      pendingOutput: TerminalOutputResponse;
    };

/**
 * Extra metadata that can be given when creating a new session.
 */
export type NewSessionMeta = {
  claudeCode?: {
    /**
     * Options forwarded to Claude Code when starting a new session.
     * Those parameters will be ignored and managed by ACP:
     *   - cwd
     *   - includePartialMessages
     *   - allowDangerouslySkipPermissions
     *   - permissionMode
     *   - canUseTool
     *   - executable
     * Those parameters will be used and updated to work with ACP:
     *   - hooks (merged with ACP's hooks)
     *   - mcpServers (merged with ACP's mcpServers)
     *   - disallowedTools (merged with ACP's disallowedTools)
     *   - tools (passed through; defaults to claude_code preset if not provided)
     */
    options?: Options;
  };
  additionalRoots?: string[];
};

/**
 * Extra metadata for 'gateway' authentication requests.
 */
type GatewayAuthMeta = {
  /**
   * These parameters are mapped to environment variables to:
   * - Redirect API calls via baseUrl
   * - Inject custom headers
   * - Bypass the default Claude login requirement
   */
  gateway: {
    baseUrl: string;
    headers: Record<string, string>;
  };
};

/**
 * Extra metadata that the agent provides for each tool_call / tool_update update.
 */
export type ToolUpdateMeta = {
  claudeCode?: {
    /* The name of the tool that was used in Claude Code. */
    toolName: string;
    /* The structured output provided by Claude Code. */
    toolResponse?: unknown;
  };
  /* Terminal metadata for Bash tool execution, matching codex-acp's _meta protocol. */
  terminal_info?: {
    terminal_id: string;
  };
  terminal_output?: {
    terminal_id: string;
    data: string;
  };
  terminal_exit?: {
    terminal_id: string;
    exit_code: number;
    signal: string | null;
  };
};

export type ToolUseCache = {
  [key: string]: {
    type: "tool_use" | "server_tool_use" | "mcp_tool_use";
    id: string;
    name: string;
    input: unknown;
  };
};

function isStaticBinary(): boolean {
  return process.env.CLAUDE_AGENT_ACP_IS_SINGLE_FILE_BUN !== undefined;
}

export async function claudeCliPath(): Promise<string> {
  return isStaticBinary()
    ? (await import("@anthropic-ai/claude-agent-sdk/embed")).default
    : import.meta.resolve("@anthropic-ai/claude-agent-sdk").replace("sdk.mjs", "cli.js");
}

function shouldHideClaudeAuth(): boolean {
  return process.argv.includes("--hide-claude-auth");
}

// Bypass Permissions doesn't work if we are a root/sudo user
const IS_ROOT = (process.geteuid?.() ?? process.getuid?.()) === 0;
const ALLOW_BYPASS = !IS_ROOT || !!process.env.IS_SANDBOX;

// Slash commands that the SDK handles locally without replaying the user
// message and without invoking the model.
const LOCAL_ONLY_COMMANDS = new Set(["/context", "/heapdump", "/extra-usage"]);

// How long the goal/set and goal/clear ext methods wait for the native
// confirmation (the goal_status sentinel row in the transcript) before
// failing. Idle sessions confirm in well under a second; a turn in flight
// degrades /goal commands to queued prompt-mode commands that never execute.
const GOAL_SET_TIMEOUT_MS = 30_000;
const GOAL_CLEAR_TIMEOUT_MS = 30_000;

// How long the loop/set and loop/clear ext methods wait for the matching
// CronCreate/CronDelete tool_use before answering optimistically.
const LOOP_SET_TIMEOUT_MS = 60_000;
const LOOP_CLEAR_TIMEOUT_MS = 60_000;

function isProcessExitError(message: string): boolean {
  return (
    message.includes("ProcessTransport") ||
    message.includes("terminated process") ||
    message.includes("process exited with") ||
    message.includes("process terminated by signal") ||
    message.includes("Failed to write to process stdin")
  );
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}

/** Best-effort extraction of a crons array from a CronList tool response. */
function extractCronArray(value: unknown): unknown[] | null {
  if (Array.isArray(value)) {
    return value;
  }
  if (typeof value !== "object" || value === null) {
    return null;
  }
  const record = value as Record<string, unknown>;
  for (const key of [
    "crons",
    "jobs",
    "session_crons",
    "sessionCrons",
    "result",
    "output",
    "structuredContent",
    "content",
  ]) {
    const found = extractCronArray(record[key]);
    if (found) {
      return found;
    }
  }
  return null;
}

/**
 * A provisional GoalWire returned from goal/set while the "/goal" injection is
 * deferred behind a streaming turn. nativeStatus "pending_injection" flags the
 * queued (not-yet-armed) state; the mirror is untouched and the truthful
 * transition arrives later as a goal_updated notification.
 */
function pendingGoalWire(objective: string): GoalWire {
  return {
    objective,
    status: "active",
    nativeStatus: "pending_injection",
    tokenBudget: null,
    tokensUsed: null,
    timeUsedSeconds: null,
    metReason: null,
    iterations: null,
    native: true,
    updatedAtMs: Date.now(),
  };
}

/**
 * A provisional LoopWire returned from loop/set while the "/loop" injection is
 * deferred behind a streaming turn (mirrors pendingGoalWire). It carries a
 * synthetic "provisional-" id and status "active"; the mirror is left untouched
 * and the authoritative loop — with its real cron id — arrives later as a
 * loop_upserted notification when the deferred "/loop" runs and its CronCreate
 * is observed. The notification, not this response, is the source of truth.
 */
function pendingLoopWire(prompt: string, schedule: LoopSchedule, recurring: boolean): LoopWire {
  return {
    loopId: `provisional-${randomUUID().slice(0, 8)}`,
    prompt,
    schedule,
    recurring,
    status: "active",
    native: true,
    lastFiredAtMs: null,
    fireCount: 0,
    updatedAtMs: Date.now(),
  };
}

/** Extracts the text of a plain user message (used to attribute cron wakes). */
function userMessageText(content: unknown): string | undefined {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    const texts = content
      .filter(
        (block): block is { type: "text"; text: string } =>
          typeof block === "object" &&
          block !== null &&
          block.type === "text" &&
          typeof block.text === "string",
      )
      .map((block) => block.text);
    if (texts.length > 0) {
      return texts.join("\n");
    }
  }
  return undefined;
}

const PERMISSION_MODE_ALIASES: Record<string, PermissionMode> = {
  default: "default",
  acceptedits: "acceptEdits",
  dontask: "dontAsk",
  plan: "plan",
  bypasspermissions: "bypassPermissions",
  bypass: "bypassPermissions",
};

export function resolvePermissionMode(defaultMode?: unknown): PermissionMode {
  if (defaultMode === undefined) {
    return "default";
  }

  if (typeof defaultMode !== "string") {
    throw new Error("Invalid permissions.defaultMode: expected a string.");
  }

  const normalized = defaultMode.trim().toLowerCase();
  if (normalized === "") {
    throw new Error("Invalid permissions.defaultMode: expected a non-empty string.");
  }

  const mapped = PERMISSION_MODE_ALIASES[normalized];
  if (!mapped) {
    throw new Error(`Invalid permissions.defaultMode: ${defaultMode}.`);
  }

  if (mapped === "bypassPermissions" && !ALLOW_BYPASS) {
    throw new Error(
      "Invalid permissions.defaultMode: bypassPermissions is not available when running as root.",
    );
  }

  return mapped;
}

// Implement the ACP Agent interface
export class ClaudeAcpAgent implements Agent {
  sessions: {
    [key: string]: Session;
  };
  client: AgentSideConnection;
  toolUseCache: ToolUseCache;
  backgroundTerminals: { [key: string]: BackgroundTerminal } = {};
  clientCapabilities?: ClientCapabilities;
  logger: Logger;
  gatewayAuthMeta?: GatewayAuthMeta;
  /** transcript paths reported by hooks before their session is registered. */
  private pendingTranscriptPaths: Map<string, string> = new Map();

  constructor(client: AgentSideConnection, logger?: Logger) {
    this.sessions = {};
    this.client = client;
    this.toolUseCache = {};
    this.logger = logger ?? console;
  }

  async initialize(request: InitializeRequest): Promise<InitializeResponse> {
    this.clientCapabilities = request.clientCapabilities;

    // Bypasses standard auth by routing requests through a custom Anthropic-protocol gateway.
    // Only offered when the client advertises `auth._meta.gateway` capability.
    const supportsGatewayAuth = request.clientCapabilities?.auth?._meta?.gateway === true;

    const gatewayAuthMethod: AuthMethod = {
      id: "gateway",
      name: "Custom model gateway",
      description: "Use a custom gateway to authenticate and access models",
      _meta: {
        gateway: {
          protocol: "anthropic",
        },
      },
    };

    const terminalAuthMethod: any = {
      description: "Run `claude /login` in the terminal",
      name: "Log in with Claude",
      id: "claude-login",
      type: "terminal",
      args: ["--cli"],
    };
    const supportsTerminalAuth = request.clientCapabilities?.auth?.terminal === true;

    // If client supports terminal-auth capability, use that instead.
    const supportsMetaTerminalAuth = request.clientCapabilities?._meta?.["terminal-auth"] === true;
    if (supportsMetaTerminalAuth) {
      terminalAuthMethod._meta = {
        "terminal-auth": {
          command: process.execPath,
          args: [...process.argv.slice(1), "--cli"],
          label: "Claude Login",
        },
      };
    }

    return {
      protocolVersion: 1,
      // anyharness GoalPort/LoopPort capability advertisement (wire contract v1).
      _meta: {
        anyharness: ANYHARNESS_CAPABILITIES,
      },
      agentCapabilities: {
        _meta: {
          claudeCode: {
            promptQueueing: true,
          },
        },
        promptCapabilities: {
          image: true,
          embeddedContext: true,
        },
        mcpCapabilities: {
          http: true,
          sse: true,
        },
        loadSession: true,
        sessionCapabilities: {
          fork: {},
          list: {},
          resume: {},
          close: {},
        },
      },
      agentInfo: {
        name: packageJson.name,
        title: "Claude Agent",
        version: packageJson.version,
      },
      authMethods: [
        ...(!shouldHideClaudeAuth() && (supportsTerminalAuth || supportsMetaTerminalAuth)
          ? [terminalAuthMethod]
          : []),
        ...(supportsGatewayAuth ? [gatewayAuthMethod] : []),
      ],
    };
  }

  async newSession(params: NewSessionRequest): Promise<NewSessionResponse> {
    if (
      !this.gatewayAuthMeta &&
      fs.existsSync(path.resolve(os.homedir(), ".claude.json.backup")) &&
      !fs.existsSync(path.resolve(os.homedir(), ".claude.json"))
    ) {
      throw RequestError.authRequired();
    }

    const response = await this.createSession(params, {
      // Revisit these meta values once we support resume
      resume: (params._meta as NewSessionMeta | undefined)?.claudeCode?.options?.resume,
    });
    // Needs to happen after we return the session
    setTimeout(() => {
      this.sendAvailableCommandsUpdate(response.sessionId);
    }, 0);
    return response;
  }

  async unstable_forkSession(params: ForkSessionRequest): Promise<ForkSessionResponse> {
    const response = await this.createSession(
      {
        cwd: params.cwd,
        mcpServers: params.mcpServers ?? [],
        _meta: params._meta,
      },
      {
        resume: params.sessionId,
        forkSession: true,
      },
    );
    // Needs to happen after we return the session
    setTimeout(() => {
      this.sendAvailableCommandsUpdate(response.sessionId);
    }, 0);
    return response;
  }

  async unstable_resumeSession(params: ResumeSessionRequest): Promise<ResumeSessionResponse> {
    const result = await this.getOrCreateSession(params);

    // Needs to happen after we return the session
    setTimeout(() => {
      this.sendAvailableCommandsUpdate(params.sessionId);
    }, 0);
    return result;
  }

  async loadSession(params: LoadSessionRequest): Promise<LoadSessionResponse> {
    const result = await this.getOrCreateSession(params);

    await this.replaySessionHistory(params.sessionId);

    // Send available commands after replay so it doesn't interleave with history
    setTimeout(() => {
      this.sendAvailableCommandsUpdate(params.sessionId);
    }, 0);

    return result;
  }

  async listSessions(params: ListSessionsRequest): Promise<ListSessionsResponse> {
    const sdk_sessions = await listSessions({ dir: params.cwd ?? undefined });
    const sessions = [];

    for (const session of sdk_sessions) {
      if (!session.cwd) continue;
      sessions.push({
        sessionId: session.sessionId,
        cwd: session.cwd,
        title: sanitizeTitle(session.summary),
        updatedAt: new Date(session.lastModified).toISOString(),
      });
    }
    return {
      sessions,
    };
  }

  async authenticate(_params: AuthenticateRequest): Promise<void> {
    if (_params.methodId === "gateway") {
      this.gatewayAuthMeta = _params._meta as GatewayAuthMeta | undefined;
      return;
    }
    throw new Error("Method not implemented.");
  }

  async prompt(params: PromptRequest): Promise<PromptResponse> {
    const session = this.sessions[params.sessionId];
    if (!session) {
      throw new Error("Session not found");
    }

    session.cancelled = false;

    const userMessage = promptToClaude(params);

    const promptUuid = randomUUID();
    userMessage.uuid = promptUuid;

    // These local-only commands return a result without replaying the user
    // message. Mark promptReplayed=true so their result isn't consumed as a
    // background task result.
    const firstText = params.prompt[0]?.type === "text" ? params.prompt[0].text : "";
    const isLocalOnlyCommand =
      firstText.startsWith("/") && LOCAL_ONLY_COMMANDS.has(firstText.split(" ", 1)[0]);

    if (session.promptRunning || session.anyharness.pumpRunning) {
      // Either a prompt drain or the idle background pump currently owns the
      // message stream — queue up and wait for it to hand over. An
      // idle-blocked pump yields immediately via its interrupt.
      session.input.push(userMessage);
      const order = session.nextPendingOrder++;
      const pending = new Promise<boolean>((resolve) => {
        session.pendingMessages.set(promptUuid, { resolve, order });
      });
      session.anyharness.pumpInterrupt?.();
      const cancelled = await pending;
      if (cancelled) {
        return { stopReason: "cancelled" };
      }
    } else {
      session.input.push(userMessage);
    }

    session.promptRunning = true;
    // Reset usage accounting only now that this prompt owns the stream — never
    // at the top of prompt(), where it could zero a still-running idle pump
    // turn's or a concurrently-queued prompt's in-flight accumulation. The
    // idle pump and the spontaneous pre-turns a handed-off drain sees first
    // never write session.accumulatedUsage (see drainTurn), so from here it
    // reflects only this prompt's own turn(s).
    session.accumulatedUsage = {
      inputTokens: 0,
      outputTokens: 0,
      cachedReadTokens: 0,
      cachedWriteTokens: 0,
    };
    let handedOff = false;

    try {
      const outcome = await this.drainTurn({
        sessionId: params.sessionId,
        session,
        owner: "prompt",
        promptUuid,
        isLocalOnlyCommand,
      });
      if (outcome.kind === "handed_off") {
        handedOff = true;
        // the current loop stops with end_turn,
        // the loop of the next prompt continues running
        return { stopReason: "end_turn", usage: sessionUsage(session) };
      }
      if (outcome.kind === "stream_ended") {
        if (session.cancelled) {
          return { stopReason: "cancelled" };
        }
        throw new Error("Session did not end in result");
      }
      return { stopReason: outcome.stopReason, usage: sessionUsage(session) };
    } catch (error) {
      if (error instanceof RequestError || !(error instanceof Error)) {
        throw error;
      }
      const message = error.message;
      if (isProcessExitError(message)) {
        this.logger.error(`Session ${params.sessionId}: Claude Agent process died: ${message}`);
        session.settingsManager.dispose();
        session.anyharness.tailer?.dispose();
        session.input.end();
        delete this.sessions[params.sessionId];
        throw RequestError.internalError(
          undefined,
          "The Claude Agent process exited unexpectedly. Please start a new session.",
        );
      }
      throw error;
    } finally {
      if (!handedOff) {
        session.promptRunning = false;
        session.anyharness.turnActive = false;
        // This usually should not happen, but in case the loop finishes
        // without claude sending all message replays, we resolve the
        // next pending prompt call to ensure no prompts get stuck.
        this.resolveNextPendingPrompt(session);
        // The session is idle again — flush any goal/loop injection deferred
        // behind this prompt's turn (unless resolveNextPendingPrompt just handed
        // off to another queued prompt, in which case the flush no-ops).
        this.tryFlushDeferredInjections(params.sessionId);
        // Resume the idle background pump so injected goal/loop instructions
        // and spontaneous cron wake turns keep draining between prompts.
        this.startIdlePump(params.sessionId);
      }
    }
  }

  /**
   * Hands the message stream to the oldest queued prompt() call. Marks
   * promptRunning before resolving so no idle pump can start (and race the
   * awakened prompt's drain) in the gap before its continuation runs.
   */
  private resolveNextPendingPrompt(session: Session): void {
    if (session.pendingMessages.size === 0) {
      return;
    }
    const next = [...session.pendingMessages.entries()].sort((a, b) => a[1].order - b[1].order)[0];
    if (next) {
      session.promptRunning = true;
      next[1].resolve(false);
      session.pendingMessages.delete(next[0]);
    }
  }

  /**
   * Drains SDK messages for one turn, translating them into ACP session
   * updates. Shared between prompt() and the idle background pump.
   *
   * Returns when:
   * - the harness goes idle (turn ended),
   * - a queued prompt's user message replay is observed (handed off to the
   *   prompt() call awaiting it), or
   * - the SDK message stream ends.
   */
  private async drainTurn(params: {
    sessionId: string;
    session: Session;
    owner: "prompt" | "pump";
    promptUuid?: string;
    isLocalOnlyCommand?: boolean;
  }): Promise<
    | { kind: "turn_ended"; stopReason: StopReason }
    | { kind: "handed_off" }
    | { kind: "stream_ended" }
  > {
    const session = params.session;
    const promptUuid = params.promptUuid;
    const isLocalOnlyCommand = params.isLocalOnlyCommand === true;

    let lastAssistantTotalUsage: number | null = null;
    let stopReason: StopReason = "end_turn";

    // Classification of a turn drained by the idle pump:
    // - "injected": triggered by a goal/loop instruction we pushed ourselves
    // - "wake": a spontaneous native cron wake turn (emits loop_fired)
    // - "plain": spontaneous activity with no loop armed
    let turnKind: "unknown" | "injected" | "wake" | "plain" = "unknown";
    // True once a genuine spontaneous pre-turn — a native cron wake or an
    // injected goal/loop instruction — has been classified ahead of the
    // prompt's own message replay. Tracked as a plain boolean (rather than
    // reading turnKind at the idle boundary) because "wake"/"plain" are only
    // ever assigned inside the markSpontaneousTurn closure, so TS control-flow
    // narrowing treats a direct `turnKind === "wake"` comparison as dead. A
    // handed-off prompt drain only defers past an idle boundary when this is
    // set; otherwise the handed-off stream was the prompt's own turn.
    let sawSpontaneousPreTurn = false;
    // Whether the drain has reached the turn it is responsible for reporting.
    // A prompt() drain normally owns the stream from the start, but one handed
    // the stream by an interrupted idle pump (pendingQueryNext still holds the
    // pump's in-flight next()) can receive spontaneous cron-wake / injected
    // "pre-turns" first. Those pre-turns must be classified (loop_fired) and
    // streamed exactly as the pump would, and must NOT end the prompt() call or
    // count toward its usage — the prompt's own queued message replay is still
    // ahead. Local-only commands never replay a user message, so they own the
    // stream immediately. The idle pump never "owns" a turn in this sense.
    let inOwnTurn =
      params.owner !== "prompt" ? false : isLocalOnlyCommand || !session.pendingQueryNext;
    const markSpontaneousTurn = async (userText?: string) => {
      if (inOwnTurn || turnKind !== "unknown") {
        return;
      }
      const loops = activeLoops(session.anyharness);
      if (loops.length === 0) {
        turnKind = "plain";
        return;
      }
      if (!userText) {
        // A bare spontaneous assistant turn with no wake user-prompt in front of
        // it is NOT evidence of a specific loop fire: it may be a goal
        // continuation, a background-task wake, or a wake whose prompt didn't
        // match. Crediting loops[0] here corrupts fireCount and streams a
        // phantom loop_fired (goal continuations / task wakes must never count
        // as loop fires). Classify it as plain spontaneous activity — it is
        // still a genuine pre-turn, so mark it so a handed-off prompt drain
        // defers past its idle boundary; the wake attribution is left to the
        // user-prompt replay path (a native cron wake always injects one).
        turnKind = "plain";
        sawSpontaneousPreTurn = true;
        return;
      }
      const loop = matchLoopForWake(loops, userText);
      if (!loop) {
        // The user message matched no armed loop prompt (or matched more than
        // one ambiguously) — not enough evidence of a specific wake. Wait for
        // further activity rather than guessing a loop.
        return;
      }
      turnKind = "wake";
      sawSpontaneousPreTurn = true;
      const now = Date.now();
      loop.fireCount += 1;
      loop.lastFiredAtMs = now;
      loop.updatedAtMs = now;
      await this.sendAnyharnessEvent(params.sessionId, "loop_fired", {
        loop: loopWireFromState(loop),
        loopId: loop.loopId,
      });
    };

    let messagesSeen = 0;
    while (true) {
      const pull = session.pendingQueryNext ?? (session.pendingQueryNext = session.query.next());
      let iteration: IteratorResult<SDKMessage, void>;
      if (params.owner === "pump" && messagesSeen === 0) {
        // While idle-blocked (nothing drained yet), let a queued prompt()
        // take the stream over immediately. The in-flight next() promise is
        // left on the session so no message is lost across the handoff.
        const interrupted = new Promise<"interrupted">((resolve) => {
          session.anyharness.pumpInterrupt = () => resolve("interrupted");
        });
        let winner: IteratorResult<SDKMessage, void> | "interrupted";
        try {
          winner = await Promise.race([pull, interrupted]);
        } catch (error) {
          // The in-flight next() rejected. Clear it so the poisoned promise
          // isn't re-awaited by every future drain (which would wedge the
          // session forever); a later drain starts a fresh next().
          session.anyharness.pumpInterrupt = null;
          session.pendingQueryNext = null;
          throw error;
        }
        session.anyharness.pumpInterrupt = null;
        if (winner === "interrupted") {
          // Leave pendingQueryNext set: the pull is still in flight and the
          // prompt taking over will await it.
          return { kind: "handed_off" };
        }
        iteration = winner;
      } else {
        try {
          iteration = await pull;
        } catch (error) {
          // See above: a rejected next() must not stay cached, or the next
          // prompt()/pump drain re-throws the same stale error indefinitely.
          session.pendingQueryNext = null;
          throw error;
        }
      }
      session.pendingQueryNext = null;
      const { value: message, done } = iteration;

      if (done || !message) {
        return { kind: "stream_ended" };
      }
      messagesSeen += 1;
      // A message is being processed → a turn is active. Deferred goal/loop
      // injections wait for this to fall (at a turn boundary) so a `/goal`
      // local command can't degrade into a queued mid-turn prompt.
      session.anyharness.turnActive = true;

      switch (message.type) {
        case "system": {
          const systemMessage = message as
            | typeof message
            | LegacySessionStateChangedMessage
            | LegacyApiRetryMessage;

          if (systemMessage.subtype === "init") {
            // Cron wakes re-emit system:init at the start of the spontaneous
            // turn — reset classification so each wake is evaluated fresh.
            // A prompt draining pre-turns (handed off from an interrupted idle
            // pump) resets the same way until it reaches its own turn.
            if (!inOwnTurn) {
              turnKind = "unknown";
            }
            break;
          }
          if (systemMessage.subtype === "status") {
            if (systemMessage.status === "compacting") {
              await this.client.sessionUpdate({
                sessionId: systemMessage.session_id,
                update: {
                  sessionUpdate: "agent_message_chunk",
                  content: { type: "text", text: "Compacting..." },
                },
              });
            }
            break;
          }
          if (systemMessage.subtype === "compact_boundary") {
            // We don't know the exact size, but since we compacted,
            // we set it to zero. The client gets the exact size on the next message.
            lastAssistantTotalUsage = 0;
            await this.client.sessionUpdate({
              sessionId: systemMessage.session_id,
              update: {
                sessionUpdate: "agent_message_chunk",
                content: { type: "text", text: "\n\nCompacting completed." },
              },
            });
            break;
          }
          if (systemMessage.subtype === "local_command_output") {
            await this.client.sessionUpdate({
              sessionId: systemMessage.session_id,
              update: {
                sessionUpdate: "agent_message_chunk",
                content: { type: "text", text: systemMessage.content },
              },
            });
            break;
          }
          if (isLegacySessionStateChangedMessage(systemMessage)) {
            if (systemMessage.state === "idle") {
              // The CLI acknowledged an interrupt: cancel() set session.cancelled
              // and called query.interrupt(), and the native process recorded
              // "[Request interrupted by user]". No own-message replay ever
              // follows an interrupt, and once cancelled the user/assistant
              // branch short-circuits before the promptUuid match so inOwnTurn
              // can never flip. ANY drain — including a prompt still working
              // through handed-off pre-turns — must therefore end here, or the
              // next next() blocks forever and prompt() hangs (the 135s wedge).
              // This is the guarantee that every interrupt resolves prompt() as
              // cancelled.
              if (session.cancelled) {
                return { kind: "turn_ended", stopReason: "cancelled" };
              }
              if (params.owner === "pump" || inOwnTurn) {
                return { kind: "turn_ended", stopReason };
              }
              // A prompt() drain that doesn't yet own its turn was handed an
              // in-flight next() by an interrupted idle pump. Defer past this
              // idle boundary ONLY when a genuine spontaneous pre-turn (a native
              // cron wake or an injected goal/loop instruction) is what just
              // ended — then the prompt's own message replay is still queued
              // behind it, so reset classification and keep draining. When
              // nothing spontaneous was classified, the handed-off stream WAS
              // the prompt's own turn (the common no-cron-wake case — a normal
              // prompt while the idle pump was running), so end here instead of
              // hanging forever waiting for a replay that will never come.
              if (sawSpontaneousPreTurn) {
                turnKind = "unknown";
                sawSpontaneousPreTurn = false;
                stopReason = "end_turn";
                break;
              }
              return { kind: "turn_ended", stopReason };
            }
            break;
          }
          if (
            systemMessage.subtype === "task_started" ||
            systemMessage.subtype === "task_notification" ||
            systemMessage.subtype === "task_progress"
          ) {
            // Read-only activity roster: normalize the Claude task machinery
            // (background bash + subagents) into process_upserted /
            // subagent_upserted records. Best-effort — never fail the drain.
            void this.handleTaskEvent(params.sessionId, systemMessage);
            break;
          }
          if (
            systemMessage.subtype === "hook_started" ||
            systemMessage.subtype === "hook_progress" ||
            systemMessage.subtype === "hook_response" ||
            systemMessage.subtype === "files_persisted" ||
            systemMessage.subtype === "elicitation_complete" ||
            isLegacyApiRetryMessage(systemMessage)
          ) {
            // Todo: process via status api: https://docs.claude.com/en/docs/claude-code/hooks#hook-output
            break;
          }
          unreachable(systemMessage as never, this.logger);
          break;
        }
        case "result": {
          // Accumulate usage only for the turn(s) this prompt() call reports.
          // The idle pump (owner "pump") and the spontaneous cron-wake /
          // injected pre-turns a handed-off prompt drains before its own turn
          // (inOwnTurn still false) must not pollute the usage returned to the
          // prompt() caller. (inOwnTurn is only ever true for owner "prompt".)
          if (inOwnTurn) {
            session.accumulatedUsage.inputTokens += message.usage.input_tokens;
            session.accumulatedUsage.outputTokens += message.usage.output_tokens;
            session.accumulatedUsage.cachedReadTokens += message.usage.cache_read_input_tokens;
            session.accumulatedUsage.cachedWriteTokens += message.usage.cache_creation_input_tokens;
          }

          // Calculate context window size from modelUsage (minimum across all models used)
          const contextWindows = Object.values(message.modelUsage).map((m) => m.contextWindow);
          const contextWindowSize =
            contextWindows.length > 0 ? Math.min(...contextWindows) : 200000;

          // Send usage_update notification
          if (lastAssistantTotalUsage !== null) {
            await this.client.sessionUpdate({
              sessionId: params.sessionId,
              update: {
                sessionUpdate: "usage_update",
                used: lastAssistantTotalUsage,
                size: contextWindowSize,
                cost: {
                  amount: message.total_cost_usd,
                  currency: "USD",
                },
              },
            });
          }

          if (session.cancelled) {
            stopReason = "cancelled";
            break;
          }

          switch (message.subtype) {
            case "success": {
              if (message.result.includes("Please run /login")) {
                throw RequestError.authRequired();
              }
              if (message.stop_reason === "max_tokens") {
                stopReason = "max_tokens";
                break;
              }
              if (message.is_error) {
                throw RequestError.internalError(undefined, message.result);
              }
              // For local-only commands (no model invocation), the result
              // text is the command output — forward it to the client.
              if (isLocalOnlyCommand) {
                for (const notification of toAcpNotifications(
                  message.result,
                  "assistant",
                  params.sessionId,
                  this.toolUseCache,
                  this.client,
                  this.logger,
                )) {
                  await this.client.sessionUpdate(notification);
                }
              }
              break;
            }
            case "error_during_execution": {
              if (message.stop_reason === "max_tokens") {
                stopReason = "max_tokens";
                break;
              }
              if (message.is_error) {
                throw RequestError.internalError(
                  undefined,
                  message.errors.join(", ") || message.subtype,
                );
              }
              stopReason = "end_turn";
              break;
            }
            case "error_max_budget_usd":
            case "error_max_turns":
            case "error_max_structured_output_retries":
              if (message.is_error) {
                throw RequestError.internalError(
                  undefined,
                  message.errors.join(", ") || message.subtype,
                );
              }
              stopReason = "max_turn_requests";
              break;
            default:
              unreachable(message, this.logger);
              break;
          }
          break;
        }
        case "stream_event": {
          await markSpontaneousTurn();
          for (const notification of streamEventToAcpNotifications(
            message,
            params.sessionId,
            this.toolUseCache,
            this.client,
            this.logger,
            {
              clientCapabilities: this.clientCapabilities,
              cwd: session.cwd,
            },
          )) {
            await this.client.sessionUpdate(notification);
          }
          break;
        }
        case "user":
        case "assistant": {
          // Capture background-bash command / output-file for the activity
          // roster before any early-out below (runs regardless of cancellation).
          this.captureTaskIo(params.sessionId, message);

          if (session.cancelled) {
            break;
          }

          if (message.type === "assistant" && message.message.model !== "<synthetic>") {
            // Synthetic assistant messages are local-command echoes (e.g. an
            // injected /goal), not evidence of a cron wake turn.
            await markSpontaneousTurn();
          }

          // Check for prompt replay
          if (message.type === "user" && "uuid" in message && message.uuid) {
            if (message.uuid === promptUuid) {
              // This prompt's own message replay — from here the drain owns
              // the turn (stop classifying pre-turns and stop deferring the
              // turn_ended / usage boundary to a later replay).
              inOwnTurn = true;
              break;
            }

            const pending = session.pendingMessages.get(message.uuid as string);
            if (pending) {
              session.promptRunning = true;
              pending.resolve(false);
              session.pendingMessages.delete(message.uuid as string);
              // the current loop stops with end_turn,
              // the loop of the next prompt continues running
              return { kind: "handed_off" };
            }
            if (session.anyharness.injectedUuids.has(message.uuid as string)) {
              // Replay of a goal/loop instruction we injected — not a cron
              // wake, and not content the client should see.
              session.anyharness.injectedUuids.delete(message.uuid as string);
              turnKind = "injected";
              sawSpontaneousPreTurn = true;
              break;
            }
            if (!inOwnTurn && turnKind === "unknown") {
              // A spontaneous user message may be a native cron wake prompt.
              // This also covers a prompt drain working through a pre-turn it
              // was handed by an interrupted idle pump.
              await markSpontaneousTurn(userMessageText(message.message.content));
            }
            if ("isReplay" in message && message.isReplay) {
              // not pending or unrelated replay message
              break;
            }
          }

          // Store latest assistant usage (excluding subagents)
          if ((message.message as any).usage && message.parent_tool_use_id === null) {
            const messageWithUsage = message.message as unknown as SDKResultMessage;
            lastAssistantTotalUsage =
              messageWithUsage.usage.input_tokens +
              messageWithUsage.usage.output_tokens +
              messageWithUsage.usage.cache_read_input_tokens +
              messageWithUsage.usage.cache_creation_input_tokens;
          }

          // Slash commands like /compact can generate invalid output... doesn't match
          // their own docs: https://docs.anthropic.com/en/docs/claude-code/sdk/sdk-slash-commands#%2Fcompact-compact-conversation-history
          if (
            typeof message.message.content === "string" &&
            message.message.content.includes("<local-command-stdout>")
          ) {
            this.logger.log(message.message.content);
            break;
          }

          if (
            typeof message.message.content === "string" &&
            message.message.content.includes("<local-command-stderr>")
          ) {
            this.logger.error(message.message.content);
            break;
          }
          // Skip these user messages for now, since they seem to just be messages we don't want in the feed
          if (
            message.type === "user" &&
            (typeof message.message.content === "string" ||
              (Array.isArray(message.message.content) &&
                message.message.content.length === 1 &&
                message.message.content[0].type === "text"))
          ) {
            break;
          }

          if (
            message.type === "assistant" &&
            message.message.model === "<synthetic>" &&
            Array.isArray(message.message.content) &&
            message.message.content.length === 1 &&
            message.message.content[0].type === "text" &&
            message.message.content[0].text.includes("Please run /login")
          ) {
            throw RequestError.authRequired();
          }

          const content =
            message.type === "assistant"
              ? // Handled by stream events above
                message.message.content.filter((item) => !["text", "thinking"].includes(item.type))
              : message.message.content;

          for (const notification of toAcpNotifications(
            content,
            message.message.role,
            params.sessionId,
            this.toolUseCache,
            this.client,
            this.logger,
            {
              clientCapabilities: this.clientCapabilities,
              parentToolUseId: message.parent_tool_use_id,
              cwd: session.cwd,
            },
          )) {
            await this.client.sessionUpdate(notification);
          }
          break;
        }
        case "tool_progress":
        case "tool_use_summary":
        case "auth_status":
        case "prompt_suggestion":
        case "rate_limit_event":
          break;
        default:
          unreachable(message);
          break;
      }
    }
  }

  /**
   * Background pump: drains the SDK message stream while no client prompt()
   * is running, so injected goal/loop instructions are processed while idle
   * and spontaneous native cron wake turns are translated as they happen.
   */
  private startIdlePump(sessionId: string): void {
    const session = this.sessions[sessionId];
    if (!session) {
      return;
    }
    const ah = session.anyharness;
    if (ah.pumpRunning || session.promptRunning) {
      return;
    }
    ah.pumpRunning = true;
    void this.runIdlePump(sessionId, session)
      .catch((error) => {
        this.logger.error(`[anyharness] idle pump for session ${sessionId} crashed:`, error);
      })
      .finally(() => {
        ah.pumpRunning = false;
        // If a prompt queued up while the pump was exiting, unstick it the
        // same way prompt() unsticks the next pending prompt.
        if (this.sessions[sessionId] === session && !session.promptRunning) {
          this.resolveNextPendingPrompt(session);
        }
      });
  }

  private async runIdlePump(sessionId: string, session: Session): Promise<void> {
    while (true) {
      if (this.sessions[sessionId] !== session || session.promptRunning) {
        return;
      }
      if (session.pendingMessages.size > 0) {
        // A prompt queued while a spontaneous turn was draining — yield at
        // the turn boundary (the finally above resolves it).
        return;
      }
      session.cancelled = false;
      try {
        const outcome = await this.drainTurn({ sessionId, session, owner: "pump" });
        // The pump is idle again between spontaneous turns — flush any goal/loop
        // injection deferred behind the turn that just ended.
        session.anyharness.turnActive = false;
        this.tryFlushDeferredInjections(sessionId);
        if (outcome.kind === "handed_off" || outcome.kind === "stream_ended") {
          return;
        }
        // turn_ended: keep pumping for the next spontaneous turn.
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (error instanceof Error && isProcessExitError(message)) {
          this.logger.error(`Session ${sessionId}: Claude Agent process died: ${message}`);
          if (this.sessions[sessionId] === session) {
            session.settingsManager.dispose();
            session.anyharness.tailer?.dispose();
            session.input.end();
            for (const [, pending] of session.pendingMessages) {
              pending.resolve(true);
            }
            session.pendingMessages.clear();
            delete this.sessions[sessionId];
          }
          return;
        }
        this.logger.error(`[anyharness] idle pump error for session ${sessionId}:`, error);
        // Stop instead of spinning on a persistent error; a later prompt() or
        // injected instruction restarts the pump.
        return;
      }
    }
  }

  /**
   * anyharness GoalPort/LoopPort extension methods (wire contract v1).
   * Accepts both the `_anyharness/...` wire spelling (ACP requires ext
   * methods to be `_`-prefixed on the wire) and the stripped
   * `anyharness/...` spelling in case the transport strips the prefix
   * before dispatch.
   */
  async extMethod(
    method: string,
    params: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    const normalized = method.startsWith("_") ? method.slice(1) : method;
    if (!normalized.startsWith("anyharness/")) {
      throw RequestError.methodNotFound(method);
    }
    this.logger.log(`[anyharness] extMethod received on the wire as: ${method}`);

    const sessionId = typeof params.sessionId === "string" ? params.sessionId : undefined;
    if (!sessionId) {
      throw RequestError.invalidParams(undefined, "sessionId is required");
    }
    const session = this.sessions[sessionId];
    if (!session) {
      throw RequestError.resourceNotFound(sessionId);
    }

    switch (normalized) {
      case "anyharness/goal/set":
        return await this.anyharnessGoalSet(sessionId, session, params);
      case "anyharness/goal/get": {
        // Ensure the tailer's synchronous seed has run before answering. On a
        // resumed/forked session a native goal survives --resume but the
        // mirror starts null; without the seed, an attach-time reconcile that
        // races the CLI's SessionStart hook would read null and clear a goal
        // that is still active natively. ensureTranscriptTailer is idempotent
        // and runs readLastGoalStatus() to seed ah.goal for resume/fork.
        this.ensureTranscriptTailer(sessionId, session);
        const goal = session.anyharness.goal;
        return { goal: goal ? goalWireFromState(goal) : null };
      }
      case "anyharness/goal/clear":
        return await this.anyharnessGoalClear(sessionId, session);
      case "anyharness/loop/set":
        return await this.anyharnessLoopSet(sessionId, session, params);
      case "anyharness/loop/clear":
        return await this.anyharnessLoopClear(sessionId, session, params);
      case "anyharness/loop/list":
        return { loops: activeLoops(session.anyharness).map(loopWireFromState) };
      case "anyharness/activity/list": {
        // The reconcile pull (attach/resume): the whole SessionActivity mirror —
        // goal + loops + read-only rosters — served from the adapter's tracked
        // state, so a reattaching runtime heals without replaying the stream.
        this.ensureTranscriptTailer(sessionId, session);
        const ah = session.anyharness;
        return {
          goal: ah.goal ? goalWireFromState(ah.goal) : null,
          loops: activeLoops(ah).map(loopWireFromState),
          processes: [...ah.processes.values()].map(processWireFromState),
          subagents: [...ah.subagents.values()],
        };
      }
      default:
        throw RequestError.methodNotFound(method);
    }
  }

  /**
   * Registers a bounded wait for a goal_status transcript row matching the
   * predicate. Rows are delivered by handleTranscriptRow AFTER the mirror
   * has transitioned and the tagged notification has been emitted, so a
   * resolved wait means the native write round-tripped.
   */
  private waitForGoalRow(
    session: Session,
    predicate: (row: GoalStatusRow) => boolean,
    timeoutMs: number,
  ): Promise<GoalStatusRow | null> {
    const ah = session.anyharness;
    return new Promise<GoalStatusRow | null>((resolve) => {
      const watcher = (row: GoalStatusRow) => {
        if (!predicate(row)) {
          return;
        }
        remove();
        clearTimeout(timer);
        resolve(row);
      };
      const remove = () => {
        const index = ah.goalRowWatchers.indexOf(watcher);
        if (index >= 0) {
          ah.goalRowWatchers.splice(index, 1);
        }
      };
      const timer = setTimeout(() => {
        remove();
        resolve(null);
      }, timeoutMs);
      timer.unref?.();
      ah.goalRowWatchers.push(watcher);
    });
  }

  private async anyharnessGoalSet(
    sessionId: string,
    session: Session,
    params: Record<string, unknown>,
  ): Promise<{ goal: GoalWire }> {
    const ah = session.anyharness;
    const status = typeof params.status === "string" ? params.status : undefined;
    if (status !== undefined && status !== "active" && status !== "paused") {
      throw RequestError.invalidParams(undefined, 'status must be "active" or "paused"');
    }
    if (status === "paused") {
      throw RequestError.invalidParams(
        undefined,
        'status "paused" is not supported: Claude Code has no native goal pause',
      );
    }
    const objective = typeof params.objective === "string" ? params.objective.trim() : undefined;

    if (objective === undefined || objective === "") {
      // Status/budget-only patch (codex semantics). Claude goals are always
      // active and have no token budget — the patch is a no-op that returns
      // the current goal unchanged.
      if (!ah.goal || ah.goal.status !== "active") {
        throw RequestError.invalidParams(
          undefined,
          "objective is required (no active goal to patch)",
        );
      }
      // Emit the tagged goal_updated notification even though nothing changed.
      // anyharness treats every set as a mutation and blocks on a
      // goal_updated/goal_met/goal_cleared round-trip after the ext response;
      // codex always emits one for an objective-omitted patch, so without this
      // the wait would time out into a 409 despite the call succeeding.
      await this.sendAnyharnessEvent(sessionId, "goal_updated", {
        goal: goalWireFromState(ah.goal),
      });
      return { goal: goalWireFromState(ah.goal) };
    }

    this.ensureTranscriptTailer(sessionId, session);

    if (!this.canInjectNow(session)) {
      // Deferral fix: a turn is streaming, so a `/goal` sent now would degrade
      // to a never-executing queued prompt (the 30s-timeout 409). Hold it until
      // the turn boundary. The arm sentinel that fires after injection drives
      // the real goal_updated notification via handleTranscriptRow (the mirror
      // stays untouched until then — no optimistic state). Return a provisional
      // pending goal; anyharness treats ext-method responses as
      // optimistic-pending only and reconciles from the notification.
      this.enqueueInjectedInstruction(sessionId, session, `/goal ${objective}`);
      return { goal: pendingGoalWire(objective) };
    }

    // Idle path: "/goal <condition>" arms the goal; re-sending replaces. The
    // mirror transitions only once the native arm sentinel round-trips through
    // the transcript — no optimistic saved-state. The confirmation clock starts
    // at injection (which, while idle, happens synchronously below).
    const confirmed = this.waitForGoalRow(
      session,
      (row) => classifyGoalStatus(row) === "armed" && (row.condition ?? "").trim() === objective,
      GOAL_SET_TIMEOUT_MS,
    );
    this.enqueueInjectedInstruction(sessionId, session, `/goal ${objective}`);

    const row = await confirmed;
    if (!row || !ah.goal) {
      throw RequestError.internalError(
        undefined,
        `goal arming was not confirmed by the harness within ${GOAL_SET_TIMEOUT_MS}ms ` +
          "(goal commands sent while a turn is streaming queue as prompt-mode commands and do not execute)",
      );
    }
    return { goal: goalWireFromState(ah.goal) };
  }

  private async anyharnessGoalClear(
    sessionId: string,
    session: Session,
  ): Promise<{ cleared: boolean }> {
    const ah = session.anyharness;
    const hadActiveGoal = ah.goal !== null && ah.goal.status === "active";

    this.ensureTranscriptTailer(sessionId, session);

    if (!this.canInjectNow(session)) {
      // Deferral fix (as goal/set): hold "/goal clear" until the turn boundary
      // rather than degrading it mid-turn. The clear sentinel fired after
      // injection drives the goal_cleared notification via handleTranscriptRow;
      // the ext response is provisional.
      this.enqueueInjectedInstruction(sessionId, session, "/goal clear");
      return { cleared: hadActiveGoal };
    }

    const confirmed = this.waitForGoalRow(
      session,
      (row) => classifyGoalStatus(row) === "cleared",
      GOAL_CLEAR_TIMEOUT_MS,
    );
    // Always send the native clear so mirror drift heals; with no native
    // goal armed it is a zero-token no-op ("No goal set") that writes no
    // transcript row, so only wait for confirmation when one is expected.
    this.enqueueInjectedInstruction(sessionId, session, "/goal clear");
    if (!hadActiveGoal) {
      return { cleared: false };
    }

    const row = await confirmed;
    if (!row) {
      throw RequestError.internalError(
        undefined,
        `goal clear was not confirmed by the harness within ${GOAL_CLEAR_TIMEOUT_MS}ms ` +
          "(goal commands sent while a turn is streaming queue as prompt-mode commands and do not execute)",
      );
    }
    return { cleared: true };
  }

  private async anyharnessLoopSet(
    sessionId: string,
    session: Session,
    params: Record<string, unknown>,
  ): Promise<{ loop: LoopWire }> {
    const ah = session.anyharness;
    const prompt = typeof params.prompt === "string" ? params.prompt.trim() : "";
    if (!prompt) {
      throw RequestError.invalidParams(undefined, "prompt is required");
    }
    const schedule = params.schedule as LoopSchedule | undefined;
    if (
      !schedule ||
      (schedule.kind !== "interval" && schedule.kind !== "cron") ||
      typeof schedule.expr !== "string" ||
      !schedule.expr.trim()
    ) {
      throw RequestError.invalidParams(
        undefined,
        'schedule { kind: "interval"|"cron", expr } is required',
      );
    }
    const expr = schedule.expr.trim();
    const recurring = params.recurring !== false;

    // Ensure the transcript tailer is running before the loop can fire — it is
    // the sole source of loop-fire bookkeeping (native cron wakes don't replay a
    // matchable prompt on the SDK stream; see recordLoopFireFromTranscript).
    this.ensureTranscriptTailer(sessionId, session);

    if (!this.canInjectNow(session)) {
      // Deferral fix (mirrors goal/set): a turn is streaming, so "/loop" sent
      // now would degrade to a never-executing queued prompt. Blocking on the
      // 60s CronCreate race here — while the /loop injection is still deferred
      // behind the turn — guarantees a timeout on any turn longer than 60s,
      // which minted a phantom "provisional-" loop AND (when the deferred /loop
      // finally ran) a second real-id loop for the same cron. Instead: register
      // the pending set so the deferred /loop's CronCreate attributes to this
      // request by prompt, enqueue, and return a provisional loop immediately.
      // The mirror stays untouched (no optimistic state) until the loop_upserted
      // notification lands — that notification is the source of truth.
      ah.pendingLoopSets.push({
        prompt,
        schedule: { kind: schedule.kind, expr },
        recurring,
        requestedAtMs: Date.now(),
        resolve: () => {},
      });
      this.enqueueInjectedInstruction(sessionId, session, `/loop ${expr} ${prompt}`);
      return { loop: pendingLoopWire(prompt, { kind: schedule.kind, expr }, recurring) };
    }

    // Idle path: "/loop" injects synchronously, so racing the CronCreate here is
    // bounded by real execution time. Both schedule kinds translate to
    // "/loop <expr> <prompt>": interval expressions like "5m" directly, cron
    // kind as the raw crontab string.
    const created = new Promise<LoopState>((resolve) => {
      ah.pendingLoopSets.push({
        prompt,
        schedule: { kind: schedule.kind, expr },
        recurring,
        requestedAtMs: Date.now(),
        resolve,
      });
    });
    this.enqueueInjectedInstruction(sessionId, session, `/loop ${expr} ${prompt}`);

    // Wait for the CronCreate tool_use so the response carries the real
    // native cron id; fall back to a provisional loop on timeout (the
    // loop_upserted notification remains the source of truth either way, and
    // handleCronTool collapses the placeholder when the real CronCreate lands).
    const observed = await Promise.race([created, sleep(LOOP_SET_TIMEOUT_MS).then(() => null)]);
    if (observed) {
      return { loop: loopWireFromState(observed) };
    }

    this.logger.error(
      `[anyharness] loop/set: CronCreate not observed within ${LOOP_SET_TIMEOUT_MS}ms; returning provisional loop`,
    );
    const pendingIndex = ah.pendingLoopSets.findIndex((p) => p.prompt === prompt);
    if (pendingIndex >= 0) {
      ah.pendingLoopSets.splice(pendingIndex, 1);
    }
    const loop: LoopState = {
      loopId: `provisional-${randomUUID().slice(0, 8)}`,
      prompt,
      schedule: { kind: schedule.kind, expr },
      recurring,
      status: "active",
      lastFiredAtMs: null,
      fireCount: 0,
      updatedAtMs: Date.now(),
    };
    ah.loops.set(loop.loopId, loop);
    void this.sendAnyharnessEvent(sessionId, "loop_upserted", {
      loop: loopWireFromState(loop),
      loopId: loop.loopId,
    });
    return { loop: loopWireFromState(loop) };
  }

  private async anyharnessLoopClear(
    sessionId: string,
    session: Session,
    params: Record<string, unknown>,
  ): Promise<{ cleared: number }> {
    const ah = session.anyharness;
    const loopId = typeof params.loopId === "string" ? params.loopId : undefined;
    const targets = activeLoops(ah).filter((loop) => !loopId || loop.loopId === loopId);
    if (targets.length === 0) {
      return { cleared: 0 };
    }

    const instruction = loopId
      ? `Use the CronDelete tool to delete the cron job with id "${loopId}". Do nothing else and reply with only: done`
      : "Use the CronList tool to list all cron jobs, then use the CronDelete tool to delete each one. Do nothing else and reply with only: done";
    this.enqueueInjectedInstruction(sessionId, session, instruction);

    // Wait (bounded) for CronDelete tool_use observations to confirm.
    const confirmed = new Promise<void>((resolve) => {
      const check = () => {
        if (targets.every((loop) => loop.status === "cleared")) {
          resolve();
        }
      };
      check();
      ah.loopClearWatchers.push(check);
    });
    await Promise.race([confirmed, sleep(LOOP_CLEAR_TIMEOUT_MS)]);

    // Mark any unconfirmed targets cleared anyway (optimistic) so state and
    // notifications stay consistent with the requested mutation.
    for (const loop of targets) {
      if (loop.status !== "cleared") {
        this.logger.error(
          `[anyharness] loop/clear: CronDelete for ${loop.loopId} not observed within ${LOOP_CLEAR_TIMEOUT_MS}ms; marking cleared optimistically`,
        );
        loop.status = "cleared";
        loop.updatedAtMs = Date.now();
        void this.sendAnyharnessEvent(sessionId, "loop_removed", { loopId: loop.loopId });
      }
    }
    return { cleared: targets.length };
  }

  /** Observes CronCreate/CronDelete/CronList executions (via PostToolUse hook). */
  private async handleCronTool(
    sessionId: string,
    toolName: string,
    toolInput: unknown,
    toolResponse: unknown,
  ): Promise<void> {
    const session = this.sessions[sessionId];
    if (!session) {
      return;
    }
    const ah = session.anyharness;
    this.logger.log(
      `[anyharness] observed ${toolName}: input=${safeJson(toolInput)} response=${safeJson(toolResponse)}`,
    );

    if (toolName === "CronCreate") {
      const input = (toolInput ?? {}) as { cron?: unknown; prompt?: unknown; recurring?: unknown };
      const promptText = typeof input.prompt === "string" ? input.prompt : "";
      // Live-verified: CronCreate returns a prose result ("Scheduled recurring
      // job <id> …") — the real id is only in the string, so parse it first
      // before the structured/synthesized fallbacks.
      const loopId =
        parseCronIdFromResult(toolResponse) ??
        extractCronId(toolResponse) ??
        extractCronId(toolInput) ??
        `cron-${randomUUID().slice(0, 8)}`;

      // Match the oldest pending loop/set (by prompt when possible).
      let pendingIndex = ah.pendingLoopSets.findIndex((p) => p.prompt === promptText);
      if (pendingIndex < 0 && ah.pendingLoopSets.length > 0) {
        pendingIndex = 0;
      }
      const pending = pendingIndex >= 0 ? ah.pendingLoopSets.splice(pendingIndex, 1)[0] : undefined;

      const loop: LoopState = {
        loopId,
        prompt: promptText || pending?.prompt || "",
        schedule:
          pending?.schedule ??
          (typeof input.cron === "string"
            ? { kind: "cron", expr: input.cron }
            : { kind: "cron", expr: "" }),
        recurring:
          typeof input.recurring === "boolean" ? input.recurring : (pending?.recurring ?? true),
        status: "active",
        lastFiredAtMs: null,
        fireCount: 0,
        updatedAtMs: Date.now(),
      };

      // Collapse any synthetic placeholder loop for the same prompt into this
      // real cron. A loop/set that fell back to a "provisional-" loop (its
      // CronCreate not observed before the idle-path RPC timeout) is THIS cron
      // under a placeholder id — carry its fire bookkeeping forward and drop it,
      // so one cron never shows as two loops. reconcileSessionCrons can't heal
      // this once a real-id loop exists (its synthetic-upgrade branch is skipped
      // and its removal sweep spares synthetic ids), so it must be collapsed here.
      const placeholders =
        loop.prompt.length > 0
          ? activeLoops(ah).filter(
              (other) =>
                other.loopId !== loopId &&
                isSyntheticLoopId(other.loopId) &&
                other.prompt === loop.prompt,
            )
          : [];
      for (const placeholder of placeholders) {
        loop.fireCount = Math.max(loop.fireCount, placeholder.fireCount);
        loop.lastFiredAtMs = loop.lastFiredAtMs ?? placeholder.lastFiredAtMs;
        ah.loops.delete(placeholder.loopId);
      }

      ah.loops.set(loopId, loop);
      pending?.resolve(loop);
      await this.sendAnyharnessEvent(sessionId, "loop_upserted", {
        loop: loopWireFromState(loop),
        loopId,
      });
      for (const placeholder of placeholders) {
        await this.sendAnyharnessEvent(sessionId, "loop_removed", { loopId: placeholder.loopId });
      }
      return;
    }

    if (toolName === "CronDelete") {
      const deletedId =
        extractCronId(toolInput) ??
        extractCronId(toolResponse) ??
        parseCronIdFromResult(toolResponse);
      const active = activeLoops(ah);
      let target = deletedId ? ah.loops.get(deletedId) : undefined;
      if (!target && active.length === 1) {
        target = active[0];
      }
      if (!target) {
        this.logger.error(
          `[anyharness] CronDelete observed but no matching loop (id=${deletedId ?? "unknown"})`,
        );
        return;
      }
      if (target.status !== "cleared") {
        target.status = "cleared";
        target.updatedAtMs = Date.now();
        await this.sendAnyharnessEvent(sessionId, "loop_removed", { loopId: target.loopId });
      }
      for (const watcher of ah.loopClearWatchers) {
        watcher();
      }
      return;
    }

    if (toolName === "CronList") {
      // A CronList result is itself an authoritative snapshot of the armed
      // crons — reconcile the mirror against it, same as a hook's session_crons.
      const list = extractCronArray(toolResponse) ?? extractCronArray(toolInput);
      if (list) {
        await this.handleSessionCrons(sessionId, list);
      }
    }
  }

  /**
   * Reconciles the loop mirror against a `session_crons` snapshot. Every hook
   * firing hands the adapter this free, authoritative list of armed crons
   * (harness-runtime-mechanics §3); reconciling against it heals drift — a cron
   * armed or deleted by a bare TUI against the same session, or a synthesized
   * placeholder id upgraded to its real cron id — without a model-costing
   * CronList. Emits loop_upserted / loop_removed for the resulting transitions.
   */
  private async handleSessionCrons(sessionId: string, crons: unknown): Promise<void> {
    const session = this.sessions[sessionId];
    if (!session) {
      return;
    }
    if (!Array.isArray(crons)) {
      return;
    }
    const ah = session.anyharness;
    const { upserted, removed } = reconcileSessionCrons(ah, crons, Date.now());
    if (upserted.length === 0 && removed.length === 0) {
      return;
    }
    this.logger.log(`[anyharness] session_crons reconcile: +${upserted.length} -${removed.length}`);
    for (const loop of upserted) {
      await this.sendAnyharnessEvent(sessionId, "loop_upserted", {
        loop: loopWireFromState(loop),
        loopId: loop.loopId,
      });
    }
    for (const loopId of removed) {
      await this.sendAnyharnessEvent(sessionId, "loop_removed", { loopId });
    }
    // A reconcile can satisfy an outstanding loop/clear (a cron deleted out of
    // band vanished from the snapshot).
    if (removed.length > 0) {
      for (const watcher of ah.loopClearWatchers) {
        watcher();
      }
    }
  }

  /** The per-subagent transcript path used as its live nested-transcript feed. */
  private subagentFeedPathFor(session: Session, sessionId: string, taskId: string): string {
    const parentTranscript =
      session.anyharness.transcriptPath ??
      this.pendingTranscriptPaths.get(sessionId) ??
      computeTranscriptPath(CLAUDE_CONFIG_DIR, session.cwd, sessionId);
    return subagentFeedPath(parentTranscript, sessionId, taskId);
  }

  /**
   * Normalizes a Claude task lifecycle system event into the read-only activity
   * roster. Background bash (`task_type: local_bash`) becomes an
   * ActivityProcess (process_upserted); a subagent (`task_type: local_agent`)
   * becomes an ActivitySubagent (subagent_upserted). The feed transport is
   * always a membrane-side tail_file(path) that the runtime swaps for an opaque
   * FeedRef. Best-effort — a roster miss must never break the drain.
   */
  private async handleTaskEvent(sessionId: string, msg: SDKMessage): Promise<void> {
    const session = this.sessions[sessionId];
    if (!session) {
      return;
    }
    const ah = session.anyharness;
    const now = Date.now();
    try {
      const event = msg as unknown as {
        subtype: "task_started" | "task_progress" | "task_notification";
        task_id?: string;
        tool_use_id?: string;
        task_type?: string;
        subagent_type?: string;
        agent_type?: string;
        model?: string;
        description?: string;
        prompt?: string;
        summary?: string;
        status?: "completed" | "failed" | "stopped";
        output_file?: string;
        usage?: { total_tokens: number; tool_uses: number; duration_ms: number };
      };
      const taskId = event.task_id;
      if (!taskId) {
        return;
      }
      // Flat usage fields matching the ActivitySubagentWire contract (seconds,
      // not milliseconds). Nesting them or using different names makes the
      // runtime read them as absent.
      const usage = event.usage
        ? {
            tokensUsed: event.usage.total_tokens,
            toolCalls: event.usage.tool_uses,
            durationSeconds:
              typeof event.usage.duration_ms === "number" ? event.usage.duration_ms / 1000 : null,
          }
        : null;
      const outputFile = typeof event.output_file === "string" ? event.output_file : null;

      if (event.subtype === "task_started") {
        if (event.task_type === "local_agent") {
          const subagent: SubagentState = {
            id: taskId,
            agentType: event.subagent_type ?? event.agent_type ?? null,
            description: event.description ?? null,
            prompt: event.prompt ?? null,
            model: event.model ?? null,
            background: true,
            status: "running",
            summary: null,
            tokensUsed: null,
            toolCalls: null,
            durationSeconds: null,
            // The per-agent transcript exists from spawn — open a live feed now.
            feed: {
              transport: "tail_file",
              path: this.subagentFeedPathFor(session, sessionId, taskId),
            },
            updatedAtMs: now,
          };
          ah.subagents.set(taskId, subagent);
          await this.sendAnyharnessEvent(sessionId, "subagent_upserted", { subagent });
          return;
        }
        // local_bash (and any other non-agent task) → a background process.
        const captured = event.tool_use_id ? ah.taskToolUse.get(event.tool_use_id) : undefined;
        const process: ProcessState = {
          id: taskId,
          command: captured?.command || event.description || "",
          cwd: session.cwd ?? null,
          status: "running",
          exitCode: null,
          pid: null,
          startedAtMs: now,
          endedAtMs: null,
          feed: captured?.outputFile ? { transport: "tail_file", path: captured.outputFile } : null,
          toolUseId: event.tool_use_id ?? null,
          updatedAtMs: now,
        };
        ah.processes.set(taskId, process);
        await this.sendAnyharnessEvent(sessionId, "process_upserted", {
          process: processWireFromState(process),
        });
        return;
      }

      if (event.subtype === "task_progress") {
        const subagent = ah.subagents.get(taskId);
        if (subagent) {
          if (usage) {
            Object.assign(subagent, usage);
          }
          if (event.summary) {
            subagent.summary = event.summary;
          }
          subagent.updatedAtMs = now;
          await this.sendAnyharnessEvent(sessionId, "subagent_upserted", { subagent });
        }
        return;
      }

      // task_notification: terminal status (completed | failed | stopped).
      const terminalFailed = event.status === "failed" || event.status === "stopped";
      const subagent = ah.subagents.get(taskId);
      if (subagent) {
        subagent.status = terminalFailed ? "failed" : "completed";
        if (event.summary) {
          subagent.summary = event.summary;
        }
        if (usage) {
          Object.assign(subagent, usage);
        }
        if (outputFile) {
          subagent.feed = { transport: "tail_file", path: outputFile };
        }
        subagent.updatedAtMs = now;
        await this.sendAnyharnessEvent(sessionId, "subagent_upserted", { subagent });
        return;
      }
      const process = ah.processes.get(taskId);
      if (process) {
        process.status = "exited";
        // Claude reports no numeric exit code; leave it null (unknown).
        process.endedAtMs = now;
        if (outputFile) {
          process.feed = { transport: "tail_file", path: outputFile };
        }
        process.updatedAtMs = now;
        await this.sendAnyharnessEvent(sessionId, "process_upserted", {
          process: processWireFromState(process),
        });
      }
    } catch (error) {
      this.logger.error(`[anyharness] handleTaskEvent failed:`, error);
    }
  }

  /**
   * Captures the two facts the task lifecycle events don't carry themselves: a
   * background Bash command (from the spawning assistant tool_use) and its
   * output-file path (from the tool_result). Keyed by tool_use id so task_started
   * — which carries tool_use_id — can resolve the command, and so a process feed
   * can open on the live output file before the completion notification.
   */
  private captureTaskIo(sessionId: string, message: SDKMessage): void {
    const session = this.sessions[sessionId];
    if (!session) {
      return;
    }
    const ah = session.anyharness;
    const content = (message as { message?: { content?: unknown } }).message?.content;
    if (!Array.isArray(content)) {
      return;
    }
    for (const block of content) {
      if (typeof block !== "object" || block === null) {
        continue;
      }
      const b = block as {
        type?: string;
        id?: string;
        name?: string;
        input?: { command?: unknown };
        tool_use_id?: string;
        content?: unknown;
      };
      if (b.type === "tool_use" && typeof b.id === "string") {
        const command =
          b.input && typeof b.input.command === "string" ? b.input.command : undefined;
        if (command) {
          const existing = ah.taskToolUse.get(b.id) ?? {};
          existing.command = command;
          ah.taskToolUse.set(b.id, existing);
        }
      } else if (b.type === "tool_result" && typeof b.tool_use_id === "string") {
        const outputFile = parseBackgroundOutputFile(b.content);
        if (!outputFile) {
          continue;
        }
        const existing = ah.taskToolUse.get(b.tool_use_id) ?? {};
        existing.outputFile = outputFile;
        ah.taskToolUse.set(b.tool_use_id, existing);
        // If the process already started, attach the live feed and re-emit.
        const process = [...ah.processes.values()].find((p) => p.toolUseId === b.tool_use_id);
        if (process && !process.feed) {
          process.feed = { transport: "tail_file", path: outputFile };
          process.updatedAtMs = Date.now();
          void this.sendAnyharnessEvent(sessionId, "process_upserted", {
            process: processWireFromState(process),
          });
        }
      }
    }
  }

  /**
   * Queues a `/goal …` / `/loop …` instruction for injection as a user message.
   * If the session is idle it is injected immediately; otherwise it is held in
   * `deferredInjections` and flushed at the next turn boundary. This is the goal
   * deferral fix: a `/goal` local command sent while a turn is streaming
   * silently degrades to a never-executing queued prompt (the 30s-timeout bug),
   * so we hold it until idle. The deferred goal/loop set methods return a
   * provisional response immediately with NO fork-side confirmation wait — the
   * mirror reconciles from the later notification, so no clock can elapse during
   * the preceding turn. The uuid is remembered so the drain loop can tell these
   * turns apart from native cron wakes and keep their replays out of the feed.
   */
  private enqueueInjectedInstruction(sessionId: string, session: Session, text: string): void {
    const uuid = randomUUID();
    session.anyharness.injectedUuids.add(uuid);
    session.anyharness.deferredInjections.push({ uuid, text });
    this.tryFlushDeferredInjections(sessionId);
  }

  /** True when no turn is streaming, so an injected local command will execute. */
  private canInjectNow(session: Session): boolean {
    return !session.promptRunning && !session.anyharness.turnActive;
  }

  /**
   * Flushes queued injections into the message stream once the session is idle,
   * then ensures a pump drains the resulting turn(s). No-op mid-turn; the prompt
   * finally and the idle-pump loop re-drive it at each turn boundary.
   */
  private tryFlushDeferredInjections(sessionId: string): void {
    const session = this.sessions[sessionId];
    if (!session) {
      return;
    }
    const ah = session.anyharness;
    if (ah.deferredInjections.length === 0 || !this.canInjectNow(session)) {
      return;
    }
    const pending = ah.deferredInjections.splice(0, ah.deferredInjections.length);
    for (const injection of pending) {
      const message: SDKUserMessage = {
        type: "user",
        message: { role: "user", content: [{ type: "text", text: injection.text }] },
        session_id: sessionId,
        parent_tool_use_id: null,
        uuid: injection.uuid as SDKUserMessage["uuid"],
      };
      session.input.push(message);
    }
    // Ensure something drains the resulting turn when no prompt is active.
    this.startIdlePump(sessionId);
  }

  /**
   * Emits a zero-content agent_message_chunk tagged with
   * _meta.anyharness.transcriptEvent (kept out of transcripts by anyharness
   * via NON_TRANSCRIPT_CHUNK_EVENTS).
   */
  private async sendAnyharnessEvent(
    sessionId: string,
    transcriptEvent: AnyharnessTranscriptEvent,
    payload: {
      goal?: GoalWire;
      loop?: LoopWire;
      loopId?: string;
      process?: ProcessWire;
      subagent?: SubagentWire;
    },
  ): Promise<void> {
    try {
      await this.client.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "" },
          _meta: {
            anyharness: {
              schemaVersion: ANYHARNESS_SCHEMA_VERSION,
              transcriptEvent,
              ...payload,
            },
          },
        },
      });
    } catch (error) {
      this.logger.error(`[anyharness] failed to send ${transcriptEvent} notification:`, error);
    }
  }

  /** Records the transcript path reported by hooks and (re)starts the tailer. */
  private noteTranscriptPath(sessionId: string, transcriptPath: string): void {
    const session = this.sessions[sessionId];
    if (!session) {
      // Hooks can fire before createSession() registers the session.
      this.pendingTranscriptPaths.set(sessionId, transcriptPath);
      return;
    }
    const ah = session.anyharness;
    if (ah.transcriptPath === transcriptPath && ah.tailer) {
      return;
    }
    if (ah.tailer && ah.transcriptPath !== transcriptPath) {
      ah.tailer.dispose();
      ah.tailer = null;
    }
    ah.transcriptPath = transcriptPath;
    this.ensureTranscriptTailer(sessionId, session);
  }

  /**
   * Starts the transcript tailer that observes native goal_status attachments
   * (goal arming sentinels, evaluations, met/failed transitions).
   */
  private ensureTranscriptTailer(sessionId: string, session: Session): void {
    const ah = session.anyharness;
    if (ah.tailer) {
      return;
    }
    const transcriptPath =
      ah.transcriptPath ??
      this.pendingTranscriptPaths.get(sessionId) ??
      computeTranscriptPath(CLAUDE_CONFIG_DIR, session.cwd, sessionId);
    this.pendingTranscriptPaths.delete(sessionId);
    ah.transcriptPath = transcriptPath;

    if (!ah.tailFromStart && !ah.goal) {
      // Resumed/forked sessions tail from EOF, but a native goal survives
      // --resume — seed the mirror from the last goal_status row already in
      // the transcript so goal/get reconciles without a fresh arm.
      const last = readLastGoalStatus(transcriptPath);
      const lastKind = last ? classifyGoalStatus(last) : null;
      if (last && (lastKind === "armed" || lastKind === "progress")) {
        ah.goal = {
          objective: last.condition ?? "",
          status: "active",
          nativeStatus: lastKind === "armed" ? "armed" : "not_met",
          metReason: last.reason ?? null,
          iterations: last.iterations ?? null,
          tokensUsed: last.tokens ?? null,
          timeUsedSeconds: typeof last.durationMs === "number" ? last.durationMs / 1000 : null,
          updatedAtMs: Date.now(),
        };
      }
    }

    const tailer = new TranscriptTailer(
      transcriptPath,
      (row) => this.handleTranscriptRow(sessionId, row),
      this.logger,
      { fromStart: ah.tailFromStart },
    );
    ah.tailer = tailer;
    tailer.start();
  }

  /**
   * Attributes a transcript-observed cron fire to the armed loop whose prompt it
   * replays, advancing that loop's fire bookkeeping and emitting `loop_fired`.
   * Matched via `matchLoopForWake` so an ambiguous/unmatched injected prompt
   * (e.g. a goal continuation, or the `/loop` help injection that slips through)
   * credits nothing rather than corrupting a loop's count.
   */
  private recordLoopFireFromTranscript(sessionId: string, prompt: string): void {
    const session = this.sessions[sessionId];
    if (!session) {
      return;
    }
    const ah = session.anyharness;
    const loop = matchLoopForWake(activeLoops(ah), prompt);
    if (!loop) {
      return;
    }
    const now = Date.now();
    loop.fireCount += 1;
    loop.lastFiredAtMs = now;
    loop.updatedAtMs = now;
    this.logger.log(
      `[anyharness] loop fire (transcript): loopId=${loop.loopId} fireCount=${loop.fireCount}`,
    );
    void this.sendAnyharnessEvent(sessionId, "loop_fired", {
      loop: loopWireFromState(loop),
      loopId: loop.loopId,
    });
  }

  private handleTranscriptRow(sessionId: string, row: unknown): void {
    // A native cron/loop FIRE is recorded in the transcript (not on the SDK
    // stream) as a dequeued isMeta user row carrying the injected prompt — see
    // extractCronFirePrompt. This is the ONLY reliable fire signal for Claude
    // native loops, so fireCount/lastFiredAtMs and loop_fired are driven here.
    const cronFirePrompt = extractCronFirePrompt(row);
    if (cronFirePrompt) {
      this.recordLoopFireFromTranscript(sessionId, cronFirePrompt);
      return;
    }

    const goalStatus = extractGoalStatus(row);
    if (!goalStatus) {
      return;
    }
    const session = this.sessions[sessionId];
    if (!session) {
      return;
    }
    const ah = session.anyharness;
    const now = Date.now();
    const durationSeconds =
      typeof goalStatus.durationMs === "number" ? goalStatus.durationMs / 1000 : null;
    const objective = goalStatus.condition ?? ah.goal?.objective ?? "";
    const kind = classifyGoalStatus(goalStatus);
    this.logger.log(`[anyharness] transcript goal_status row (${kind}): ${safeJson(goalStatus)}`);

    switch (kind) {
      case "armed": {
        ah.goal = {
          objective,
          status: "active",
          nativeStatus: "armed",
          metReason: null,
          iterations: goalStatus.iterations ?? null,
          tokensUsed: goalStatus.tokens ?? null,
          timeUsedSeconds: durationSeconds,
          updatedAtMs: now,
        };
        void this.sendAnyharnessEvent(sessionId, "goal_updated", {
          goal: goalWireFromState(ah.goal),
        });
        break;
      }
      case "cleared": {
        ah.goal = {
          objective,
          status: "cleared",
          nativeStatus: "cleared",
          metReason: null,
          iterations: goalStatus.iterations ?? ah.goal?.iterations ?? null,
          tokensUsed: goalStatus.tokens ?? ah.goal?.tokensUsed ?? null,
          timeUsedSeconds: durationSeconds ?? ah.goal?.timeUsedSeconds ?? null,
          updatedAtMs: now,
        };
        void this.sendAnyharnessEvent(sessionId, "goal_cleared", {});
        break;
      }
      case "met": {
        ah.goal = {
          objective,
          status: "met",
          nativeStatus: "met",
          metReason: goalStatus.reason ?? null,
          iterations: goalStatus.iterations ?? null,
          tokensUsed: goalStatus.tokens ?? null,
          timeUsedSeconds: durationSeconds,
          updatedAtMs: now,
        };
        void this.sendAnyharnessEvent(sessionId, "goal_met", { goal: goalWireFromState(ah.goal) });
        break;
      }
      case "failed": {
        ah.goal = {
          objective,
          status: "failed",
          nativeStatus: "failed",
          metReason: goalStatus.reason ?? null,
          iterations: goalStatus.iterations ?? null,
          tokensUsed: goalStatus.tokens ?? null,
          timeUsedSeconds: durationSeconds,
          updatedAtMs: now,
        };
        void this.sendAnyharnessEvent(sessionId, "goal_updated", {
          goal: goalWireFromState(ah.goal),
        });
        break;
      }
      case "progress": {
        // Evaluation that did not meet the goal — progress update.
        ah.goal = {
          objective,
          status: "active",
          nativeStatus: "not_met",
          metReason: goalStatus.reason ?? null,
          iterations: goalStatus.iterations ?? null,
          tokensUsed: goalStatus.tokens ?? null,
          timeUsedSeconds: durationSeconds,
          updatedAtMs: now,
        };
        void this.sendAnyharnessEvent(sessionId, "goal_updated", {
          goal: goalWireFromState(ah.goal),
        });
        break;
      }
      default:
        unreachable(kind, this.logger);
        break;
    }

    for (const watcher of [...ah.goalRowWatchers]) {
      watcher(goalStatus);
    }
  }

  async cancel(params: CancelNotification): Promise<void> {
    const session = this.sessions[params.sessionId];
    if (!session) {
      throw new Error("Session not found");
    }
    session.cancelled = true;
    for (const [, pending] of session.pendingMessages) {
      pending.resolve(true);
    }
    session.pendingMessages.clear();
    await session.query.interrupt();
  }

  async unstable_closeSession(params: CloseSessionRequest): Promise<CloseSessionResponse> {
    const session = this.sessions[params.sessionId];
    if (!session) {
      throw new Error("Session not found");
    }
    await this.cancel({ sessionId: params.sessionId });

    session.settingsManager.dispose();
    session.anyharness.tailer?.dispose();
    session.abortController.abort();
    delete this.sessions[params.sessionId];

    return {};
  }

  async unstable_setSessionModel(
    params: SetSessionModelRequest,
  ): Promise<SetSessionModelResponse | void> {
    const session = this.sessions[params.sessionId];
    if (!session) {
      throw new Error("Session not found");
    }
    await session.query.setModel(params.modelId);
    session.models = {
      ...session.models,
      currentModelId: params.modelId,
    };
    await this.emitConfigOptionsUpdate(params.sessionId);
  }

  async setSessionMode(params: SetSessionModeRequest): Promise<SetSessionModeResponse> {
    const session = this.sessions[params.sessionId];
    if (!session) {
      throw new Error("Session not found");
    }

    await this.applySessionMode(params.sessionId, params.modeId);
    session.modes = {
      ...session.modes,
      currentModeId: params.modeId,
    };
    await this.emitConfigOptionsUpdate(params.sessionId);
    return {};
  }

  async setSessionConfigOption(
    params: SetSessionConfigOptionRequest,
  ): Promise<SetSessionConfigOptionResponse> {
    const session = this.sessions[params.sessionId];
    if (!session) {
      throw new Error("Session not found");
    }
    if (typeof params.value !== "string") {
      throw new Error(`Invalid value for config option ${params.configId}: ${params.value}`);
    }

    const option = session.configOptions.find((o) => o.id === params.configId);
    if (!option) {
      throw new Error(`Unknown config option: ${params.configId}`);
    }

    const allValues =
      "options" in option && Array.isArray(option.options)
        ? option.options.flatMap((o) => ("options" in o ? o.options : [o]))
        : [];
    let validValue = allValues.find((o) => o.value === params.value);

    // For model options, fall back to resolveModelPreference when the exact
    // value doesn't match.  This lets callers use human-friendly aliases like
    // "opus" or "sonnet" instead of full model IDs like "claude-opus-4-6".
    if (!validValue && params.configId === "model") {
      const modelInfos: ModelInfo[] = allValues.map((o) => ({
        value: o.value,
        displayName: o.name,
        description: o.description ?? "",
      }));
      const resolved = resolveModelPreference(modelInfos, params.value);
      if (resolved) {
        validValue = allValues.find((o) => o.value === resolved.value);
      }
    }

    if (!validValue) {
      throw new Error(`Invalid value for config option ${params.configId}: ${params.value}`);
    }

    // Use the canonical option value so downstream code always receives the
    // model ID rather than the caller-supplied alias.
    const resolvedValue = validValue.value;

    switch (params.configId) {
      case "mode":
        await this.applySessionMode(params.sessionId, resolvedValue);
        session.modes = {
          ...session.modes,
          currentModeId: resolvedValue,
        };
        await this.client.sessionUpdate({
          sessionId: params.sessionId,
          update: {
            sessionUpdate: "current_mode_update",
            currentModeId: resolvedValue,
          },
        });
        break;
      case "model":
        await session.query.setModel(resolvedValue);
        session.models = {
          ...session.models,
          currentModelId: resolvedValue,
        };
        break;
      case "thinking":
        await session.query.applyFlagSettings({
          alwaysThinkingEnabled: resolvedValue === "on",
        });
        session.liveSettings = {
          ...session.liveSettings,
          alwaysThinkingEnabled: resolvedValue === "on",
        };
        break;
      case "effort":
        await session.query.applyFlagSettings({
          effortLevel: resolvedValue as LiveEffortLevel,
        });
        session.liveSettings = {
          ...session.liveSettings,
          effortLevel: resolvedValue as LiveEffortLevel,
        };
        break;
      case "fast_mode":
        await session.query.applyFlagSettings({
          fastMode: resolvedValue === "on",
        });
        session.liveSettings = {
          ...session.liveSettings,
          fastMode: resolvedValue === "on",
        };
        break;
      default:
        throw new Error(`Unknown config option: ${params.configId}`);
    }

    const configOptions = await this.rebuildSessionConfigOptions(session);
    return { configOptions };
  }

  private async applySessionMode(sessionId: string, modeId: string): Promise<void> {
    switch (modeId) {
      case "default":
      case "acceptEdits":
      case "bypassPermissions":
      case "dontAsk":
      case "plan":
        break;
      default:
        throw new Error("Invalid Mode");
    }
    try {
      await this.sessions[sessionId].query.setPermissionMode(modeId);
    } catch (error) {
      if (error instanceof Error) {
        if (!error.message) {
          error.message = "Invalid Mode";
        }
        throw error;
      } else {
        // eslint-disable-next-line preserve-caught-error
        throw new Error("Invalid Mode");
      }
    }
  }

  private async replaySessionHistory(sessionId: string): Promise<void> {
    const toolUseCache: ToolUseCache = {};
    const messages = await getSessionMessages(sessionId);

    for (const message of messages) {
      for (const notification of toAcpNotifications(
        // @ts-expect-error - untyped in SDK but we handle all of these
        message.message.content,
        // @ts-expect-error - untyped in SDK but we handle all of these
        message.message.role,
        sessionId,
        toolUseCache,
        this.client,
        this.logger,
        {
          registerHooks: false,
          clientCapabilities: this.clientCapabilities,
          cwd: this.sessions[sessionId]?.cwd,
        },
      )) {
        await this.client.sessionUpdate(notification);
      }
    }
  }

  async readTextFile(params: ReadTextFileRequest): Promise<ReadTextFileResponse> {
    const response = await this.client.readTextFile(params);
    return response;
  }

  async writeTextFile(params: WriteTextFileRequest): Promise<WriteTextFileResponse> {
    const response = await this.client.writeTextFile(params);
    return response;
  }

  canUseTool(sessionId: string): CanUseTool {
    return async (toolName, toolInput, { signal, suggestions, toolUseID }) => {
      const supportsTerminalOutput = this.clientCapabilities?._meta?.["terminal_output"] === true;
      const session = this.sessions[sessionId];
      if (!session) {
        return {
          behavior: "deny",
          message: "Session not found",
        };
      }

      if (toolName === "ExitPlanMode") {
        const options = [
          {
            kind: "allow_always",
            name: "Yes, and auto-accept edits",
            optionId: "acceptEdits",
          },
          { kind: "allow_once", name: "Yes, and manually approve edits", optionId: "default" },
          { kind: "reject_once", name: "No, keep planning", optionId: "plan" },
        ];
        if (ALLOW_BYPASS) {
          options.unshift({
            kind: "allow_always",
            name: "Yes, and bypass permissions",
            optionId: "bypassPermissions",
          });
        }

        const response = await this.client.requestPermission({
          options,
          sessionId,
          toolCall: {
            toolCallId: toolUseID,
            rawInput: toolInput,
            ...toolInfoFromToolUse(
              { name: toolName, input: toolInput, id: toolUseID },
              supportsTerminalOutput,
              session?.cwd,
            ),
          },
        });

        if (signal.aborted || response.outcome?.outcome === "cancelled") {
          throw new Error("Tool use aborted");
        }
        if (
          response.outcome?.outcome === "selected" &&
          (response.outcome.optionId === "default" ||
            response.outcome.optionId === "acceptEdits" ||
            response.outcome.optionId === "bypassPermissions")
        ) {
          await this.client.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "current_mode_update",
              currentModeId: response.outcome.optionId,
            },
          });
          session.modes = {
            ...session.modes,
            currentModeId: response.outcome.optionId,
          };
          await this.emitConfigOptionsUpdate(sessionId);

          return {
            behavior: "allow",
            updatedInput: toolInput,
            updatedPermissions: suggestions ?? [
              { type: "setMode", mode: response.outcome.optionId, destination: "session" },
            ],
          };
        } else {
          return {
            behavior: "deny",
            message: "User rejected request to exit plan mode.",
          };
        }
      }

      if (session.modes.currentModeId === "bypassPermissions") {
        return {
          behavior: "allow",
          updatedInput: toolInput,
          updatedPermissions: suggestions ?? [
            { type: "addRules", rules: [{ toolName }], behavior: "allow", destination: "session" },
          ],
        };
      }

      const response = await this.client.requestPermission({
        options: [
          {
            kind: "allow_always",
            name: "Always Allow",
            optionId: "allow_always",
          },
          { kind: "allow_once", name: "Allow", optionId: "allow" },
          { kind: "reject_once", name: "Reject", optionId: "reject" },
        ],
        sessionId,
        toolCall: {
          toolCallId: toolUseID,
          rawInput: toolInput,
          ...toolInfoFromToolUse(
            { name: toolName, input: toolInput, id: toolUseID },
            supportsTerminalOutput,
            session?.cwd,
          ),
        },
      });
      if (signal.aborted || response.outcome?.outcome === "cancelled") {
        throw new Error("Tool use aborted");
      }
      if (
        response.outcome?.outcome === "selected" &&
        (response.outcome.optionId === "allow" || response.outcome.optionId === "allow_always")
      ) {
        // If Claude Code has suggestions, it will update their settings already
        if (response.outcome.optionId === "allow_always") {
          return {
            behavior: "allow",
            updatedInput: toolInput,
            updatedPermissions: suggestions ?? [
              {
                type: "addRules",
                rules: [{ toolName }],
                behavior: "allow",
                destination: "session",
              },
            ],
          };
        }
        return {
          behavior: "allow",
          updatedInput: toolInput,
        };
      } else {
        return {
          behavior: "deny",
          message: "User refused permission to run tool",
        };
      }
    };
  }

  private async sendAvailableCommandsUpdate(sessionId: string): Promise<void> {
    const session = this.sessions[sessionId];
    if (!session) return;
    const commands = await session.query.supportedCommands();
    await this.client.sessionUpdate({
      sessionId,
      update: {
        sessionUpdate: "available_commands_update",
        availableCommands: getAvailableSlashCommands(commands),
      },
    });
  }

  private async rebuildSessionConfigOptions(session: Session): Promise<SessionConfigOption[]> {
    const settings = await this.getNormalizedSessionSettings(session);
    const configOptions = buildConfigOptions(
      session.modes,
      session.models,
      session.modelCapabilitiesById,
      settings,
    );
    session.liveSettings = settings;
    session.configOptions = configOptions;
    return configOptions;
  }

  private async emitConfigOptionsUpdate(sessionId: string): Promise<void> {
    const session = this.sessions[sessionId];
    if (!session) return;

    const configOptions = await this.rebuildSessionConfigOptions(session);

    await this.client.sessionUpdate({
      sessionId,
      update: {
        sessionUpdate: "config_option_update",
        configOptions,
      },
    });
  }

  private async getNormalizedSessionSettings(session: Session): Promise<Settings> {
    let settings = { ...session.liveSettings };
    const currentCapabilities = session.modelCapabilitiesById[session.models.currentModelId];
    const supportedEffortLevels = getLiveEffortLevels(currentCapabilities);

    if (
      supportedEffortLevels.length === 0 ||
      typeof session.query.applyFlagSettings !== "function"
    ) {
      return settings;
    }

    if (
      settings.effortLevel !== undefined &&
      supportedEffortLevels.includes(settings.effortLevel as LiveEffortLevel)
    ) {
      return settings;
    }

    const preferredEffortLevel = supportedEffortLevels.includes("high")
      ? "high"
      : supportedEffortLevels[0];
    await session.query.applyFlagSettings({
      effortLevel: preferredEffortLevel,
    });
    settings = {
      ...settings,
      effortLevel: preferredEffortLevel,
    };
    session.liveSettings = settings;
    return settings;
  }

  private async getOrCreateSession(params: {
    sessionId: string;
    cwd: string;
    mcpServers?: NewSessionRequest["mcpServers"];
    _meta?: NewSessionRequest["_meta"];
  }): Promise<NewSessionResponse> {
    const existingSession = this.sessions[params.sessionId];
    if (existingSession) {
      return {
        sessionId: params.sessionId,
        modes: existingSession.modes,
        models: existingSession.models,
        configOptions: existingSession.configOptions,
      };
    }

    const response = await this.createSession(
      {
        cwd: params.cwd,
        mcpServers: params.mcpServers ?? [],
        _meta: params._meta,
      },
      {
        resume: params.sessionId,
      },
    );

    return {
      sessionId: response.sessionId,
      modes: response.modes,
      models: response.models,
      configOptions: response.configOptions,
    };
  }

  private async createSession(
    params: NewSessionRequest,
    creationOpts: { resume?: string; forkSession?: boolean } = {},
  ): Promise<NewSessionResponse> {
    // We want to create a new session id unless it is resume,
    // but not resume + forkSession.
    let sessionId;
    if (creationOpts.forkSession) {
      sessionId = randomUUID();
    } else if (creationOpts.resume) {
      sessionId = creationOpts.resume;
    } else {
      sessionId = randomUUID();
    }

    const input = new Pushable<SDKUserMessage>();

    const settingsManager = new SettingsManager(params.cwd, {
      logger: this.logger,
    });
    await settingsManager.initialize();

    const mcpServers: Record<string, McpServerConfig> = {};
    if (Array.isArray(params.mcpServers)) {
      for (const server of params.mcpServers) {
        if ("type" in server && (server.type === "http" || server.type === "sse")) {
          // HTTP or SSE type MCP server
          mcpServers[server.name] = {
            type: server.type,
            url: server.url,
            headers: server.headers
              ? Object.fromEntries(server.headers.map((e) => [e.name, e.value]))
              : undefined,
          };
        } else {
          // Stdio type MCP server (with or without explicit type field)
          mcpServers[server.name] = {
            type: "stdio",
            command: server.command,
            args: server.args,
            env: server.env
              ? Object.fromEntries(server.env.map((e) => [e.name, e.value]))
              : undefined,
          };
        }
      }
    }

    let systemPrompt: Options["systemPrompt"] = { type: "preset", preset: "claude_code" };
    if (params._meta?.systemPrompt) {
      const customPrompt = params._meta.systemPrompt;
      if (typeof customPrompt === "string") {
        systemPrompt = customPrompt;
      } else if (
        typeof customPrompt === "object" &&
        "append" in customPrompt &&
        typeof customPrompt.append === "string"
      ) {
        systemPrompt.append = customPrompt.append;
      }
    }

    const permissionMode = resolvePermissionMode(
      settingsManager.getSettings().permissions?.defaultMode,
    );

    // Extract options from _meta if provided
    const sessionMeta = params._meta as NewSessionMeta | undefined;
    const userProvidedOptions = sessionMeta?.claudeCode?.options;

    // Configure thinking tokens from environment variable
    const maxThinkingTokens = process.env.MAX_THINKING_TOKENS
      ? parseInt(process.env.MAX_THINKING_TOKENS, 10)
      : undefined;

    // Disable this for now, not a great way to expose this over ACP at the moment (in progress work so we can revisit)
    const disallowedTools = ["AskUserQuestion"];

    // Resolve which built-in tools to expose.
    // Explicit tools array from _meta.claudeCode.options takes precedence.
    // disableBuiltInTools is a legacy shorthand for tools: [] — kept for
    // backward compatibility but callers should prefer the tools array.
    const tools: Options["tools"] =
      userProvidedOptions?.tools ??
      (params._meta?.disableBuiltInTools === true ? [] : { type: "preset", preset: "claude_code" });

    const abortController = userProvidedOptions?.abortController || new AbortController();

    const options: Options = {
      systemPrompt,
      settingSources: ["user", "project", "local"],
      ...(maxThinkingTokens !== undefined && { maxThinkingTokens }),
      ...userProvidedOptions,
      env: {
        ...process.env,
        ...userProvidedOptions?.env,
        ...createEnvForGateway(this.gatewayAuthMeta),
        // Opt-in to session state events like when the agent is idle
        CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS: "1",
      },
      // Override certain fields that must be controlled by ACP
      cwd: params.cwd,
      includePartialMessages: true,
      mcpServers: { ...(userProvidedOptions?.mcpServers || {}), ...mcpServers },
      // If we want bypassPermissions to be an option, we have to allow it here.
      // But it doesn't work in root mode, so we only activate it if it will work.
      allowDangerouslySkipPermissions: ALLOW_BYPASS,
      permissionMode,
      canUseTool: this.canUseTool(sessionId),
      // note: although not documented by the types, passing an absolute path
      // here works to find zed's managed node version.
      executable: isStaticBinary() ? undefined : (process.execPath as any),
      ...(process.env.CLAUDE_CODE_EXECUTABLE
        ? { pathToClaudeCodeExecutable: process.env.CLAUDE_CODE_EXECUTABLE }
        : isStaticBinary()
          ? { pathToClaudeCodeExecutable: await claudeCliPath() }
          : {}),
      extraArgs: {
        ...userProvidedOptions?.extraArgs,
        "replay-user-messages": "",
      },
      disallowedTools: [...(userProvidedOptions?.disallowedTools || []), ...disallowedTools],
      tools,
      hooks: {
        ...userProvidedOptions?.hooks,
        SessionStart: [
          ...(userProvidedOptions?.hooks?.SessionStart || []),
          {
            hooks: [
              async (input: any) => {
                // Capture the transcript path for the anyharness goal tailer.
                if (typeof input.transcript_path === "string") {
                  this.noteTranscriptPath(sessionId, input.transcript_path);
                }
                // SessionStart carries session_crons too — reconcile the loop
                // mirror so a resumed session's re-armed crons heal on attach.
                if (input.session_crons !== undefined) {
                  await this.handleSessionCrons(sessionId, input.session_crons);
                }
                return { continue: true };
              },
            ],
          },
        ],
        PostToolUse: [
          ...(userProvidedOptions?.hooks?.PostToolUse || []),
          {
            hooks: [
              createPostToolUseHook(this.logger, {
                onTranscriptPath: (transcriptPath) =>
                  this.noteTranscriptPath(sessionId, transcriptPath),
                onCronTool: (toolName, toolInput, toolResponse) =>
                  this.handleCronTool(sessionId, toolName, toolInput, toolResponse),
                onSessionCrons: (crons) => this.handleSessionCrons(sessionId, crons),
                onEnterPlanMode: async () => {
                  await this.client.sessionUpdate({
                    sessionId,
                    update: {
                      sessionUpdate: "current_mode_update",
                      currentModeId: "plan",
                    },
                  });
                  const session = this.sessions[sessionId];
                  if (session) {
                    session.modes = {
                      ...session.modes,
                      currentModeId: "plan",
                    };
                  }
                  await this.emitConfigOptionsUpdate(sessionId);
                },
              }),
            ],
          },
        ],
      },
      ...creationOpts,
      abortController,
    };

    options.additionalDirectories = [
      ...(userProvidedOptions?.additionalDirectories ?? []),
      ...(sessionMeta?.additionalRoots ?? []),
    ];

    if (creationOpts?.resume === undefined || creationOpts?.forkSession) {
      // Set our own session id if not resuming an existing session.
      options.sessionId = sessionId;
    }

    // Handle abort controller from meta options
    if (abortController?.signal.aborted) {
      throw new Error("Cancelled");
    }

    const q = query({
      prompt: input,
      options,
    }) as MutableQuery;

    let initializationResult;
    try {
      initializationResult = await q.initializationResult();
    } catch (error) {
      if (
        creationOpts.resume &&
        error instanceof Error &&
        error.message === "Query closed before response received"
      ) {
        throw RequestError.resourceNotFound(sessionId);
      }
      throw error;
    }

    if (
      shouldHideClaudeAuth() &&
      initializationResult.account.subscriptionType &&
      !this.gatewayAuthMeta
    ) {
      throw RequestError.authRequired(
        undefined,
        "This integration does not support using claude.ai subscriptions.",
      );
    }

    const { state: models, capabilitiesById } = await getAvailableModels(
      q,
      initializationResult.models,
      settingsManager,
    );

    const availableModes = [
      {
        id: "default",
        name: "Default",
        description: "Standard behavior, prompts for dangerous operations",
      },
      {
        id: "acceptEdits",
        name: "Accept Edits",
        description: "Auto-accept file edit operations",
      },
      {
        id: "plan",
        name: "Plan Mode",
        description: "Planning mode, no actual tool execution",
      },
      {
        id: "dontAsk",
        name: "Don't Ask",
        description: "Don't prompt for permissions, deny if not pre-approved",
      },
    ];
    // Only works in non-root mode
    if (ALLOW_BYPASS) {
      availableModes.push({
        id: "bypassPermissions",
        name: "Bypass Permissions",
        description: "Bypass all permission checks",
      });
    }

    const modes = {
      currentModeId: permissionMode,
      availableModes,
    };

    const session: Session = {
      query: q,
      input: input,
      cancelled: false,
      cwd: params.cwd,
      settingsManager,
      accumulatedUsage: {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      },
      modes,
      models,
      modelCapabilitiesById: capabilitiesById,
      liveSettings: await getQuerySettings(q),
      configOptions: [],
      promptRunning: false,
      pendingMessages: new Map(),
      nextPendingOrder: 0,
      abortController,
      // Fresh sessions tail their transcript from the start; resumed/forked
      // sessions only from EOF (their transcript already contains history).
      anyharness: newAnyharnessSessionState(creationOpts.resume === undefined),
    };
    const configOptions = await this.rebuildSessionConfigOptions(session);
    this.sessions[sessionId] = session;

    // If a hook already reported the transcript path (SessionStart fires
    // during initialization), start the goal tailer now. Otherwise it starts
    // when the path is first reported or when a goal is set.
    if (this.pendingTranscriptPaths.has(sessionId)) {
      this.ensureTranscriptTailer(sessionId, session);
    }
    // Drain idle-time SDK messages (injected instructions, cron wake turns).
    this.startIdlePump(sessionId);

    return {
      sessionId,
      models,
      modes,
      configOptions,
    };
  }
}

function sessionUsage(session: Session) {
  return {
    inputTokens: session.accumulatedUsage.inputTokens,
    outputTokens: session.accumulatedUsage.outputTokens,
    cachedReadTokens: session.accumulatedUsage.cachedReadTokens,
    cachedWriteTokens: session.accumulatedUsage.cachedWriteTokens,
    totalTokens:
      session.accumulatedUsage.inputTokens +
      session.accumulatedUsage.outputTokens +
      session.accumulatedUsage.cachedReadTokens +
      session.accumulatedUsage.cachedWriteTokens,
  };
}

function createEnvForGateway(gatewayMeta?: GatewayAuthMeta) {
  if (!gatewayMeta) {
    return {};
  }
  return {
    ANTHROPIC_BASE_URL: gatewayMeta.gateway.baseUrl,
    ANTHROPIC_CUSTOM_HEADERS: Object.entries(gatewayMeta.gateway.headers)
      .map(([key, value]) => `${key}: ${value}`)
      .join("\n"),
    ANTHROPIC_AUTH_TOKEN: "", // Must be specified to bypass claude login requirement
  };
}

async function getQuerySettings(query: MutableQuery): Promise<Settings> {
  if (typeof query.getSettings !== "function") {
    return {};
  }

  return await query.getSettings();
}

function buildConfigOptions(
  modes: SessionModeState,
  models: SessionModelState,
  modelCapabilitiesById: ModelCapabilitiesById,
  settings: Settings,
): SessionConfigOption[] {
  const configOptions: SessionConfigOption[] = [
    {
      id: "mode",
      name: "Mode",
      description: "Session permission mode",
      category: "mode",
      type: "select",
      currentValue: modes.currentModeId,
      options: modes.availableModes.map((m) => ({
        value: m.id,
        name: m.name,
        description: m.description,
      })),
    },
    {
      id: "model",
      name: "Model",
      description: "AI model to use",
      category: "model",
      type: "select",
      currentValue: models.currentModelId,
      options: models.availableModels.map((m) => ({
        value: m.modelId,
        name: m.name,
        description: m.description ?? undefined,
      })),
    },
  ];

  const currentCapabilities = modelCapabilitiesById[models.currentModelId];
  if (supportsReasoningControls(currentCapabilities)) {
    configOptions.push({
      id: "thinking",
      name: "Thinking",
      description: "Enable or disable Claude thinking",
      category: "_thinking",
      type: "select",
      currentValue: settings.alwaysThinkingEnabled === false ? "off" : "on",
      options: [
        {
          value: "on",
          name: "On",
        },
        {
          value: "off",
          name: "Off",
        },
      ],
    });
  }

  const effortLevels = getLiveEffortLevels(currentCapabilities);
  if (effortLevels.length > 0) {
    const currentEffortLevel = effortLevels.includes(settings.effortLevel as LiveEffortLevel)
      ? (settings.effortLevel as LiveEffortLevel)
      : effortLevels.includes("high")
        ? "high"
        : effortLevels[0];
    configOptions.push({
      id: "effort",
      name: "Effort",
      description: "Reasoning depth",
      category: "thought_level",
      type: "select",
      currentValue: currentEffortLevel,
      options: effortLevels.map((value) => ({
        value,
        name: value[0].toUpperCase() + value.slice(1),
      })),
    });
  }

  if (currentCapabilities?.supportsFastMode) {
    configOptions.push({
      id: "fast_mode",
      name: "Fast Mode",
      description: "Favor faster responses",
      category: "_fast_mode",
      type: "select",
      currentValue: settings.fastMode === true ? "on" : "off",
      options: [
        {
          value: "off",
          name: "Off",
        },
        {
          value: "on",
          name: "On",
        },
      ],
    });
  }

  return configOptions;
}

// Claude Code CLI persists display strings like "opus[1m]" in settings,
// but the SDK model list uses IDs like "claude-opus-4-6-1m".
const MODEL_CONTEXT_HINT_PATTERN = /\[(\d+m)\]$/i;

function tokenizeModelPreference(model: string): { tokens: string[]; contextHint?: string } {
  const lower = model.trim().toLowerCase();
  const contextHint = lower.match(MODEL_CONTEXT_HINT_PATTERN)?.[1]?.toLowerCase();

  const normalized = lower.replace(MODEL_CONTEXT_HINT_PATTERN, " $1 ");
  const rawTokens = normalized.split(/[^a-z0-9]+/).filter(Boolean);
  const tokens = rawTokens
    .map((token) => {
      if (token === "opusplan") return "opus";
      if (token === "best" || token === "default") return "";
      return token;
    })
    .filter((token) => token && token !== "claude")
    .filter((token) => /[a-z]/.test(token) || token.endsWith("m"));

  return { tokens, contextHint };
}

function scoreModelMatch(model: ModelInfo, tokens: string[], contextHint?: string): number {
  const haystack = `${model.value} ${model.displayName}`.toLowerCase();
  let score = 0;
  for (const token of tokens) {
    if (haystack.includes(token)) {
      score += token === contextHint ? 3 : 1;
    }
  }
  return score;
}

function resolveModelPreference(models: ModelInfo[], preference: string): ModelInfo | null {
  const trimmed = preference.trim();
  if (!trimmed) return null;

  const lower = trimmed.toLowerCase();

  // Exact match on value or display name
  const directMatch = models.find(
    (model) =>
      model.value === trimmed ||
      model.value.toLowerCase() === lower ||
      model.displayName.toLowerCase() === lower,
  );
  if (directMatch) return directMatch;

  // Substring match
  const includesMatch = models.find((model) => {
    const value = model.value.toLowerCase();
    const display = model.displayName.toLowerCase();
    return value.includes(lower) || display.includes(lower) || lower.includes(value);
  });
  if (includesMatch) return includesMatch;

  // Tokenized matching for aliases like "opus[1m]"
  const { tokens, contextHint } = tokenizeModelPreference(trimmed);
  if (tokens.length === 0) return null;

  let bestMatch: ModelInfo | null = null;
  let bestScore = 0;
  for (const model of models) {
    const score = scoreModelMatch(model, tokens, contextHint);
    if (0 < score && (!bestMatch || bestScore < score)) {
      bestMatch = model;
      bestScore = score;
    }
  }

  return bestMatch;
}

async function getAvailableModels(
  query: Query,
  models: ModelInfo[],
  settingsManager: SettingsManager,
): Promise<AvailableModelsResult> {
  const settings = settingsManager.getSettings();

  let currentModel = models[0];

  if (settings.model) {
    const match = resolveModelPreference(models, settings.model);
    if (match) {
      currentModel = match;
    }
  }

  await query.setModel(currentModel.value);

  return {
    state: {
      availableModels: models.map((model) => ({
        modelId: model.value,
        name: model.displayName,
        description: model.description,
      })),
      currentModelId: currentModel.value,
    },
    capabilitiesById: Object.fromEntries(
      models.map((model) => [
        model.value,
        {
          supportsEffort: model.supportsEffort ?? false,
          supportedEffortLevels: [...(model.supportedEffortLevels ?? [])],
          supportsAdaptiveThinking: model.supportsAdaptiveThinking ?? false,
          supportsFastMode: model.supportsFastMode ?? false,
        } satisfies ModelCapabilities,
      ]),
    ),
  };
}

function supportsReasoningControls(capabilities?: ModelCapabilities): boolean {
  return !!(
    capabilities &&
    (capabilities.supportsAdaptiveThinking ||
      capabilities.supportsEffort ||
      capabilities.supportedEffortLevels.length > 0)
  );
}

function getLiveEffortLevels(capabilities?: ModelCapabilities): LiveEffortLevel[] {
  if (!capabilities) {
    return [];
  }

  return capabilities.supportedEffortLevels.filter(isLiveEffortLevel);
}

function isLiveEffortLevel(level: SupportedEffortLevel): level is LiveEffortLevel {
  return level === "low" || level === "medium" || level === "high";
}

function getAvailableSlashCommands(commands: SlashCommand[]): AvailableCommand[] {
  const UNSUPPORTED_COMMANDS = [
    "cost",
    "keybindings-help",
    "login",
    "logout",
    "output-style:new",
    "release-notes",
    "todos",
  ];

  return commands
    .map((command) => {
      const input = command.argumentHint
        ? {
            hint: Array.isArray(command.argumentHint)
              ? command.argumentHint.join(" ")
              : command.argumentHint,
          }
        : null;
      let name = command.name;
      if (command.name.endsWith(" (MCP)")) {
        name = `mcp:${name.replace(" (MCP)", "")}`;
      }
      return {
        name,
        description: command.description || "",
        input,
      };
    })
    .filter((command: AvailableCommand) => !UNSUPPORTED_COMMANDS.includes(command.name));
}

function formatUriAsLink(uri: string): string {
  try {
    if (uri.startsWith("file://")) {
      const path = uri.slice(7); // Remove "file://"
      const name = path.split("/").pop() || path;
      return `[@${name}](${uri})`;
    } else if (uri.startsWith("zed://")) {
      const parts = uri.split("/");
      const name = parts[parts.length - 1] || uri;
      return `[@${name}](${uri})`;
    }
    return uri;
  } catch {
    return uri;
  }
}

export function promptToClaude(prompt: PromptRequest): SDKUserMessage {
  const content: any[] = [];
  const context: any[] = [];

  for (const chunk of prompt.prompt) {
    switch (chunk.type) {
      case "text": {
        let text = chunk.text;
        // change /mcp:server:command args -> /server:command (MCP) args
        const mcpMatch = text.match(/^\/mcp:([^:\s]+):(\S+)(?:\s(.*))?$/);
        if (mcpMatch) {
          const [, server, command, args] = mcpMatch;
          text = `/${server}:${command} (MCP)${args ? ` ${args}` : ""}`;
        }
        content.push({ type: "text", text });
        break;
      }
      case "resource_link": {
        const formattedUri = formatUriAsLink(chunk.uri);
        content.push({
          type: "text",
          text: formattedUri,
        });
        break;
      }
      case "resource": {
        if ("text" in chunk.resource) {
          const formattedUri = formatUriAsLink(chunk.resource.uri);
          content.push({
            type: "text",
            text: formattedUri,
          });
          context.push({
            type: "text",
            text: `\n<context ref="${chunk.resource.uri}">\n${chunk.resource.text}\n</context>`,
          });
        }
        // Ignore blob resources (unsupported)
        break;
      }
      case "image":
        if (chunk.data) {
          content.push({
            type: "image",
            source: {
              type: "base64",
              data: chunk.data,
              media_type: chunk.mimeType,
            },
          });
        } else if (chunk.uri && chunk.uri.startsWith("http")) {
          content.push({
            type: "image",
            source: {
              type: "url",
              url: chunk.uri,
            },
          });
        }
        break;
      // Ignore audio and other unsupported types
      default:
        break;
    }
  }

  content.push(...context);

  return {
    type: "user",
    message: {
      role: "user",
      content: content,
    },
    session_id: prompt.sessionId,
    parent_tool_use_id: null,
  };
}

/**
 * Convert an SDKAssistantMessage (Claude) to a SessionNotification (ACP).
 * Only handles text, image, and thinking chunks for now.
 */
export function toAcpNotifications(
  content: string | ContentBlockParam[] | BetaContentBlock[] | BetaRawContentBlockDelta[],
  role: "assistant" | "user",
  sessionId: string,
  toolUseCache: ToolUseCache,
  client: AgentSideConnection,
  logger: Logger,
  options?: {
    registerHooks?: boolean;
    clientCapabilities?: ClientCapabilities;
    parentToolUseId?: string | null;
    cwd?: string;
  },
): SessionNotification[] {
  const registerHooks = options?.registerHooks !== false;
  const supportsTerminalOutput = options?.clientCapabilities?._meta?.["terminal_output"] === true;
  if (typeof content === "string") {
    const update: SessionNotification["update"] = {
      sessionUpdate: role === "assistant" ? "agent_message_chunk" : "user_message_chunk",
      content: {
        type: "text",
        text: content,
      },
    };

    if (options?.parentToolUseId) {
      update._meta = {
        ...update._meta,
        claudeCode: {
          ...(update._meta?.claudeCode || {}),
          parentToolUseId: options.parentToolUseId,
        },
      };
    }

    return [{ sessionId, update }];
  }

  const output = [];
  // Only handle the first chunk for streaming; extend as needed for batching
  for (const chunk of content) {
    let update: SessionNotification["update"] | null = null;
    switch (chunk.type) {
      case "text":
      case "text_delta":
        update = {
          sessionUpdate: role === "assistant" ? "agent_message_chunk" : "user_message_chunk",
          content: {
            type: "text",
            text: chunk.text,
          },
        };
        break;
      case "image":
        update = {
          sessionUpdate: role === "assistant" ? "agent_message_chunk" : "user_message_chunk",
          content: {
            type: "image",
            data: chunk.source.type === "base64" ? chunk.source.data : "",
            mimeType: chunk.source.type === "base64" ? chunk.source.media_type : "",
            uri: chunk.source.type === "url" ? chunk.source.url : undefined,
          },
        };
        break;
      case "thinking":
      case "thinking_delta":
        update = {
          sessionUpdate: "agent_thought_chunk",
          content: {
            type: "text",
            text: chunk.thinking,
          },
        };
        break;
      case "tool_use":
      case "server_tool_use":
      case "mcp_tool_use": {
        const alreadyCached = chunk.id in toolUseCache;
        toolUseCache[chunk.id] = chunk;
        if (chunk.name === "TodoWrite") {
          // @ts-expect-error - sometimes input is empty object
          if (Array.isArray(chunk.input.todos)) {
            update = {
              sessionUpdate: "plan",
              entries: planEntries(chunk.input as { todos: ClaudePlanEntry[] }),
            };
          }
        } else {
          // Only register hooks on first encounter to avoid double-firing
          if (registerHooks && !alreadyCached) {
            registerHookCallback(chunk.id, {
              onPostToolUseHook: async (toolUseId, toolInput, toolResponse) => {
                const toolUse = toolUseCache[toolUseId];
                if (toolUse) {
                  const editDiff =
                    toolUse.name === "Edit" ? toolUpdateFromEditToolResponse(toolResponse) : {};
                  const update: SessionNotification["update"] = {
                    _meta: {
                      claudeCode: {
                        toolResponse,
                        toolName: toolUse.name,
                      },
                    } satisfies ToolUpdateMeta,
                    toolCallId: toolUseId,
                    sessionUpdate: "tool_call_update",
                    ...editDiff,
                  };
                  await client.sessionUpdate({
                    sessionId,
                    update,
                  });
                } else {
                  logger.error(
                    `[claude-agent-acp] Got a tool response for tool use that wasn't tracked: ${toolUseId}`,
                  );
                }
              },
            });
          }

          let rawInput;
          try {
            rawInput = JSON.parse(JSON.stringify(chunk.input));
          } catch {
            // ignore if we can't turn it to JSON
          }

          if (alreadyCached) {
            // Second encounter (full assistant message after streaming) —
            // send as tool_call_update to refine the existing tool_call
            // rather than emitting a duplicate tool_call.
            update = {
              _meta: {
                claudeCode: {
                  toolName: chunk.name,
                },
              } satisfies ToolUpdateMeta,
              toolCallId: chunk.id,
              sessionUpdate: "tool_call_update",
              rawInput,
              ...toolInfoFromToolUse(chunk, supportsTerminalOutput, options?.cwd),
            };
          } else {
            // First encounter (streaming content_block_start or replay) —
            // send as tool_call with terminal_info for Bash tools.
            update = {
              _meta: {
                claudeCode: {
                  toolName: chunk.name,
                },
                ...(chunk.name === "Bash" && supportsTerminalOutput
                  ? { terminal_info: { terminal_id: chunk.id } }
                  : {}),
              } satisfies ToolUpdateMeta,
              toolCallId: chunk.id,
              sessionUpdate: "tool_call",
              rawInput,
              status: "pending",
              ...toolInfoFromToolUse(chunk, supportsTerminalOutput, options?.cwd),
            };
          }
        }
        break;
      }

      case "tool_result":
      case "tool_search_tool_result":
      case "web_fetch_tool_result":
      case "web_search_tool_result":
      case "code_execution_tool_result":
      case "bash_code_execution_tool_result":
      case "text_editor_code_execution_tool_result":
      case "mcp_tool_result": {
        const toolUse = toolUseCache[chunk.tool_use_id];
        if (!toolUse) {
          logger.error(
            `[claude-agent-acp] Got a tool result for tool use that wasn't tracked: ${chunk.tool_use_id}`,
          );
          break;
        }

        if (toolUse.name !== "TodoWrite") {
          const { _meta: toolMeta, ...toolUpdate } = toolUpdateFromToolResult(
            chunk,
            toolUseCache[chunk.tool_use_id],
            supportsTerminalOutput,
          );

          // When terminal output is supported, send terminal_output as a
          // separate notification to match codex-acp's streaming lifecycle:
          //   1. tool_call       → _meta.terminal_info  (already sent above)
          //   2. tool_call_update → _meta.terminal_output (sent here)
          //   3. tool_call_update → _meta.terminal_exit  (sent below with status)
          if (toolMeta?.terminal_output) {
            output.push({
              sessionId,
              update: {
                _meta: {
                  terminal_output: toolMeta.terminal_output,
                  ...(options?.parentToolUseId
                    ? { claudeCode: { parentToolUseId: options.parentToolUseId } }
                    : {}),
                },
                toolCallId: chunk.tool_use_id,
                sessionUpdate: "tool_call_update" as const,
              },
            });
          }

          update = {
            _meta: {
              claudeCode: {
                toolName: toolUse.name,
              },
              ...(toolMeta?.terminal_exit ? { terminal_exit: toolMeta.terminal_exit } : {}),
            } satisfies ToolUpdateMeta,
            toolCallId: chunk.tool_use_id,
            sessionUpdate: "tool_call_update",
            status: "is_error" in chunk && chunk.is_error ? "failed" : "completed",
            rawOutput: chunk.content,
            ...toolUpdate,
          };
        }
        break;
      }

      case "document":
      case "search_result":
      case "redacted_thinking":
      case "input_json_delta":
      case "citations_delta":
      case "signature_delta":
      case "container_upload":
      case "compaction":
      case "compaction_delta":
        break;

      default:
        unreachable(chunk, logger);
        break;
    }
    if (update) {
      if (options?.parentToolUseId) {
        update._meta = {
          ...update._meta,
          claudeCode: {
            ...(update._meta?.claudeCode || {}),
            parentToolUseId: options.parentToolUseId,
          },
        };
      }
      output.push({ sessionId, update });
    }
  }

  return output;
}

export function streamEventToAcpNotifications(
  message: SDKPartialAssistantMessage,
  sessionId: string,
  toolUseCache: ToolUseCache,
  client: AgentSideConnection,
  logger: Logger,
  options?: {
    clientCapabilities?: ClientCapabilities;
    cwd?: string;
  },
): SessionNotification[] {
  const event = message.event;
  switch (event.type) {
    case "content_block_start":
      return toAcpNotifications(
        [event.content_block],
        "assistant",
        sessionId,
        toolUseCache,
        client,
        logger,
        {
          clientCapabilities: options?.clientCapabilities,
          parentToolUseId: message.parent_tool_use_id,
          cwd: options?.cwd,
        },
      );
    case "content_block_delta":
      return toAcpNotifications(
        [event.delta],
        "assistant",
        sessionId,
        toolUseCache,
        client,
        logger,
        {
          clientCapabilities: options?.clientCapabilities,
          parentToolUseId: message.parent_tool_use_id,
          cwd: options?.cwd,
        },
      );
    // No content
    case "message_start":
    case "message_delta":
    case "message_stop":
    case "content_block_stop":
      return [];

    default:
      unreachable(event, logger);
      return [];
  }
}

export function runAcp() {
  const input = nodeToWebWritable(process.stdout);
  const output = nodeToWebReadable(process.stdin);

  const stream = ndJsonStream(input, output);
  new AgentSideConnection((client) => new ClaudeAcpAgent(client), stream);
}
