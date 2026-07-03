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
  PermissionOption,
  PromptRequest,
  PromptResponse,
  ReadTextFileRequest,
  ReadTextFileResponse,
  RequestError,
  ResumeSessionRequest,
  ResumeSessionResponse,
  SessionConfigOption,
  SessionModeState,
  SessionNotification,
  SetSessionConfigOptionRequest,
  SetSessionConfigOptionResponse,
  SetSessionModeRequest,
  SetSessionModeResponse,
  CloseSessionRequest,
  CloseSessionResponse,
  DeleteSessionRequest,
  DeleteSessionResponse,
  WriteTextFileRequest,
  WriteTextFileResponse,
  StopReason,
} from "@agentclientprotocol/sdk";
import {
  CanUseTool,
  deleteSession,
  getSessionMessages,
  listSessions,
  McpServerConfig,
  ModelInfo,
  ModelUsage,
  OnElicitation,
  Options,
  PermissionMode,
  PermissionResult,
  PermissionUpdate,
  Query,
  query,
  Settings,
  SDKAssistantMessageError,
  SDKMessage,
  SDKMessageOrigin,
  SDKPartialAssistantMessage,
  SDKUserMessage,
  SlashCommand,
  ThinkingConfig,
} from "@anthropic-ai/claude-agent-sdk";
import { ContentBlockParam } from "@anthropic-ai/sdk/resources";
import { BetaContentBlock, BetaRawContentBlockDelta } from "@anthropic-ai/sdk/resources/beta.mjs";
import { randomUUID } from "node:crypto";
import type { Stats } from "node:fs";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import packageJson from "../package.json" with { type: "json" };
import {
  applyAskElicitationResponse,
  askUserQuestionsToCreateRequest,
  createElicitationResponseToElicitResult,
  ElicitationSupport,
  extractAskUserQuestions,
  mcpElicitationToCreateRequest,
} from "./elicitation.js";
import {
  activeLoops,
  ANYHARNESS_CAPABILITIES,
  ANYHARNESS_SCHEMA_VERSION,
  AnyharnessSessionState,
  AnyharnessTranscriptEvent,
  classifyGoalStatus,
  computeTranscriptPath,
  extractGoalStatus,
  GoalStatusRow,
  GoalWire,
  goalWireFromState,
  LoopSchedule,
  LoopState,
  LoopWire,
  loopWireFromState,
  newAnyharnessSessionState,
  readLastGoalStatus,
  TranscriptTailer,
} from "./anyharness.js";
import { SettingsManager } from "./settings.js";
import {
  applyTaskCreate,
  applyTaskUpdate,
  ClaudePlanEntry,
  createPostToolUseHook,
  createTaskHook,
  parseTaskCreateOutput,
  planEntries,
  registerHookCallback,
  TaskState,
  taskStateToPlanEntries,
  toolInfoFromToolUse,
  toolUpdateFromDiffToolResponse,
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

type UsageSnapshot = {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens: number;
  cache_creation_input_tokens: number;
};

const ZERO_USAGE = Object.freeze({
  input_tokens: 0,
  output_tokens: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
});

const DEFAULT_CONTEXT_WINDOW = 200000;

/** Floor after `session/cancel` before the adapter forces the active prompt
 *  loop to return "cancelled". `query.interrupt()` normally makes the SDK
 *  yield a trailing idle within milliseconds, and the loop returns through its
 *  usual path — so this timer is armed and cleared, never fired, on healthy
 *  cancels. It only trips when the SDK is genuinely wedged (e.g. a
 *  `TaskOutput { block: true }` poll against a hung background task — issue
 *  #680) and never yields. The value is deliberately loose: it's an
 *  "obviously stuck" ceiling, not a guess at interrupt latency, so it can't
 *  pre-empt a slow-but-healthy interrupt. */
const DEFAULT_FORCE_CANCEL_GRACE_MS = 30_000;

/** Internal model-selection state. Mirrors the shape the ACP SDK exposed as
 *  `SessionModelState` before model selection moved entirely into
 *  `SessionConfigOption` (category "model"). Retained internally to track the
 *  current model and build the "model" config option. */
type SessionModelState = {
  availableModels: Array<{ modelId: string; name: string; description?: string }>;
  currentModelId: string;
};

type Session = {
  query: Query;
  input: Pushable<SDKUserMessage>;
  cancelled: boolean;
  cwd: string;
  /** Serialized snapshot of session-defining params (cwd, mcpServers) used to
   *  detect when loadSession/resumeSession is called with changed values. */
  sessionFingerprint: string;
  settingsManager: SettingsManager;
  accumulatedUsage: AccumulatedUsage;
  modes: SessionModeState;
  models: SessionModelState;
  modelInfos: ModelInfo[];
  configOptions: SessionConfigOption[];
  promptRunning: boolean;
  pendingMessages: Map<string, { resolve: (cancelled: boolean) => void; order: number }>;
  nextPendingOrder: number;
  abortController: AbortController;
  /** Per-turn signal the active prompt loop races `query.next()` against.
   *  Aborted by cancel() (after a grace period) to force the loop to return
   *  "cancelled" when the SDK is wedged and `query.next()` never yields again
   *  (issue #680). Distinct from `abortController`: this only wakes the loop;
   *  it does NOT touch the SDK query/subprocess. Undefined when no prompt is
   *  actively consuming the query. */
  cancelController?: AbortController;
  /** Pending grace-period timer that aborts `cancelController`. Cleared when
   *  the loop returns normally so the backstop never fires after a clean
   *  cancel. */
  forceCancelTimer?: ReturnType<typeof setTimeout>;
  emitRawSDKMessages: boolean | SDKMessageFilter[];
  /** Context window size of the last top-level assistant model, carried across
   *  prompts so mid-stream usage_update notifications report a correct `size`
   *  before the turn's first result message arrives. Defaults to
   *  DEFAULT_CONTEXT_WINDOW, refreshed from each result's modelUsage, and
   *  invalidated when the user switches the session's model. */
  contextWindowSize: number;
  /** Accumulated task list for the session, keyed by task ID. Task IDs are
   *  per-session, so this state must not be shared across sessions. */
  taskState: TaskState;
  /** Caches `tool_use` blocks by id so the matching `tool_result` can recover
   *  the tool name/input when mapping it to a `tool_call_update`. Per-session
   *  (tool_use ids are only unique within a session) and pruned at
   *  `tool_result` time so a long-running session doesn't accumulate every
   *  tool call for its whole lifetime. */
  toolUseCache: ToolUseCache;
  /** Maps the ACP `messageId` we expose to clients (see `messageIdForGrouping`)
   *  to the SDK message uuid that the Agent SDK's rewind/resume APIs key on
   *  (`Query.rewindFiles` takes a user-message uuid; `resumeSessionAt` takes an
   *  `SDKAssistantMessage.uuid`). For assistant turns the two differ — the ACP
   *  id is the Anthropic API message id (`msg_…`), available at `message_start`
   *  so streamed chunks can carry it, while the uuid only arrives on the
   *  consolidated message — so a client can only ask to rewind/fork by the id it
   *  was given, and we need this table to translate it back.
   *
   *  Populated as a byproduct of the message loop (the consolidated message
   *  carries both ids) and of `replaySessionHistory` on load, so no extra
   *  `getSessionMessages` read is needed at rewind time. Last-write-wins
   *  naturally yields the turn-boundary uuid when one `msg_…` spans several
   *  content-block messages.
   *
   *  NOT READ YET — recorded now so the mapping exists if/when we wire up
   *  fork/rewind. */
  messageIdToUuid: Map<string, string>;
  /** Whether fast mode is currently enabled for this session. Tracks the
   *  client-requested state so config option rebuilds (e.g. on model switch)
   *  can preserve it. Defaults to false (fast mode off). */
  fastModeEnabled?: boolean;
  /** Goal/loop state + transcript tailer for the anyharness GoalPort/LoopPort
   *  extension. */
  anyharness: AnyharnessSessionState;
  /**
   * In-flight query.next() shared between the prompt drain and the idle
   * pump so an interrupted pump never loses a pulled message.
   */
  pendingQueryNext?: Promise<IteratorResult<SDKMessage, void>> | null;
};

/** Compute a stable fingerprint of the session-defining params so we can
 *  detect when a loadSession/resumeSession call requires tearing down and
 *  recreating the underlying Query process.  MCP servers are sorted by name
 *  so that ordering differences don't trigger unnecessary recreations. */
function computeSessionFingerprint(params: {
  cwd: string;
  mcpServers?: NewSessionRequest["mcpServers"];
}): string {
  const servers = [...(params.mcpServers ?? [])].sort((a, b) => a.name.localeCompare(b.name));
  return JSON.stringify({ cwd: params.cwd, mcpServers: servers });
}

export type SDKMessageFilter = {
  type: string;
  subtype?: string;
  origin?: SDKMessageOrigin["kind"];
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
    /**
     * When set, raw SDK messages are emitted as extNotification("_claude/sdkMessage", message)
     * in addition to normal processing.
     * - true: emit all messages
     * - false/undefined: emit nothing (default)
     * - SDKMessageFilter[]: emit only messages matching at least one filter
     */
    emitRawSDKMessages?: boolean | SDKMessageFilter[];
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

type GatewayAuthRequest = AuthenticateRequest & { _meta?: GatewayAuthMeta };

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

export async function claudeCliPath(): Promise<string> {
  if (process.env.CLAUDE_CODE_EXECUTABLE) {
    return process.env.CLAUDE_CODE_EXECUTABLE;
  }
  // The SDK's CLI is a native binary shipped as a platform-specific optional
  // dependency of @anthropic-ai/claude-agent-sdk. Resolve via a require bound
  // to the SDK so nested installs are found even when npm doesn't hoist.
  const { createRequire } = await import("node:module");
  const req = createRequire(import.meta.resolve("@anthropic-ai/claude-agent-sdk"));
  const ext = process.platform === "win32" ? ".exe" : "";
  // On linux, both glibc and musl variants may be installed side-by-side
  // (e.g. bunx hydrates every optional dep), so picking one by trial is
  // unreliable: the wrong binary segfaults at runtime instead of failing to
  // spawn. Detect the runtime libc and prefer the matching variant, falling
  // back to the other only if the preferred one isn't installed.
  const candidates =
    process.platform === "linux"
      ? isMuslLibc()
        ? [
            `@anthropic-ai/claude-agent-sdk-linux-${process.arch}-musl/claude${ext}`,
            `@anthropic-ai/claude-agent-sdk-linux-${process.arch}/claude${ext}`,
          ]
        : [
            `@anthropic-ai/claude-agent-sdk-linux-${process.arch}/claude${ext}`,
            `@anthropic-ai/claude-agent-sdk-linux-${process.arch}-musl/claude${ext}`,
          ]
      : [`@anthropic-ai/claude-agent-sdk-${process.platform}-${process.arch}/claude${ext}`];
  for (const candidate of candidates) {
    try {
      return req.resolve(candidate);
    } catch {
      // try next candidate
    }
  }
  throw new Error(
    `Claude native binary not found for ${process.platform}-${process.arch}. ` +
      `Reinstall @anthropic-ai/claude-agent-sdk without --omit=optional, or set CLAUDE_CODE_EXECUTABLE.`,
  );
}

function isMuslLibc(): boolean {
  // process.report.getReport().header.glibcVersionRuntime is populated when
  // Node is dynamically linked against glibc, and absent on musl.
  const report = process.report?.getReport() as
    | { header?: { glibcVersionRuntime?: string } }
    | undefined;
  return !report?.header?.glibcVersionRuntime;
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

/** Best-effort extraction of the cron job id from CronCreate/CronDelete tool IO. */
function extractCronId(value: unknown): string | undefined {
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

// The Claude SDK persists local slash command invocations (e.g. `/model`) and
// their output as user messages in the session transcript, wrapping the
// payload in these XML-like markers that the CLI uses for its own display.
// The live prompt loop drops them; replay must strip them too or they leak
// into the UI on session/load.
const LOCAL_COMMAND_MARKERS = [
  "command-name",
  "command-message",
  "command-args",
  "local-command-stdout",
  "local-command-stderr",
].map((tag) => ({ open: `<${tag}>`, close: `</${tag}>` }));

// Single-pass scanner that removes each `<tag>…</tag>` marker (matching the
// nearest closing tag of the same name, like a lazy regex would).
function stripMarkerTags(text: string): string {
  const dead = new Set<string>();
  let result = "";
  let copiedUpTo = 0;
  let i = 0;
  while (i < text.length) {
    if (text[i] === "<") {
      const marker = LOCAL_COMMAND_MARKERS.find(
        (m) => !dead.has(m.open) && text.startsWith(m.open, i),
      );
      if (marker) {
        const end = text.indexOf(marker.close, i + marker.open.length);
        if (end !== -1) {
          result += text.slice(copiedUpTo, i);
          i = copiedUpTo = end + marker.close.length;
          continue;
        }
        // No closing marker remains anywhere ahead, and `indexOf` only ever
        // searches forward from here on, so stop treating this tag as an
        // opener — that avoids rescanning the tail for it on every match.
        dead.add(marker.open);
      }
    }
    i++;
  }
  return result + text.slice(copiedUpTo);
}

/**
 * Return user-message content with local-command marker tags removed, or
 * `null` if nothing meaningful remains (caller should skip the message).
 * Preserves real prose that's mixed in alongside the markers — e.g. a
 * message like `<command-name>…</command-name>hi` becomes `hi`.
 */
export function stripLocalCommandMetadata(content: unknown): unknown | null {
  if (typeof content === "string") {
    const stripped = stripMarkerTags(content);
    return stripped.trim() === "" ? null : stripped;
  }
  if (!Array.isArray(content)) return content;

  const kept: unknown[] = [];
  for (const block of content) {
    if (
      block &&
      typeof block === "object" &&
      "type" in block &&
      (block as { type: unknown }).type === "text" &&
      "text" in block &&
      typeof (block as { text: unknown }).text === "string"
    ) {
      const stripped = stripMarkerTags((block as { text: string }).text);
      if (stripped.trim() === "") continue;
      kept.push({ ...(block as object), text: stripped });
    } else {
      kept.push(block);
    }
  }
  if (kept.length === 0) return null;
  return kept;
}

export function isLocalCommandMetadata(content: unknown): boolean {
  return stripLocalCommandMetadata(content) === null;
}

const PERMISSION_MODE_ALIASES: Record<string, PermissionMode> = {
  auto: "auto",
  default: "default",
  acceptedits: "acceptEdits",
  dontask: "dontAsk",
  plan: "plan",
  bypasspermissions: "bypassPermissions",
  bypass: "bypassPermissions",
};

export function resolvePermissionMode(
  defaultMode?: unknown,
  logger: Logger = console,
): PermissionMode {
  if (defaultMode === undefined) {
    return "default";
  }

  if (typeof defaultMode !== "string") {
    logger.error("Ignoring permissions.defaultMode from settings: expected a string.");
    return "default";
  }

  const normalized = defaultMode.trim().toLowerCase();
  if (normalized === "") {
    logger.error("Ignoring permissions.defaultMode from settings: expected a non-empty string.");
    return "default";
  }

  const mapped = PERMISSION_MODE_ALIASES[normalized];
  if (!mapped) {
    logger.error(`Ignoring permissions.defaultMode from settings: unknown value '${defaultMode}'.`);
    return "default";
  }

  if (mapped === "bypassPermissions" && !ALLOW_BYPASS) {
    logger.error(
      "Ignoring permissions.defaultMode from settings: bypassPermissions is not available when running as root.",
    );
    return "default";
  }

  return mapped;
}

/**
 * Builds the label for the "Always Allow" permission option so the user can see
 * the exact scope they are committing to. Uses the SDK-provided suggestions
 * when available (e.g. `Bash(npm test:*)`) and falls back to naming the whole
 * tool so "Always Allow" is never a blank check without disclosure.
 */
export function describeAlwaysAllow(
  suggestions: PermissionUpdate[] | undefined,
  toolName: string,
): string {
  if (!suggestions || suggestions.length === 0) {
    return `Always Allow all ${toolName}`;
  }

  const ruleLabels: string[] = [];
  const directories: string[] = [];

  for (const update of suggestions) {
    if (update.type === "addRules" && update.behavior === "allow") {
      for (const rule of update.rules) {
        ruleLabels.push(
          rule.ruleContent ? `${rule.toolName}(${rule.ruleContent})` : `all ${rule.toolName}`,
        );
      }
    } else if (update.type === "addDirectories") {
      directories.push(...update.directories);
    }
  }

  const parts: string[] = [];
  if (ruleLabels.length > 0) {
    parts.push(ruleLabels.join(", "));
  }
  if (directories.length > 0) {
    parts.push(`access to ${directories.join(", ")}`);
  }

  if (parts.length === 0) {
    return `Always Allow all ${toolName}`;
  }

  return `Always Allow ${parts.join(" and ")}`;
}

// Implement the ACP Agent interface
export class ClaudeAcpAgent implements Agent {
  sessions: {
    [key: string]: Session;
  };
  client: AgentSideConnection;
  clientCapabilities?: ClientCapabilities;
  logger: Logger;
  gatewayAuthRequest?: GatewayAuthRequest;
  /** Grace period before a `session/cancel` forces a wedged prompt loop to
   *  return "cancelled". See {@link DEFAULT_FORCE_CANCEL_GRACE_MS}. Mutable so
   *  tests can shrink it. */
  forceCancelGraceMs: number = DEFAULT_FORCE_CANCEL_GRACE_MS;
  /** Transcript paths reported by PostToolUse hooks before the session was
   *  registered, replayed into the tailer once the session exists. */
  private pendingTranscriptPaths: Map<string, string> = new Map();

  constructor(client: AgentSideConnection, logger?: Logger) {
    this.sessions = {};
    this.client = client;
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

    const gatewayBedrockAuthMethod: AuthMethod = {
      id: "gateway-bedrock",
      name: "Custom model gateway",
      description: "Use a custom gateway to authenticate and access models",
      _meta: {
        gateway: {
          protocol: "bedrock",
        },
      },
    };

    const supportsTerminalAuth = request.clientCapabilities?.auth?.terminal === true;
    const supportsMetaTerminalAuth = request.clientCapabilities?._meta?.["terminal-auth"] === true;

    // Detect remote environments where the OAuth browser redirect to localhost
    // won't work. This matches the SDK's internal isRemote check. In these cases,
    // the `auth login` subcommand would fall back to a device-code-like manual
    // flow, which doesn't work well over ACP, so we offer the TUI login instead.
    const isRemote = !!(
      process.env.NO_BROWSER ||
      process.env.SSH_CONNECTION ||
      process.env.SSH_CLIENT ||
      process.env.SSH_TTY ||
      process.env.CLAUDE_CODE_REMOTE
    );
    const terminalAuthMethods: AuthMethod[] = [];

    if (isRemote) {
      const remoteLoginMethod: AuthMethod = {
        description: "Run `claude /login` in the terminal",
        name: "Log in with Claude",
        id: "claude-login",
        type: "terminal",
        args: ["--cli"],
      };

      if (supportsMetaTerminalAuth) {
        remoteLoginMethod._meta = {
          "terminal-auth": {
            command: process.execPath,
            args: [...process.argv.slice(1), "--cli"],
            label: "Claude Login",
          },
        };
      }

      if (!shouldHideClaudeAuth() && (supportsTerminalAuth || supportsMetaTerminalAuth)) {
        terminalAuthMethods.push(remoteLoginMethod);
      }
    } else {
      const claudeLoginMethod: AuthMethod = {
        description: "Use Claude subscription ",
        name: "Claude Subscription",
        id: "claude-ai-login",
        type: "terminal",
        args: ["--cli", "auth", "login", "--claudeai"],
      };

      const consoleLoginMethod: AuthMethod = {
        description: "Use Anthropic Console (API usage billing)",
        name: "Anthropic Console",
        id: "console-login",
        type: "terminal",
        args: ["--cli", "auth", "login", "--console"],
      };

      if (supportsMetaTerminalAuth) {
        const baseArgs = process.argv.slice(1);
        claudeLoginMethod._meta = {
          "terminal-auth": {
            command: process.execPath,
            args: [...baseArgs, "--cli", "auth", "login", "--claudeai"],
            label: "Claude Login",
          },
        };
        consoleLoginMethod._meta = {
          "terminal-auth": {
            command: process.execPath,
            args: [...baseArgs, "--cli", "auth", "login", "--console"],
            label: "Anthropic Console Login",
          },
        };
      }

      if (!shouldHideClaudeAuth() && (supportsTerminalAuth || supportsMetaTerminalAuth)) {
        terminalAuthMethods.push(claudeLoginMethod);
      }
      if (supportsTerminalAuth || supportsMetaTerminalAuth) {
        terminalAuthMethods.push(consoleLoginMethod);
      }
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
          additionalDirectories: {},
          close: {},
          delete: {},
          fork: {},
          list: {},
          resume: {},
        },
      },
      agentInfo: {
        name: packageJson.name,
        title: "Claude Agent",
        version: packageJson.version,
      },
      authMethods: [
        ...terminalAuthMethods,
        ...(supportsGatewayAuth ? [gatewayAuthMethod, gatewayBedrockAuthMethod] : []),
      ],
    };
  }

  async newSession(params: NewSessionRequest): Promise<NewSessionResponse> {
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
        additionalDirectories: params.additionalDirectories,
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

  async resumeSession(params: ResumeSessionRequest): Promise<ResumeSessionResponse> {
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
    if (_params.methodId === "gateway" || _params.methodId === "gateway-bedrock") {
      this.gatewayAuthRequest = _params as GatewayAuthRequest;
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
    session.accumulatedUsage = {
      inputTokens: 0,
      outputTokens: 0,
      cachedReadTokens: 0,
      cachedWriteTokens: 0,
    };

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
    let handedOff = false;
    let errored = false;

    // Wake-up channel so cancel() can force the drain to return "cancelled"
    // even when query.next() is wedged and never yields again (issue #680).
    const cancelController = new AbortController();
    session.cancelController = cancelController;
    const cancelled = new Promise<void>((resolve) => {
      cancelController.signal.addEventListener("abort", () => resolve(), { once: true });
    });

    try {
      const outcome = await this.drainTurn({
        sessionId: params.sessionId,
        session,
        owner: "prompt",
        promptUuid,
        isLocalOnlyCommand,
        cancelController,
        cancelled,
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
      // The #680 force-cancel backstop (a wedged query.next() that never
      // yielded) resolves as "cancelled" with no meaningful usage — match the
      // wedge-recovery contract by returning the bare stop reason.
      if (cancelController.signal.aborted) {
        return { stopReason: "cancelled" };
      }
      return { stopReason: outcome.stopReason, usage: sessionUsage(session) };
    } catch (error) {
      errored = true;
      // A failed turn typically leaves a trailing `session_state_changed: idle`
      // (and possibly more) in the query iterator. If we don't drain it here,
      // the next prompt's first `query.next()` consumes that stale idle and
      // short-circuits to end_turn with zero usage.
      // Bounded so a misbehaving SDK can't hang the next prompt indefinitely.
      try {
        await session.query.interrupt();
        const MAX_DRAIN = 100;
        for (let i = 0; i < MAX_DRAIN; i++) {
          const { value: m, done } = await session.query.next();
          if (done || !m) break;
          if (m.type === "system" && m.subtype === "session_state_changed" && m.state === "idle") {
            break;
          }
          if (i === MAX_DRAIN - 1) {
            this.logger.error(
              `Session ${params.sessionId}: drained ${MAX_DRAIN} messages after error without observing idle`,
            );
          }
        }
      } catch (drainErr) {
        this.logger.error(
          `Session ${params.sessionId}: failed to drain query after prompt error:`,
          drainErr,
        );
      }

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
      // The drain is returning — interrupt() succeeded or the prompt finished
      // — so disarm the force-cancel backstop and release the wake-up channel
      // (only if we still own it; a handoff installs the next prompt's).
      if (session.forceCancelTimer) {
        clearTimeout(session.forceCancelTimer);
        session.forceCancelTimer = undefined;
      }
      if (session.cancelController === cancelController) {
        session.cancelController = undefined;
      }
      if (!handedOff) {
        session.promptRunning = false;
        if (errored) {
          // The query stream was just drained — handing pending prompts off
          // onto it would let them race with the recovery. Cancel them so
          // each waiting prompt() returns stopReason: "cancelled" and the
          // client can decide whether to retry.
          for (const pending of session.pendingMessages.values()) {
            pending.resolve(true);
          }
          session.pendingMessages.clear();
        } else {
          // This usually should not happen, but in case the loop finishes
          // without claude sending all message replays, we resolve the
          // next pending prompt call to ensure no prompts get stuck.
          this.resolveNextPendingPrompt(session);
        }
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
    cancelController?: AbortController;
    cancelled?: Promise<void>;
  }): Promise<
    | { kind: "turn_ended"; stopReason: StopReason }
    | { kind: "handed_off" }
    | { kind: "stream_ended" }
  > {
    const session = params.session;
    const promptUuid = params.promptUuid;
    const isLocalOnlyCommand = params.isLocalOnlyCommand === true;

    let stopReason: StopReason = "end_turn";
    let lastAssistantTotalUsage: number | null = null;
    let lastAssistantUsage: UsageSnapshot | null = null;
    let lastAssistantModel: string | null = null;
    // See prompt() history: categorical SDK error captured for structured data.
    let lastAssistantError: SDKAssistantMessageError | undefined;
    // Human-readable refusal explanation surfaced on the terminal result.
    let lastRefusalExplanation: string | null = null;
    // Tracks whether we're inside a compaction (the SDK emits the terminal
    // compact_result status twice for a single failed compaction).
    let compactionInProgress = false;
    // Anthropic API message id of the assistant message currently streaming.
    let currentStreamMessageId: string | undefined;
    // Per-message-id record of which assistant content streamed live, so the
    // `assistant` case can drop duplicates but forward un-streamed blocks.
    const streamedTextIds = new Set<string>();
    const streamedThinkingIds = new Set<string>();

    // Classification of a turn drained by the idle pump:
    // - "injected": triggered by a goal/loop instruction we pushed ourselves
    // - "wake": a spontaneous native cron wake turn (emits loop_fired)
    // - "plain": spontaneous activity with no loop armed
    let turnKind: "unknown" | "injected" | "wake" | "plain" = "unknown";
    const markSpontaneousTurn = async (userText?: string) => {
      if (params.owner !== "pump" || turnKind !== "unknown") {
        return;
      }
      const loops = activeLoops(session.anyharness);
      if (loops.length === 0) {
        turnKind = "plain";
        return;
      }
      let loop: LoopState | undefined;
      if (userText) {
        loop = loops.find((l) => userText.includes(l.prompt) || l.prompt.includes(userText));
        if (!loop) {
          // A user message matching no armed loop prompt isn't enough
          // evidence of a wake — wait for assistant activity.
          return;
        }
      } else {
        loop = loops[0];
        if (loops.length > 1) {
          this.logger.error(
            `[anyharness] ambiguous cron wake (${loops.length} loops armed); attributing to ${loop.loopId}`,
          );
        }
      }
      turnKind = "wake";
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
        const winner = await Promise.race([pull, interrupted]);
        session.anyharness.pumpInterrupt = null;
        if (winner === "interrupted") {
          return { kind: "handed_off" };
        }
        iteration = winner;
      } else if (params.owner === "prompt" && params.cancelled) {
        // The #680 force-cancel backstop races the shared pull: a wedged
        // query.next() that never yields is abandoned so cancel() still
        // resolves the prompt as "cancelled" per the ACP contract.
        const next = await Promise.race([pull, params.cancelled]);
        if (params.cancelController?.signal.aborted) {
          void pull.catch(() => {});
          session.pendingQueryNext = null;
          return { kind: "turn_ended", stopReason: "cancelled" };
        }
        iteration = next as IteratorResult<SDKMessage, void>;
      } else {
        iteration = await pull;
      }
      session.pendingQueryNext = null;
      const { value: message, done } = iteration;

      if (done || !message) {
        return { kind: "stream_ended" };
      }
      messagesSeen += 1;

      if (session.emitRawSDKMessages && shouldEmitRawMessage(session.emitRawSDKMessages, message)) {
        await this.client.extNotification("_claude/sdkMessage", {
          sessionId: params.sessionId,
          message: message as Record<string, unknown>,
        });
      }

      switch (message.type) {
        case "system":
          switch (message.subtype) {
            case "init":
              break;
            case "status": {
              if (message.status === "compacting") {
                compactionInProgress = true;
                await this.client.sessionUpdate({
                  sessionId: message.session_id,
                  update: {
                    sessionUpdate: "agent_message_chunk",
                    content: { type: "text", text: "Compacting..." },
                  },
                });
              } else if (message.compact_result === "success" && compactionInProgress) {
                // The SDK signals manual `/compact` completion with a status
                // message carrying `compact_result`, not the `compact_boundary`
                // message (which only fires when there's content to compact).
                compactionInProgress = false;
                await this.client.sessionUpdate({
                  sessionId: message.session_id,
                  update: {
                    sessionUpdate: "agent_message_chunk",
                    content: { type: "text", text: "\n\nCompacting completed." },
                  },
                });
              } else if (message.compact_result === "failed" && compactionInProgress) {
                compactionInProgress = false;
                const reason = message.compact_error ? `: ${message.compact_error}` : ".";
                await this.client.sessionUpdate({
                  sessionId: message.session_id,
                  update: {
                    sessionUpdate: "agent_message_chunk",
                    content: { type: "text", text: `\n\nCompacting failed${reason}` },
                  },
                });
              }
              break;
            }
            case "compact_boundary": {
              // Refresh the displayed usage immediately so the client doesn't
              // keep showing the stale pre-compaction size (e.g. "944k/1m")
              // right after the user sees "Compacting completed", which is
              // confusing and wrong.
              //
              // Prefer the SDK's authoritative post-compaction `used` via
              // getContextUsage — it reflects the real retained context
              // (system prompt + tools + surviving messages), which the
              // per-message API usage numbers can't give us until the next
              // turn's result. If the control request fails, fall back to the
              // used:0 approximation: directionally correct (context just
              // dropped dramatically) and replaced within seconds by the next
              // result message.
              //
              // `size` keeps coming from session.contextWindowSize (learned
              // from modelUsage / the model heuristic) — getContextUsage's
              // window field under-reports extended 1M windows.
              //
              // The "Compacting completed." text is emitted from the `status`
              // handler (keyed on `compact_result`), not here, so the failure
              // path gets a message too.
              const usedTokens = await fetchContextUsedTokens(session.query, this.logger);
              lastAssistantUsage = null;
              lastAssistantTotalUsage = usedTokens ?? 0;
              await this.client.sessionUpdate({
                sessionId: message.session_id,
                update: {
                  sessionUpdate: "usage_update",
                  used: lastAssistantTotalUsage,
                  size: session.contextWindowSize,
                },
              });
              break;
            }
            case "local_command_output": {
              await this.client.sessionUpdate({
                sessionId: message.session_id,
                update: {
                  sessionUpdate: "agent_message_chunk",
                  content: { type: "text", text: message.content },
                },
              });
              break;
            }
            case "session_state_changed": {
              if (message.state === "idle") {
                if (session.cancelled) {
                  stopReason = "cancelled";
                }
                return { kind: "turn_ended", stopReason };
              }
              break;
            }
            case "memory_recall": {
              const isSynthesis = message.mode === "synthesize";
              const locations = isSynthesis ? [] : message.memories.map((m) => ({ path: m.path }));
              const content = isSynthesis
                ? message.memories
                    .filter(
                      (m): m is (typeof message.memories)[number] & { content: string } =>
                        typeof m.content === "string",
                    )
                    .map((m) => ({
                      type: "content" as const,
                      content: { type: "text" as const, text: m.content },
                    }))
                : [];
              const count = message.memories.length;
              const title = isSynthesis
                ? "Recalled synthesized memory"
                : `Recalled ${count} ${count === 1 ? "memory" : "memories"}`;
              await this.client.sessionUpdate({
                sessionId: message.session_id,
                update: {
                  sessionUpdate: "tool_call",
                  toolCallId: message.uuid,
                  title,
                  kind: "read",
                  status: "completed",
                  ...(locations.length > 0 && { locations }),
                  ...(content.length > 0 && { content }),
                  _meta: {
                    claudeCode: {
                      toolName: "memory_recall",
                      toolResponse: { mode: message.mode },
                    },
                  } satisfies ToolUpdateMeta,
                },
              });
              break;
            }
            case "commands_changed": {
              // Push the full slash-command list after a mid-session change
              // (e.g. skills discovered dynamically as the agent works in a
              // subdirectory). The client should REPLACE its cached command
              // list with this payload: supportedCommands() is captured once
              // at initialize and never reflects mid-session changes, so we
              // forward message.commands directly rather than re-querying.
              await this.client.sessionUpdate({
                sessionId: message.session_id,
                update: {
                  sessionUpdate: "available_commands_update",
                  availableCommands: getAvailableSlashCommands(message.commands),
                },
              });
              break;
            }
            case "mirror_error": {
              // The SDK failed to persist session history (SessionStore
              // append rejected/timed out after retry) — potential data loss
              // the user should know about rather than a silent gap on
              // resume. Log it and surface a warning in the conversation.
              this.logger.error(
                `Session ${message.session_id}: failed to persist history: ${message.error}`,
              );
              break;
            }
            case "permission_denied": {
              // A tool call was auto-denied (by a rule, the classifier,
              // dontAsk mode, etc.) before running. The tool_use block was
              // already emitted as a `tool_call`, so mark it failed with the
              // rejection reason — otherwise the client shows a tool call
              // that silently never resolves.
              const reason = message.decision_reason ?? message.message;
              await this.client.sessionUpdate({
                sessionId: message.session_id,
                update: {
                  sessionUpdate: "tool_call_update",
                  toolCallId: message.tool_use_id,
                  status: "failed",
                  content: [
                    {
                      type: "content",
                      content: { type: "text", text: `Permission denied: ${reason}` },
                    },
                  ],
                  _meta: {
                    claudeCode: {
                      toolName: message.tool_name,
                      toolResponse: {
                        decisionReasonType: message.decision_reason_type,
                        decisionReason: message.decision_reason,
                        message: message.message,
                      },
                    },
                  } satisfies ToolUpdateMeta,
                },
              });
              break;
            }
            case "hook_started":
            case "hook_progress":
            case "hook_response":
            case "files_persisted":
            case "task_started":
            case "task_notification":
            case "task_progress":
            case "task_updated":
              break;
            case "elicitation_complete": {
              // A url-mode MCP elicitation finished server-side. Let the client
              // dismiss any UI it opened for it. Only meaningful when the
              // client supports url elicitation; ignore failures otherwise.
              if (this.clientCapabilities?.elicitation?.url) {
                try {
                  await this.client.unstable_completeElicitation({
                    elicitationId: message.elicitation_id,
                  });
                } catch (error) {
                  this.logger.error(`Failed to complete elicitation: ${error}`);
                }
              }
              break;
            }
            case "plugin_install":
            case "notification":
            case "api_retry":
            case "thinking_tokens":
            case "model_refusal_fallback":
              // Todo: process via status api: https://docs.claude.com/en/docs/claude-code/hooks#hook-output
              break;
            default:
              unreachable(message, this.logger);
              break;
          }
          break;
        case "result": {
          // Accumulate usage from this result
          session.accumulatedUsage.inputTokens += message.usage.input_tokens;
          session.accumulatedUsage.outputTokens += message.usage.output_tokens;
          session.accumulatedUsage.cachedReadTokens += message.usage.cache_read_input_tokens;
          session.accumulatedUsage.cachedWriteTokens += message.usage.cache_creation_input_tokens;

          const matchingModelUsage = lastAssistantModel
            ? getMatchingModelUsage(message.modelUsage, lastAssistantModel)
            : null;
          // Only overwrite when we have an authoritative value — a miss
          // (e.g. a turn with no top-level assistant message) would
          // otherwise discard the window learned on a prior turn and
          // leave the next prompt's mid-stream updates reporting 200k.
          if (matchingModelUsage) {
            session.contextWindowSize = matchingModelUsage.contextWindow;
          }

          // Task-notification followups are autonomous work triggered by a
          // task-notification system message, not by the user's prompt.
          // They should not influence the user-turn lifecycle (stop reason,
          // slash-command output forwarding) but their cost is real.
          const isTaskNotification = message.origin?.kind === "task-notification";

          // Send usage_update notification
          if (lastAssistantTotalUsage !== null) {
            await this.client.sessionUpdate({
              sessionId: params.sessionId,
              update: {
                sessionUpdate: "usage_update",
                used: lastAssistantTotalUsage,
                size: session.contextWindowSize,
                cost: {
                  amount: message.total_cost_usd,
                  currency: "USD",
                },
                ...(message.origin && {
                  _meta: { "_claude/origin": message.origin },
                }),
              },
            });
          }

          if (session.cancelled) {
            if (!isTaskNotification) {
              stopReason = "cancelled";
            }
            break;
          }

          // A refusal can arrive on any result subtype (and may even set
          // is_error), so handle it before the subtype switch — otherwise the
          // is_error throw below would surface it as an internal error. The
          // refused assistant message carries no visible content, so surface
          // the classifier's explanation (when available) and report ACP's
          // dedicated `refusal` stop reason.
          if (message.stop_reason === "refusal" && !isTaskNotification) {
            if (lastRefusalExplanation) {
              await this.client.sessionUpdate({
                sessionId: params.sessionId,
                update: {
                  sessionUpdate: "agent_message_chunk",
                  content: { type: "text", text: lastRefusalExplanation },
                },
              });
            }
            stopReason = "refusal";
            break;
          }

          switch (message.subtype) {
            case "success": {
              if (message.result.includes("Please run /login")) {
                throw RequestError.authRequired();
              }
              if (message.stop_reason === "max_tokens") {
                if (!isTaskNotification) {
                  stopReason = "max_tokens";
                }
                break;
              }
              if (message.is_error) {
                throw RequestError.internalError(errorKindData(lastAssistantError), message.result);
              }
              // For local-only commands (no model invocation), the result
              // text is the command output — forward it to the client.
              // Task-notification followups never originate from a user
              // slash command, so skip the forwarding for them.
              if (isLocalOnlyCommand && !isTaskNotification) {
                for (const notification of toAcpNotifications(
                  message.result,
                  "assistant",
                  params.sessionId,
                  session.toolUseCache,
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
                if (!isTaskNotification) {
                  stopReason = "max_tokens";
                }
                break;
              }
              if (message.is_error) {
                throw RequestError.internalError(
                  errorKindData(lastAssistantError),
                  message.errors.join(", ") || message.subtype,
                );
              }
              if (!isTaskNotification) {
                stopReason = "end_turn";
              }
              break;
            }
            case "error_max_budget_usd":
            case "error_max_turns":
            case "error_max_structured_output_retries":
              if (message.is_error) {
                throw RequestError.internalError(
                  errorKindData(lastAssistantError),
                  message.errors.join(", ") || message.subtype,
                );
              }
              if (!isTaskNotification) {
                stopReason = "max_turn_requests";
              }
              break;
            default:
              unreachable(message, this.logger);
              break;
          }
          break;
        }
        case "stream_event": {
          await markSpontaneousTurn();
          // `message_start` carries the Anthropic API message id; capture it
          // so the streamed chunks that follow (whose delta events don't carry
          // it) can all be tagged with the same, replay-stable id.
          if (message.event.type === "message_start") {
            currentStreamMessageId = message.event.message.id || undefined;
          }
          // Record that this top-level message id actually streamed text/
          // thinking, so the `assistant` case below knows its assembled blocks
          // are duplicates (filter them) rather than the only copy (forward
          // them). Gated on `parent_tool_use_id === null` so a subagent stream
          // can't attribute its content to the top-level message id.
          if (
            currentStreamMessageId &&
            message.parent_tool_use_id === null &&
            message.event.type === "content_block_delta"
          ) {
            if (message.event.delta.type === "text_delta") {
              streamedTextIds.add(currentStreamMessageId);
            } else if (message.event.delta.type === "thinking_delta") {
              streamedThinkingIds.add(currentStreamMessageId);
            }
          }
          if (
            message.parent_tool_use_id === null &&
            (message.event.type === "message_start" || message.event.type === "message_delta")
          ) {
            if (message.event.type === "message_start") {
              lastAssistantUsage = snapshotFromUsage(message.event.message.usage);
              const model = message.event.message.model;
              if (model && model !== "<synthetic>") {
                lastAssistantModel = model;
                // Only upgrade from the default — once a `result` has given
                // us an authoritative window, trust it over the heuristic.
                // Model switches invalidate the cached window via
                // `syncSessionConfigState`, which resets us back to the
                // default so this branch runs again for the new model.
                if (session.contextWindowSize === DEFAULT_CONTEXT_WINDOW) {
                  const inferred = inferContextWindowFromModel(model);
                  if (inferred !== null) {
                    session.contextWindowSize = inferred;
                  }
                }
              }
            } else {
              const usage = message.event.usage;
              const prev: Readonly<UsageSnapshot> = lastAssistantUsage ?? ZERO_USAGE;
              // Per Anthropic API, message_delta usage fields are *cumulative*;
              // nullable fields (input_tokens and the cache fields) fall back
              // to the prior snapshot when the server omits them from this
              // delta. Only output_tokens is guaranteed non-null.
              lastAssistantUsage = {
                input_tokens: usage.input_tokens ?? prev.input_tokens,
                output_tokens: usage.output_tokens,
                cache_read_input_tokens:
                  usage.cache_read_input_tokens ?? prev.cache_read_input_tokens,
                cache_creation_input_tokens:
                  usage.cache_creation_input_tokens ?? prev.cache_creation_input_tokens,
              };
            }

            const nextUsage = totalTokens(lastAssistantUsage);
            if (nextUsage !== lastAssistantTotalUsage) {
              lastAssistantTotalUsage = nextUsage;
              await this.client.sessionUpdate({
                sessionId: params.sessionId,
                update: {
                  sessionUpdate: "usage_update",
                  used: nextUsage,
                  size: session.contextWindowSize,
                },
              });
            }
          }
          for (const notification of streamEventToAcpNotifications(
            message,
            params.sessionId,
            session.toolUseCache,
            this.client,
            this.logger,
            {
              clientCapabilities: this.clientCapabilities,
              cwd: session.cwd,
              taskState: session.taskState,
              messageId: currentStreamMessageId,
            },
          )) {
            await this.client.sessionUpdate(notification);
          }
          break;
        }
        case "user":
        case "assistant": {
          if (session.cancelled) {
            break;
          }

          if (message.type === "assistant" && message.message.model !== "<synthetic>") {
            // Synthetic assistant messages are local-command echoes (e.g. an
            // injected /goal), not evidence of a cron wake turn.
            await markSpontaneousTurn();
          }

          // Record the ACP messageId -> SDK uuid mapping for this message. The
          // consolidated message carries both ids, so this is where we learn
          // the uuid that the SDK's rewind/resume APIs key on for the id we
          // hand clients. Not read yet (see Session.messageIdToUuid).
          const mappedMessageId = messageIdForGrouping(message);
          if (mappedMessageId && typeof message.uuid === "string" && message.uuid.length > 0) {
            session.messageIdToUuid.set(mappedMessageId, message.uuid);
          }

          // Check for prompt replay
          if (message.type === "user" && "uuid" in message && message.uuid) {
            if (message.uuid === promptUuid) {
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
              break;
            }
            if (params.owner === "pump" && turnKind === "unknown") {
              // A spontaneous user message may be a native cron wake prompt.
              await markSpontaneousTurn(userMessageText(message.message.content));
            }
            if ("isReplay" in message && message.isReplay) {
              // not pending or unrelated replay message
              break;
            }
          }

          // Snapshot the latest top-level assistant usage and model so the
          // next `result` can emit a usage_update tied to the right context
          // window. Subagent messages are excluded to keep the snapshot
          // aligned with what the user's current selection is producing.
          if (message.type === "assistant" && message.parent_tool_use_id === null) {
            lastAssistantUsage = snapshotFromUsage(message.message.usage);
            lastAssistantTotalUsage = totalTokens(lastAssistantUsage);
            if (message.message.model && message.message.model !== "<synthetic>") {
              lastAssistantModel = message.message.model;
            }
            if (message.error) {
              lastAssistantError = message.error;
            }
            if (message.message.stop_reason === "refusal") {
              lastRefusalExplanation = message.message.stop_details?.explanation ?? null;
            }
          }

          // Strip <command-*>/<local-command-stdout> markers and render any
          // remaining prose. Skill bodies and built-in slash commands (e.g.
          // /usage, /status, /model) arrive wrapped in these tags; pure-marker
          // payloads (e.g. /compact's malformed output) strip to null and are
          // skipped. Mirrors the replay path at replaySessionHistory.
          if (
            message.message.role !== "system" &&
            typeof message.message.content === "string" &&
            message.message.content.includes("<local-command-stdout>")
          ) {
            const stripped = stripLocalCommandMetadata(message.message.content);
            if (typeof stripped === "string") {
              for (const notification of toAcpNotifications(
                stripped,
                message.message.role,
                params.sessionId,
                session.toolUseCache,
                this.client,
                this.logger,
                {
                  clientCapabilities: this.clientCapabilities,
                  parentToolUseId: message.parent_tool_use_id,
                  cwd: session.cwd,
                  taskState: session.taskState,
                  messageId: messageIdForGrouping(message),
                },
              )) {
                await this.client.sessionUpdate(notification);
              }
            } else {
              this.logger.log(message.message.content);
            }
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
          if (message.message.role === "system") {
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

          let content: typeof message.message.content;
          if (message.type === "assistant" && message.parent_tool_use_id === null) {
            // Top-level assistant message: drop text/thinking blocks already
            // streamed live as chunks, and forward (as a fallback) any that were
            // not, so non-streaming gateways still deliver the final answer.
            const id = messageIdForGrouping(message);
            content = message.message.content.filter((item) => {
              // Non-text blocks (tool_use, etc.) always pass through; their own
              // dedupe (`toolUseCache`) collapses the streamed/assembled pair.
              if (item.type !== "text" && item.type !== "thinking") {
                return true;
              }
              // Already delivered live as a chunk of this exact type — drop the
              // duplicate. Checked per type so streaming one (e.g. text) doesn't
              // suppress an un-streamed block of the other (e.g. thinking).
              const streamedLive =
                id !== undefined &&
                (item.type === "text" ? streamedTextIds : streamedThinkingIds).has(id);
              if (streamedLive) {
                return false;
              }
              // Empty assembled blocks carry nothing (some gateways emit an empty
              // `thinking` block before the real text) — don't forward stray
              // empty chunks.
              const text = item.type === "text" ? item.text : item.thinking;
              if (text.length === 0) {
                return false;
              }
              return true;
            });
          } else if (message.type === "assistant") {
            // Subagent assistant message (`parent_tool_use_id !== null`). It is
            // never streamed live and its text/thinking is internal to the tool
            // call — keep dropping it so subagent prose doesn't leak into the
            // top-level feed.
            content = message.message.content.filter(
              (item) => item.type !== "text" && item.type !== "thinking",
            );
          } else {
            content = message.message.content;
          }

          for (const notification of toAcpNotifications(
            content,
            message.message.role,
            params.sessionId,
            session.toolUseCache,
            this.client,
            this.logger,
            {
              clientCapabilities: this.clientCapabilities,
              parentToolUseId: message.parent_tool_use_id,
              cwd: session.cwd,
              taskState: session.taskState,
              messageId: messageIdForGrouping(message),
            },
          )) {
            await this.client.sessionUpdate(notification);
          }
          break;
        }
        case "tool_progress": {
          await this.client.sessionUpdate({
            sessionId: message.session_id,
            update: {
              sessionUpdate: "tool_call_update",
              toolCallId: message.tool_use_id,
              status: "in_progress",
              _meta: {
                claudeCode: {
                  toolName: message.tool_name,
                  toolResponse: { elapsedTimeSeconds: message.elapsed_time_seconds },
                },
              } satisfies ToolUpdateMeta,
            },
          });
          break;
        }
        case "rate_limit_event": {
          if (lastAssistantTotalUsage !== null) {
            await this.client.sessionUpdate({
              sessionId: message.session_id,
              update: {
                sessionUpdate: "usage_update",
                used: lastAssistantTotalUsage,
                size: session.contextWindowSize,
                _meta: { "_claude/rateLimit": message.rate_limit_info },
              },
            });
          }
          break;
        }
        case "tool_use_summary":
        case "auth_status":
        case "prompt_suggestion":
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
      // active and have no token budget — return the current goal unchanged.
      if (!ah.goal || ah.goal.status !== "active") {
        throw RequestError.invalidParams(
          undefined,
          "objective is required (no active goal to patch)",
        );
      }
      return { goal: goalWireFromState(ah.goal) };
    }

    this.ensureTranscriptTailer(sessionId, session);
    // Native semantics: "/goal <condition>" arms the goal; re-sending
    // replaces. The mirror transitions only once the native arm sentinel
    // round-trips through the transcript — no optimistic saved-state.
    const confirmed = this.waitForGoalRow(
      session,
      (row) => classifyGoalStatus(row) === "armed" && (row.condition ?? "").trim() === objective,
      GOAL_SET_TIMEOUT_MS,
    );
    this.pushInjectedInstruction(sessionId, session, `/goal ${objective}`);

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
    const confirmed = this.waitForGoalRow(
      session,
      (row) => classifyGoalStatus(row) === "cleared",
      GOAL_CLEAR_TIMEOUT_MS,
    );
    // Always send the native clear so mirror drift heals; with no native
    // goal armed it is a zero-token no-op ("No goal set") that writes no
    // transcript row, so only wait for confirmation when one is expected.
    this.pushInjectedInstruction(sessionId, session, "/goal clear");
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

    // Both schedule kinds translate to "/loop <expr> <prompt>": interval
    // expressions like "5m" directly, cron kind as the raw crontab string.
    const created = new Promise<LoopState>((resolve) => {
      ah.pendingLoopSets.push({
        prompt,
        schedule: { kind: schedule.kind, expr },
        recurring,
        requestedAtMs: Date.now(),
        resolve,
      });
    });
    this.pushInjectedInstruction(sessionId, session, `/loop ${expr} ${prompt}`);

    // Wait for the CronCreate tool_use so the response carries the real
    // native cron id; fall back to a provisional loop on timeout (the
    // loop_updated notification remains the source of truth either way).
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
    void this.sendAnyharnessEvent(sessionId, "loop_updated", {
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
    this.pushInjectedInstruction(sessionId, session, instruction);

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
        void this.sendAnyharnessEvent(sessionId, "loop_cleared", { loopId: loop.loopId });
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
      const loopId =
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
      ah.loops.set(loopId, loop);
      pending?.resolve(loop);
      await this.sendAnyharnessEvent(sessionId, "loop_updated", {
        loop: loopWireFromState(loop),
        loopId,
      });
      return;
    }

    if (toolName === "CronDelete") {
      const deletedId = extractCronId(toolInput) ?? extractCronId(toolResponse);
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
        await this.sendAnyharnessEvent(sessionId, "loop_cleared", { loopId: target.loopId });
      }
      for (const watcher of ah.loopClearWatchers) {
        watcher();
      }
      return;
    }
    // CronList: observation only (logged above).
  }

  /**
   * Pushes an instruction into the session as a user message the pump (or a
   * running prompt drain) will process. The uuid is remembered so the drain
   * loop can tell these turns apart from native cron wakes and keep their
   * replays out of the client feed.
   */
  private pushInjectedInstruction(sessionId: string, session: Session, text: string): void {
    const uuid = randomUUID();
    session.anyharness.injectedUuids.add(uuid);
    const message: SDKUserMessage = {
      type: "user",
      message: { role: "user", content: [{ type: "text", text }] },
      session_id: sessionId,
      parent_tool_use_id: null,
      uuid,
    };
    session.input.push(message);
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
    payload: { goal?: GoalWire; loop?: LoopWire; loopId?: string },
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

  private handleTranscriptRow(sessionId: string, row: unknown): void {
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
      return;
    }
    session.cancelled = true;
    for (const [, pending] of session.pendingMessages) {
      pending.resolve(true);
    }
    session.pendingMessages.clear();

    // Arm a backstop before interrupting: if a prompt is actively consuming
    // the query and interrupt() doesn't make the SDK yield (e.g. a wedged
    // TaskOutput block — issue #680), force the loop to return "cancelled"
    // after the floor elapses so the pending session/prompt still resolves per
    // the ACP cancellation contract instead of hanging forever. The loop's
    // `finally` clears this timer when interrupt() works and it returns through
    // the normal idle path, so on healthy cancels it is armed but never fires.
    //
    // Arm at most once per turn: the floor is an absolute ceiling from the
    // first cancel, so a client that re-sends cancel (each call still retries
    // interrupt() below) can't keep pushing the deadline out.
    if (
      session.promptRunning &&
      session.cancelController &&
      !session.cancelController.signal.aborted &&
      !session.forceCancelTimer
    ) {
      const cancelController = session.cancelController;
      session.forceCancelTimer = setTimeout(() => {
        this.logger.error(
          `Session ${params.sessionId}: cancel floor elapsed without the SDK yielding; forcing "cancelled". The underlying query may still be wedged — a new session may be required.`,
        );
        cancelController.abort();
      }, this.forceCancelGraceMs);
    }

    await session.query.interrupt();
  }

  /** Cleanly tear down a session: cancel in-flight work, dispose resources,
   *  and remove it from the session map. */
  private async teardownSession(sessionId: string): Promise<void> {
    const session = this.sessions[sessionId];
    if (!session) {
      return;
    }
    await this.cancel({ sessionId });
    // cancel() arms the force-cancel floor and interrupts gracefully, but a
    // wedged prompt loop only wakes when `cancelController` aborts — closing
    // the query/abortController below doesn't touch it. Since we're tearing the
    // session down anyway, wake the loop now so the in-flight prompt() resolves
    // immediately instead of after the floor, and clear the timer so it can't
    // outlive the deleted session (it isn't unref'd and would otherwise keep
    // the event loop alive until it fires).
    if (session.forceCancelTimer) {
      clearTimeout(session.forceCancelTimer);
      session.forceCancelTimer = undefined;
    }
    session.cancelController?.abort();
    session.settingsManager.dispose();
    session.abortController.abort();
    session.query.close();
    delete this.sessions[sessionId];
  }

  /** Tear down all active sessions. Called when the ACP connection closes. */
  async dispose(): Promise<void> {
    await Promise.all(Object.keys(this.sessions).map((id) => this.teardownSession(id)));
  }

  async closeSession(params: CloseSessionRequest): Promise<CloseSessionResponse> {
    if (!this.sessions[params.sessionId]) {
      throw new Error("Session not found");
    }
    await this.teardownSession(params.sessionId);
    return {};
  }

  async deleteSession(params: DeleteSessionRequest): Promise<DeleteSessionResponse> {
    // Tear down any active in-memory state first so the on-disk file isn't
    // recreated by an outstanding query writing to it.
    if (this.sessions[params.sessionId]) {
      await this.teardownSession(params.sessionId);
    }
    await deleteSession(params.sessionId);
    return {};
  }

  async setSessionMode(params: SetSessionModeRequest): Promise<SetSessionModeResponse> {
    if (!this.sessions[params.sessionId]) {
      throw new Error("Session not found");
    }

    await this.applySessionMode(params.sessionId, params.modeId);
    await this.updateConfigOption(params.sessionId, "mode", params.modeId);
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

    if (params.configId === "mode") {
      await this.applySessionMode(params.sessionId, resolvedValue);
      await this.client.sessionUpdate({
        sessionId: params.sessionId,
        update: {
          sessionUpdate: "current_mode_update",
          currentModeId: resolvedValue,
        },
      });
    } else if (params.configId === "model") {
      await this.sessions[params.sessionId].query.setModel(resolvedValue);
    }
    // Effort SDK sync is handled inside applyConfigOptionValue so that direct
    // effort changes and effort changes induced by a model switch go through
    // the same path.

    await this.applyConfigOptionValue(params.sessionId, session, params.configId, resolvedValue);

    return { configOptions: session.configOptions };
  }

  private async applySessionMode(sessionId: string, modeId: string): Promise<void> {
    switch (modeId) {
      case "auto":
      case "default":
      case "acceptEdits":
      case "bypassPermissions":
      case "dontAsk":
      case "plan":
        break;
      default:
        throw new Error("Invalid Mode");
    }

    const session = this.sessions[sessionId];
    if (!session) {
      throw new Error("Session not found");
    }
    if (!session.modes.availableModes.some((mode) => mode.id === modeId)) {
      throw new Error(`Mode ${modeId} is not available in this session`);
    }

    try {
      await session.query.setPermissionMode(modeId);
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
      // Backfill the ACP messageId -> SDK uuid mapping for messages we didn't
      // observe live (resumed/loaded sessions), so rewind/resume can translate
      // a client-supplied id without an extra getSessionMessages read. Not read
      // yet (see Session.messageIdToUuid).
      const replayMessageId = messageIdForGrouping(message);
      const replaySession = this.sessions[sessionId];
      if (replaySession && replayMessageId && message.uuid) {
        replaySession.messageIdToUuid.set(replayMessageId, message.uuid);
      }

      // @ts-expect-error - untyped in SDK but we handle all of these
      let content: unknown = message.message.content;
      // @ts-expect-error - untyped in SDK but we handle all of these
      if (message.message.role === "user") {
        content = stripLocalCommandMetadata(content);
        if (content === null) continue;
      }

      for (const notification of toAcpNotifications(
        // @ts-expect-error - untyped in SDK but we handle all of these
        content,
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
          taskState: this.sessions[sessionId]?.taskState,
          messageId: replayMessageId,
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
      const alwaysAllowLabel = describeAlwaysAllow(suggestions, toolName);
      const supportsTerminalOutput = this.clientCapabilities?._meta?.["terminal_output"] === true;
      const session = this.sessions[sessionId];
      if (!session) {
        return {
          behavior: "deny",
          message: "Session not found",
        };
      }

      // AskUserQuestion is surfaced to us as a normal permission check (the SDK
      // routes it through canUseTool whenever a callback is registered, rather
      // than the interactive dialog). Present it as an ACP form elicitation and
      // feed the answers back as updatedInput for the tool's own call() to read.
      if (toolName === "AskUserQuestion" && this.clientCapabilities?.elicitation?.form) {
        return this.handleAskUserQuestion(sessionId, toolInput, toolUseID, signal);
      }

      if (toolName === "ExitPlanMode") {
        const optionsAll: PermissionOption[] = [
          { kind: "allow_always", name: 'Yes, and use "auto" mode', optionId: "auto" },
          {
            kind: "allow_always",
            name: "Yes, and auto-accept edits",
            optionId: "acceptEdits",
          },
          { kind: "allow_once", name: "Yes, and manually approve edits", optionId: "default" },
          { kind: "reject_once", name: "No, keep planning", optionId: "plan" },
        ];
        if (ALLOW_BYPASS) {
          optionsAll.unshift({
            kind: "allow_always",
            name: "Yes, and bypass permissions",
            optionId: "bypassPermissions",
          });
        }
        // Filter against the session's currently-advertised modes so we never
        // present options the active model can't honor (e.g. `auto` on Haiku).
        // `bypassPermissions` is already covered by `availableModes` via
        // `buildAvailableModes`/`ALLOW_BYPASS`. The `plan` option is a
        // "keep planning" reject path; it's always present in `availableModes`.
        const options = optionsAll.filter((o) =>
          session.modes.availableModes.some((m) => m.id === o.optionId),
        );

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
        const selectedMode =
          response.outcome?.outcome === "selected" ? response.outcome.optionId : undefined;
        const selectedModeWasOffered = options.some((option) => option.optionId === selectedMode);
        if (
          selectedModeWasOffered &&
          (selectedMode === "default" ||
            selectedMode === "acceptEdits" ||
            selectedMode === "auto" ||
            selectedMode === "bypassPermissions")
        ) {
          await this.client.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "current_mode_update",
              currentModeId: selectedMode,
            },
          });
          await this.updateConfigOption(sessionId, "mode", selectedMode);

          return {
            behavior: "allow",
            updatedInput: toolInput,
            updatedPermissions: suggestions ?? [
              { type: "setMode", mode: selectedMode, destination: "session" },
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
            name: alwaysAllowLabel,
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

  /**
   * Handle elicitation requests that originate from MCP servers by forwarding
   * them to the client over ACP. Modes the client did not advertise (or
   * requests we can't represent) are declined.
   */
  private handleMcpElicitation(sessionId: string, support: ElicitationSupport): OnElicitation {
    return async (request, { signal }) => {
      const isUrl = request.mode === "url";
      if ((isUrl && !support.url) || (!isUrl && !support.form)) {
        return { action: "decline" };
      }

      const createRequest = mcpElicitationToCreateRequest(request, sessionId);
      if (!createRequest) {
        return { action: "decline" };
      }

      try {
        const response = await this.client.unstable_createElicitation(createRequest);
        if (signal.aborted) {
          return { action: "cancel" };
        }
        return createElicitationResponseToElicitResult(response);
      } catch (error) {
        this.logger.error(`Failed to forward MCP elicitation: ${error}`);
        return { action: "decline" };
      }
    };
  }

  /**
   * Present the built-in AskUserQuestion tool's questions as an ACP form
   * elicitation and return the answers as the tool's `updatedInput`. Called from
   * `canUseTool` since that is where the SDK routes the tool's permission check.
   */
  private async handleAskUserQuestion(
    sessionId: string,
    toolInput: Record<string, unknown>,
    toolUseID: string,
    signal: AbortSignal,
  ): Promise<PermissionResult> {
    const questions = extractAskUserQuestions(toolInput);
    if (!questions) {
      return { behavior: "deny", message: "AskUserQuestion called with no valid questions." };
    }

    const createRequest = askUserQuestionsToCreateRequest(questions, sessionId, toolUseID);
    let response;
    try {
      response = await this.client.unstable_createElicitation(createRequest);
    } catch (error) {
      this.logger.error(`Failed to present AskUserQuestion elicitation: ${error}`);
      return { behavior: "deny", message: "Could not present the question to the user." };
    }
    if (signal.aborted) {
      throw new Error("Tool use aborted");
    }

    const outcome = applyAskElicitationResponse(response, toolInput, questions);
    if (outcome.action === "cancel") {
      throw new Error("Tool use aborted");
    }
    return { behavior: "allow", updatedInput: outcome.updatedInput };
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

  private async updateConfigOption(
    sessionId: string,
    configId: string,
    value: string,
  ): Promise<void> {
    const session = this.sessions[sessionId];
    if (!session) return;

    await this.applyConfigOptionValue(sessionId, session, configId, value);

    await this.client.sessionUpdate({
      sessionId,
      update: {
        sessionUpdate: "config_option_update",
        configOptions: session.configOptions,
      },
    });
  }

  private async applyConfigOptionValue(
    sessionId: string,
    session: Session,
    configId: string,
    value: string,
  ): Promise<void> {
    if (configId === "mode") {
      session.modes = { ...session.modes, currentModeId: value };
      session.configOptions = session.configOptions.map((o) =>
        o.id === configId && typeof o.currentValue === "string" ? { ...o, currentValue: value } : o,
      );
    } else if (configId === "model") {
      if (session.models.currentModelId !== value) {
        // The cached context window was learned for the previous model; reset
        // to the new model's heuristic so mid-stream updates between now and
        // the next `result` reflect the user's selection instead of the old
        // model's window.
        session.contextWindowSize = inferContextWindowFromModel(value) ?? DEFAULT_CONTEXT_WINDOW;
      }
      session.models = { ...session.models, currentModelId: value };

      // Recompute availableModes for the new model and clamp the current
      // mode if the SDK no longer offers it (today: "auto" on Haiku).
      // `ModelInfo.supportsAutoMode` is the canonical SDK signal.
      const newModelInfo = session.modelInfos.find((m) => m.value === value);
      const newAvailableModes = buildAvailableModes(newModelInfo);
      // Capture BEFORE mutating session.modes so the log message reflects
      // the invalidated mode rather than "default".
      const previousModeId = session.modes.currentModeId;
      let modeDowngraded = false;
      if (!newAvailableModes.some((m) => m.id === previousModeId)) {
        session.modes = {
          availableModes: newAvailableModes,
          currentModeId: "default",
        };
        try {
          await session.query.setPermissionMode("default");
        } catch (err) {
          // Failing the entire model switch over a bookkeeping sync error is
          // worse UX than logging and continuing; the user explicitly asked
          // to change models. The next setPermissionMode from the user will
          // either succeed or surface a fresh error.
          this.logger.error(
            `Failed to sync permissionMode to "default" after model switch invalidated "${previousModeId}":`,
            err,
          );
        }
        modeDowngraded = true;
      } else {
        session.modes = { ...session.modes, availableModes: newAvailableModes };
      }

      // Rebuild config options since effort levels and fast_mode availability
      // depend on the selected model
      const effortOpt = session.configOptions.find((o) => o.id === "effort");
      const currentEffort =
        typeof effortOpt?.currentValue === "string" ? effortOpt.currentValue : undefined;
      session.configOptions = buildConfigOptions(
        session.modes,
        session.models,
        session.modelInfos,
        currentEffort,
        session.fastModeEnabled,
      );

      // Sync effort with the SDK if it changed after the model switch
      const newEffortOpt = session.configOptions.find((o) => o.id === "effort");
      const newEffort =
        typeof newEffortOpt?.currentValue === "string" ? newEffortOpt.currentValue : undefined;
      if (newEffort !== currentEffort) {
        await session.query.applyFlagSettings({
          effortLevel: toSdkEffortLevel(newEffort),
        });
      }

      // Emit current_mode_update only after session.modes AND
      // session.configOptions have been fully reconciled. This way, a failure
      // in the configOptions/effort rebuild above can't leave the client with
      // a clamped currentModeId but stale configOptions, and the notification
      // still precedes the caller's config_option_update so order-sensitive
      // clients update currentModeId before re-rendering the option list.
      if (modeDowngraded) {
        await this.client.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "current_mode_update",
            currentModeId: "default",
          },
        });
      }
    } else {
      session.configOptions = session.configOptions.map((o) =>
        o.id === configId && typeof o.currentValue === "string" ? { ...o, currentValue: value } : o,
      );
      if (configId === "effort") {
        await session.query.applyFlagSettings({
          effortLevel: toSdkEffortLevel(value),
        });
      } else if (configId === "fast_mode") {
        const enabled = value === "on";
        session.fastModeEnabled = enabled;
        await session.query.applyFlagSettings({ fastMode: enabled });
      }
    }
  }

  private async getOrCreateSession(params: {
    sessionId: string;
    cwd: string;
    mcpServers?: NewSessionRequest["mcpServers"];
    additionalDirectories?: NewSessionRequest["additionalDirectories"];
    _meta?: NewSessionRequest["_meta"];
  }): Promise<NewSessionResponse> {
    const existingSession = this.sessions[params.sessionId];
    if (existingSession) {
      const fingerprint = computeSessionFingerprint(params);
      if (fingerprint === existingSession.sessionFingerprint) {
        return {
          sessionId: params.sessionId,
          modes: existingSession.modes,
          configOptions: existingSession.configOptions,
        };
      }

      // Session-defining params changed (e.g. cwd pointed at a git worktree,
      // or MCP servers reconfigured). Tear down the existing session and
      // recreate it so the underlying Query process picks up the new values.
      await this.teardownSession(params.sessionId);
    }

    const response = await this.createSession(
      {
        cwd: params.cwd,
        mcpServers: params.mcpServers ?? [],
        additionalDirectories: params.additionalDirectories,
        _meta: params._meta,
      },
      {
        resume: params.sessionId,
      },
    );

    return {
      sessionId: response.sessionId,
      modes: response.modes,
      configOptions: response.configOptions,
    };
  }

  /**
   * Ensures the requested `cwd` is an absolute path that points at an existing
   * directory before we create a session. Throws an `invalidParams` error with
   * an actionable message so clients (e.g. Zed) can surface it to the user
   * instead of failing later with an opaque SDK error.
   */
  private async validateCwd(cwd: string): Promise<void> {
    if (!path.isAbsolute(cwd)) {
      throw RequestError.invalidParams(
        { cwd },
        `\`cwd\` must be an absolute path, but received: ${cwd}`,
      );
    }

    let stats: Stats;
    try {
      stats = await fs.stat(cwd);
    } catch {
      throw RequestError.invalidParams(
        { cwd },
        `\`cwd\` does not exist on the machine running the agent: ${cwd}`,
      );
    }

    if (!stats.isDirectory()) {
      throw RequestError.invalidParams({ cwd }, `\`cwd\` is not a directory: ${cwd}`);
    }
  }

  private async createSession(
    params: NewSessionRequest,
    creationOpts: { resume?: string; forkSession?: boolean } = {},
  ): Promise<NewSessionResponse> {
    // Validate `cwd` up front. The ACP spec requires an absolute path, and the
    // directory must actually exist on the machine running the agent. Without
    // this check a session is created against a missing directory and the
    // failure only surfaces later as a confusing "native binary failed to
    // launch" error from the SDK (see issue #749).
    await this.validateCwd(params.cwd);

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
        } else if (!("type" in server)) {
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
        customPrompt !== null &&
        !Array.isArray(customPrompt)
      ) {
        // Forward all preset options (append, excludeDynamicSections, and
        // anything the SDK adds later) while locking type/preset.
        systemPrompt = {
          ...(customPrompt as object),
          type: "preset",
          preset: "claude_code",
        } as Options["systemPrompt"];
      }
    }

    const permissionMode = resolvePermissionMode(
      settingsManager.getSettings().permissions?.defaultMode,
      this.logger,
    );

    // Extract options from _meta if provided
    const sessionMeta = params._meta as NewSessionMeta | undefined;
    const userProvidedOptions = sessionMeta?.claudeCode?.options;

    // Configure thinking behavior from environment variable
    const thinking = resolveThinkingConfig(process.env.MAX_THINKING_TOKENS, this.logger);

    // Parse model configuration from environment (e.g. Bedrock model overrides)
    const modelConfig = parseModelConfig(process.env.CLAUDE_MODEL_CONFIG);

    // Elicitation modes the connected client advertised. We only forward
    // elicitations (and only re-enable AskUserQuestion) for modes the client
    // can actually render.
    const elicitationSupport: ElicitationSupport = {
      form: !!this.clientCapabilities?.elicitation?.form,
      url: !!this.clientCapabilities?.elicitation?.url,
    };

    // AskUserQuestion surfaces as a `permission_ask_user_question` dialog that
    // we render as a form elicitation. Without form-elicitation support there
    // is no way to present it over ACP, so keep it disabled in that case.
    const disallowedTools = elicitationSupport.form ? [] : ["AskUserQuestion"];

    // Resolve which built-in tools to expose.
    // Explicit tools array from _meta.claudeCode.options takes precedence.
    // disableBuiltInTools is a legacy shorthand for tools: [] — kept for
    // backward compatibility but callers should prefer the tools array.
    const tools: Options["tools"] =
      userProvidedOptions?.tools ??
      (params._meta?.disableBuiltInTools === true ? [] : { type: "preset", preset: "claude_code" });

    const abortController = userProvidedOptions?.abortController || new AbortController();

    // Per-session task state. Created here (rather than in the session record
    // below) so the TaskCreated/TaskCompleted hook callbacks can close over
    // the same Map that the streaming message handler will read from.
    const taskState: TaskState = new Map();

    const options: Options = {
      systemPrompt,
      settingSources: ["user", "project", "local"],
      ...(thinking !== undefined && { thinking }),
      ...userProvidedOptions,
      // CLAUDE_MODEL_CONFIG env var is a fallback for model
      // configuration (e.g. Bedrock model ID overrides). When the caller
      // provides settings via _meta, we intentionally ignore the env var —
      // the caller is assumed to have full control over model configuration.
      ...(!userProvidedOptions?.settings &&
        modelConfig && {
          settings: {
            ...(modelConfig.modelOverrides && { modelOverrides: modelConfig.modelOverrides }),
            ...(modelConfig.availableModels && { availableModels: modelConfig.availableModels }),
          },
        }),
      env: {
        ...process.env,
        ...userProvidedOptions?.env,
        ...createEnvForGateway(this.gatewayAuthRequest),
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
      // Forward MCP elicitation requests onto ACP elicitation. Only attached
      // when the client advertised support, so non-supporting clients keep the
      // SDK's default (auto-decline) behavior. (AskUserQuestion is handled in
      // canUseTool, not here.)
      ...(elicitationSupport.form || elicitationSupport.url
        ? { onElicitation: this.handleMcpElicitation(sessionId, elicitationSupport) }
        : {}),
      pathToClaudeCodeExecutable: process.env.CLAUDE_CODE_EXECUTABLE ?? (await claudeCliPath()),
      extraArgs: {
        ...userProvidedOptions?.extraArgs,
        "replay-user-messages": "",
      },
      disallowedTools: [...(userProvidedOptions?.disallowedTools || []), ...disallowedTools],
      tools,
      hooks: {
        ...userProvidedOptions?.hooks,
        PostToolUse: [
          ...(userProvidedOptions?.hooks?.PostToolUse || []),
          {
            hooks: [
              createPostToolUseHook(this.logger, {
                onEnterPlanMode: async () => {
                  await this.client.sessionUpdate({
                    sessionId,
                    update: {
                      sessionUpdate: "current_mode_update",
                      currentModeId: "plan",
                    },
                  });
                  await this.updateConfigOption(sessionId, "mode", "plan");
                },
                onTranscriptPath: (transcriptPath) =>
                  this.noteTranscriptPath(sessionId, transcriptPath),
                onCronTool: (toolName, toolInput, toolResponse) =>
                  this.handleCronTool(sessionId, toolName, toolInput, toolResponse),
              }),
            ],
          },
        ],
        TaskCreated: [
          ...(userProvidedOptions?.hooks?.TaskCreated || []),
          {
            hooks: [
              createTaskHook({
                taskState,
                onChange: async () => {
                  await this.client.sessionUpdate({
                    sessionId,
                    update: {
                      sessionUpdate: "plan",
                      entries: taskStateToPlanEntries(taskState),
                    },
                  });
                },
              }),
            ],
          },
        ],
        TaskCompleted: [
          ...(userProvidedOptions?.hooks?.TaskCompleted || []),
          {
            hooks: [
              createTaskHook({
                taskState,
                onChange: async () => {
                  await this.client.sessionUpdate({
                    sessionId,
                    update: {
                      sessionUpdate: "plan",
                      entries: taskStateToPlanEntries(taskState),
                    },
                  });
                },
              }),
            ],
          },
        ],
      },
      ...creationOpts,
      abortController,
    };

    // Prefer the official ACP `additionalDirectories` field. Fall back to the
    // legacy `_meta.additionalRoots` extension for clients that haven't been
    // updated yet. Either source is merged with directories supplied via
    // `_meta.claudeCode.options.additionalDirectories` (SDK pass-through).
    const acpAdditionalDirectories =
      params.additionalDirectories ?? sessionMeta?.additionalRoots ?? [];
    options.additionalDirectories = [
      ...(userProvidedOptions?.additionalDirectories ?? []),
      ...acpAdditionalDirectories,
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
    });

    let initializationResult;
    try {
      initializationResult = await q.initializationResult();
    } catch (error) {
      if (
        creationOpts.resume &&
        error instanceof Error &&
        (error.message === "Query closed before response received" ||
          error.message.includes("No conversation found with session ID"))
      ) {
        throw RequestError.resourceNotFound(sessionId);
      }
      throw error;
    }

    if (
      shouldHideClaudeAuth() &&
      initializationResult.account.subscriptionType &&
      !this.gatewayAuthRequest
    ) {
      throw RequestError.authRequired(
        undefined,
        "This integration does not support using claude.ai subscriptions.",
      );
    }

    // Apply user's `availableModels` allowlist from settings.json before any
    // downstream model handling. The SDK only enforces this allowlist in its
    // own UI, not in `initializationResult.models`, so we filter here to keep
    // configOptions, the current-model resolver, and the stored modelInfos
    // consistent with what the user configured.
    const settingsAvailableModels = settingsManager.getSettings().availableModels;
    const allowedModels = Array.isArray(settingsAvailableModels)
      ? applyAvailableModelsAllowlist(initializationResult.models, settingsAvailableModels)
      : initializationResult.models;

    const models = await getAvailableModels(
      q,
      allowedModels,
      initializationResult.models,
      settingsManager,
      this.logger,
    );

    // Gate `auto` (and future model-specific modes) on the resolved model's
    // `ModelInfo`. See `buildAvailableModes` for the canonical SDK signal.
    const currentModelInfo = allowedModels.find((m) => m.value === models.currentModelId);
    const availableModes = buildAvailableModes(currentModelInfo);

    // Clamp `permissionMode` if the resolved session does not offer it. The
    // common case is `permissions.defaultMode: "auto"` resolving to a model
    // that does not support auto mode (e.g. Haiku); without this clamp the
    // SDK would later throw `"auto mode unavailable for this model"` from
    // `setPermissionMode`. Keep `permissionMode` as the resolved user intent
    // (matches what was passed into `options.permissionMode` above) and use
    // `effectiveMode` for the post-clamp value the session actually runs in.
    let effectiveMode: PermissionMode = permissionMode;
    if (!availableModes.some((m) => m.id === effectiveMode)) {
      if (effectiveMode === "auto") {
        this.logger.error(
          `permissions.defaultMode "auto" is not available for model ` +
            `"${models.currentModelId}"; falling back to "default".`,
        );
      } else {
        this.logger.error(
          `permissions.defaultMode "${effectiveMode}" is not available in ` +
            `this session; falling back to "default".`,
        );
      }
      effectiveMode = "default";
      // Sync the SDK so it doesn't keep "auto" cached internally. Wrapped in
      // try/catch since failing here would abort session creation entirely.
      try {
        await q.setPermissionMode("default");
      } catch (err) {
        this.logger.error("Failed to sync clamped permissionMode to SDK:", err);
      }
    }

    const modes = {
      currentModeId: effectiveMode,
      availableModes,
    };

    // Fast mode starts disabled for every new session. The client can enable it
    // via setSessionConfigOption after session creation.
    const initialFastModeEnabled = false;

    const configOptions = buildConfigOptions(
      modes,
      models,
      allowedModels,
      settingsManager.getSettings().effortLevel,
      initialFastModeEnabled,
    );

    // Apply the initial effort level to the SDK so it matches the UI default
    const initialEffort = configOptions.find((o) => o.id === "effort");
    if (
      initialEffort &&
      typeof initialEffort.currentValue === "string" &&
      initialEffort.currentValue !== "default"
    ) {
      await q.applyFlagSettings({
        effortLevel: initialEffort.currentValue as Settings["effortLevel"],
      });
    }
    this.sessions[sessionId] = {
      query: q,
      input: input,
      cancelled: false,
      cwd: params.cwd,
      sessionFingerprint: computeSessionFingerprint(params),
      settingsManager,
      accumulatedUsage: {
        inputTokens: 0,
        outputTokens: 0,
        cachedReadTokens: 0,
        cachedWriteTokens: 0,
      },
      modes,
      models,
      modelInfos: allowedModels,
      configOptions,
      promptRunning: false,
      pendingMessages: new Map(),
      nextPendingOrder: 0,
      abortController,
      emitRawSDKMessages: sessionMeta?.claudeCode?.emitRawSDKMessages ?? false,
      contextWindowSize:
        inferContextWindowFromModel(models.currentModelId) ?? DEFAULT_CONTEXT_WINDOW,
      taskState,
      toolUseCache: {},
      messageIdToUuid: new Map(),
      fastModeEnabled: initialFastModeEnabled,
      // Fresh sessions tail their transcript from the start; resumed/forked
      // sessions only from EOF (their transcript already contains history).
      anyharness: newAnyharnessSessionState(creationOpts.resume === undefined),
    };

    // If a hook already reported the transcript path (SessionStart fires
    // during initialization), start the goal tailer now. Otherwise it starts
    // when the path is first reported or when a goal is set.
    if (this.pendingTranscriptPaths.has(sessionId)) {
      this.ensureTranscriptTailer(sessionId, this.sessions[sessionId]);
    }
    // Drain idle-time SDK messages (injected instructions, cron wake turns).
    this.startIdlePump(sessionId);

    return {
      sessionId,
      modes,
      configOptions,
    };
  }
}

function shouldEmitRawMessage(
  config: boolean | SDKMessageFilter[],
  message: { type: string; subtype?: string; origin?: SDKMessageOrigin },
): boolean {
  if (config === true) return true;
  if (config === false) return false;
  return config.some(
    (f) =>
      f.type === message.type &&
      (f.subtype === undefined || f.subtype === message.subtype) &&
      (f.origin === undefined || f.origin === message.origin?.kind),
  );
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

/** Sum all four fields as a proxy for post-turn context occupancy: the current
 *  turn's output becomes next turn's input. Per the Anthropic API, input_tokens
 *  excludes cache tokens — cache_read and cache_creation are reported
 *  separately — so summing all four is not double-counting. */
function totalTokens(usage: UsageSnapshot): number {
  return (
    usage.input_tokens +
    usage.output_tokens +
    usage.cache_read_input_tokens +
    usage.cache_creation_input_tokens
  );
}

/**
 * Build the `data` payload attached to a `RequestError.internalError` when we
 * have a categorical error from the Claude SDK. Returns `undefined` when no
 * categorical error is available, matching the previous behavior of passing
 * `undefined` to `RequestError.internalError`.
 *
 * The `errorKind` field is a convention for ACP clients to dispatch on
 * without having to pattern-match the human-readable message text. Clients
 * that don't understand it fall back to the existing message-based rendering.
 */
function errorKindData(
  errorKind: SDKAssistantMessageError | undefined,
): { errorKind: SDKAssistantMessageError } | undefined {
  return errorKind ? { errorKind } : undefined;
}

/** Project a nullable API usage object into our non-null snapshot shape.
 *  Both SDK message_start and assistant message `usage` have `number | null`
 *  cache fields; we coerce absent values to 0 so `totalTokens` never hits
 *  NaN. `input_tokens`/`output_tokens` are typed `number` by the SDK but
 *  synthetic or third-party-backend stream events have been observed emitting
 *  them as null/undefined — coerce those too so a malformed upstream event
 *  can't leak NaN into the wire `used` field. Delta events have different
 *  semantics (cumulative + prev fallback) and are handled inline. */
function snapshotFromUsage(usage: {
  input_tokens?: number | null;
  output_tokens?: number | null;
  cache_read_input_tokens?: number | null;
  cache_creation_input_tokens?: number | null;
}): UsageSnapshot {
  return {
    input_tokens: usage.input_tokens ?? 0,
    output_tokens: usage.output_tokens ?? 0,
    cache_read_input_tokens: usage.cache_read_input_tokens ?? 0,
    cache_creation_input_tokens: usage.cache_creation_input_tokens ?? 0,
  };
}

function createEnvForGateway(request?: GatewayAuthRequest) {
  if (!request?._meta) {
    return {};
  }
  const customHeaders = Object.entries(request._meta.gateway.headers)
    .map(([key, value]) => `${key}: ${value}`)
    .join("\n");

  if (request.methodId === "gateway-bedrock") {
    return {
      CLAUDE_CODE_USE_BEDROCK: "1",
      AWS_BEARER_TOKEN_BEDROCK: " ", // Must be non-empty to bypass pass configuration check
      ANTHROPIC_BEDROCK_BASE_URL: request._meta.gateway.baseUrl,
      ANTHROPIC_CUSTOM_HEADERS: customHeaders,
    };
  }
  return {
    ANTHROPIC_BASE_URL: request._meta.gateway.baseUrl,
    ANTHROPIC_CUSTOM_HEADERS: customHeaders,
    ANTHROPIC_AUTH_TOKEN: " ", // Must be specified to bypass claude login requirement
  };
}

/**
 * Build the list of permission modes the agent will advertise for the given
 * model. `auto` is gated by `ModelInfo.supportsAutoMode === true`, which is
 * the SDK's model-level availability signal. `undefined`/`false` both exclude
 * `auto`. `bypassPermissions` is still gated by `ALLOW_BYPASS`.
 */
function buildAvailableModes(modelInfo: ModelInfo | undefined): SessionModeState["availableModes"] {
  const modes: SessionModeState["availableModes"] = [];

  // Only advertise "auto" when the SDK reports the model supports it.
  if (modelInfo?.supportsAutoMode === true) {
    modes.push({
      id: "auto",
      name: "Auto",
      description: "Use a model classifier to approve/deny permission prompts",
    });
  }

  modes.push(
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
  );

  if (ALLOW_BYPASS) {
    modes.push({
      id: "bypassPermissions",
      name: "Bypass Permissions",
      description: "Bypass all permission checks",
    });
  }

  return modes;
}

// Translate a UI effort value into the flag-layer payload. The SDK
// shallow-merges `applyFlagSettings`, drops `undefined` during JSON transport,
// and only clears a key when an explicit `null` is sent — see
// `applyFlagSettings` in @anthropic-ai/claude-agent-sdk. Mapping both the
// `"default"` sentinel and `undefined` (effort option absent for the model) to
// `null` ensures any previously-applied flag is actually cleared.
function toSdkEffortLevel(value: string | undefined): Settings["effortLevel"] | null {
  return value === undefined || value === "default" ? null : (value as Settings["effortLevel"]);
}

function buildConfigOptions(
  modes: SessionModeState,
  models: SessionModelState,
  modelInfos: ModelInfo[],
  currentEffortLevel?: string,
  currentFastModeEnabled?: boolean,
): SessionConfigOption[] {
  const options: SessionConfigOption[] = [
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

  // Add effort level option based on the currently selected model
  const currentModelInfo = modelInfos.find((m) => m.value === models.currentModelId);
  const supportedLevels = currentModelInfo?.supportsEffort
    ? (currentModelInfo.supportedEffortLevels ?? [])
    : [];

  if (supportedLevels.length > 0) {
    const effortOptions = [
      { value: "default", name: "Default" },
      ...supportedLevels.map((level) => ({
        value: level,
        name: level
          .split(/[_-]/)
          .map((part) => (part ? part.charAt(0).toUpperCase() + part.slice(1) : part))
          .join(" "),
      })),
    ];

    const includes = (l: string) => l === "default" || (supportedLevels as string[]).includes(l);
    const validEffort =
      currentEffortLevel && includes(currentEffortLevel) ? currentEffortLevel : "default";

    options.push({
      id: "effort",
      name: "Effort",
      description: "Available effort levels for this model",
      category: "thought_level",
      type: "select",
      currentValue: validEffort,
      options: effortOptions,
    });
  }

  // Add fast_mode option when the current model supports it
  if (currentModelInfo?.supportsFastMode) {
    options.push({
      id: "fast_mode",
      name: "Fast Mode",
      description: "Favor faster responses",
      category: "_fast_mode",
      type: "select",
      currentValue: currentFastModeEnabled === true ? "on" : "off",
      options: [
        { value: "off", name: "Off" },
        { value: "on", name: "On" },
      ],
    });
  }

  return options;
}

// Claude Code CLI persists display strings like "opus[1m]" in settings,
// but the SDK model list uses IDs like "claude-opus-4-6-1m".
const MODEL_CONTEXT_HINT_PATTERN = /\[(\d+m)\]$/i;

// Captures a model family version such as `4-6` or `4.7` so we can keep
// `claude-opus-4-6` from being copied onto the SDK's `opus` alias when that
// alias currently resolves to a different family version (e.g. Opus 4.7).
const MODEL_FAMILY_VERSION_PATTERN = /\b(\d+)[-.](\d+)\b/;

function extractModelFamilyVersion(s: string): string | null {
  const match = s.match(MODEL_FAMILY_VERSION_PATTERN);
  return match ? `${match[1]}.${match[2]}` : null;
}

function modelVersionsCompatible(preference: string, candidate: ModelInfo): boolean {
  const preferred = extractModelFamilyVersion(preference);
  if (!preferred) return true;
  const candidateVersion =
    extractModelFamilyVersion(candidate.value) ??
    extractModelFamilyVersion(candidate.displayName) ??
    extractModelFamilyVersion(candidate.description);
  if (!candidateVersion) return true;
  return preferred === candidateVersion;
}

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
  let nonHintMatched = false;
  for (const token of tokens) {
    if (haystack.includes(token)) {
      if (token !== contextHint) nonHintMatched = true;
      score += token === contextHint ? 3 : 1;
    }
  }
  if (contextHint && !nonHintMatched) return 0;
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
    if (!modelVersionsCompatible(trimmed, model)) return false;
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
    if (!modelVersionsCompatible(trimmed, model)) continue;
    const score = scoreModelMatch(model, tokens, contextHint);
    if (0 < score && (!bestMatch || bestScore < score)) {
      bestMatch = model;
      bestScore = score;
    }
  }

  return bestMatch;
}

function resolveSettingsModel(
  models: ModelInfo[],
  settingsModel: unknown,
  logger: Logger,
): ModelInfo | null {
  if (settingsModel === undefined) {
    return null;
  }
  if (typeof settingsModel !== "string") {
    const typeLabel = settingsModel === null ? "null" : typeof settingsModel;
    logger.error(`Ignoring model from settings: expected a string, got ${typeLabel}.`);
    return null;
  }
  return resolveModelPreference(models, settingsModel);
}

/**
 * Restrict the SDK's model list to the user's `availableModels` allowlist
 * (already merged-and-deduped across settings sources by `SettingsManager`).
 * The user's exact entries become the model IDs surfaced via configOptions
 * and passed to `setModel`, which prevents Claude Code from silently
 * substituting a date-pinned variant (e.g. `haiku` →
 * `claude-haiku-4-5-20251001`) that the user may not have access to.
 *
 * Display info and capability flags are copied from the closest SDK match so
 * the UI still renders sensible names and effort levels.
 *
 * Semantics from https://code.claude.com/docs/en/model-config#restrict-model-selection:
 * - `undefined` is handled by the caller (no allowlist applied).
 * - The Default option is unaffected by `availableModels` — it always remains
 *   available, even when the allowlist is `[]`.
 */
function applyAvailableModelsAllowlist(sdkModels: ModelInfo[], allowlist: string[]): ModelInfo[] {
  // Default is always preserved per the docs. Synthesize one if the SDK
  // didn't surface it so downstream code (e.g. `getAvailableModels` picking
  // `models[0]` as a fallback) still has something to work with.
  const defaultModel = sdkModels.find((m) => m.value === "default") ?? {
    value: "default",
    displayName: "Default",
    description: "",
  };
  const result: ModelInfo[] = [defaultModel];
  const seen = new Set<string>([defaultModel.value]);

  const sdkModelsWithoutDefault = sdkModels.filter((m) => m.value !== "default");

  for (const entry of allowlist) {
    const trimmed = entry.trim();
    if (!trimmed || seen.has(trimmed)) continue;

    const sdkMatch = resolveModelPreference(sdkModelsWithoutDefault, trimmed);
    if (sdkMatch) {
      result.push({ ...sdkMatch, value: trimmed });
    } else {
      result.push({ value: trimmed, displayName: trimmed, description: "" });
    }
    seen.add(trimmed);
  }

  return result;
}

async function getAvailableModels(
  query: Query,
  models: ModelInfo[],
  sdkModels: ModelInfo[],
  settingsManager: SettingsManager,
  logger: Logger,
): Promise<SessionModelState> {
  const settings = settingsManager.getSettings();

  let currentModel = models[0];
  let resolvedFromInput: string | undefined;

  // Model priority (highest to lowest):
  // 1. ANTHROPIC_MODEL environment variable
  // 2. settings.model (user configuration)
  // 3. models[0] (default first model)
  if (process.env.ANTHROPIC_MODEL) {
    const match = resolveModelPreference(models, process.env.ANTHROPIC_MODEL);
    if (match) {
      currentModel = match;
      resolvedFromInput = process.env.ANTHROPIC_MODEL;
    }
  } else if (typeof settings.model === "string") {
    const match = resolveSettingsModel(models, settings.model, logger);
    if (match) {
      currentModel = match;
      resolvedFromInput = settings.model;
    }
  }

  // Skip the setModel round-trip when we can prove the SDK has already landed
  // on the same model. Two cases qualify:
  //  (a) No override applied — currentModel stayed at models[0]; the SDK is on
  //      its own default and we have nothing to sync.
  //  (b) The resolver returned the user's input verbatim AND that value exists
  //      in the SDK's original model list — meaning no fuzzy match or
  //      allowlist rewrite was involved, and the SDK (which reads the same
  //      ANTHROPIC_MODEL / settings.json) will have arrived at the same entry.
  // Anything else (fuzzy match, allowlist-synthesized value, alias) gets a
  // setModel call so we don't drift from the user's intended pin.
  const sdkSawSameValue = sdkModels.some((m) => m.value === currentModel.value);
  const skipSetModel =
    resolvedFromInput === undefined ||
    (currentModel.value === resolvedFromInput && sdkSawSameValue);
  if (!skipSetModel) {
    await query.setModel(currentModel.value);
  }

  return {
    availableModels: models.map((model) => ({
      modelId: model.value,
      name: model.displayName,
      description: model.description,
    })),
    currentModelId: currentModel.value,
  };
}

function getAvailableSlashCommands(commands: SlashCommand[]): AvailableCommand[] {
  const UNSUPPORTED_COMMANDS = [
    "clear",
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
 * Resolves the ACP `messageId` for a Claude SDK message (live) or a persisted
 * transcript message (replay) so chunk grouping is identical in both views.
 *
 * Assistant turns are keyed by the Anthropic API message id (`message.id`),
 * which is identical at `message_start`, on the consolidated assistant message,
 * and in the persisted transcript — unlike the per-`stream_event` uuid, which is
 * unique per event and never persisted. User messages have no API id, but they
 * are never streamed, so their (stable) SDK uuid is used instead. ACP message
 * ids are opaque strings, so no particular format is required.
 */
export function messageIdForGrouping(message: {
  type?: string;
  uuid?: string | null;
  message?: unknown;
}): string | undefined {
  if (message.type === "assistant") {
    const inner = message.message;
    const apiId =
      inner && typeof inner === "object" && "id" in inner
        ? (inner as { id?: unknown }).id
        : undefined;
    if (typeof apiId === "string" && apiId.length > 0) {
      return apiId;
    }
  }
  return typeof message.uuid === "string" && message.uuid.length > 0 ? message.uuid : undefined;
}

/**
 * Stamps an ACP `messageId` onto a session update, but only on the message/
 * thought chunk variants that carry one — tool_call/plan/etc. updates never do.
 * No-op when `messageId` is falsy, so callers can pass it through unconditionally.
 */
function applyMessageId(
  update: SessionNotification["update"],
  messageId: string | undefined,
): void {
  if (
    messageId &&
    (update.sessionUpdate === "agent_message_chunk" ||
      update.sessionUpdate === "user_message_chunk" ||
      update.sessionUpdate === "agent_thought_chunk")
  ) {
    update.messageId = messageId;
  }
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
    taskState?: TaskState;
    // Opaque id identifying the message these chunks belong to (ACP message ids
    // are opaque strings — no particular format is required). Attached to
    // user/agent message and thought chunks so clients can group streamed chunks
    // into a single message. Omit it (leave undefined) when unknown — never send
    // an explicit `null`.
    messageId?: string;
  },
): SessionNotification[] {
  const taskState = options?.taskState ?? new Map();
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
    applyMessageId(update, options?.messageId);

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
          // @ts-expect-error - sometimes input is empty object or undefined
          if (Array.isArray(chunk.input?.todos)) {
            update = {
              sessionUpdate: "plan",
              entries: planEntries(chunk.input as { todos: ClaudePlanEntry[] }),
            };
          }
        } else if (
          chunk.name === "TaskCreate" ||
          chunk.name === "TaskUpdate" ||
          chunk.name === "TaskList" ||
          chunk.name === "TaskGet"
        ) {
          // Task* tool_use is suppressed; the plan update is emitted at
          // tool_result time once we have the task ID (for TaskCreate) and
          // confirmation that the change took effect.
        } else {
          // Only register hooks on first encounter to avoid double-firing
          if (registerHooks && !alreadyCached) {
            // Capture the tool name in the closure rather than re-reading the
            // cache when the hook fires. The cache entry is pruned at
            // tool_result time, and a PostToolUse hook can fire after that, so
            // closing over the name keeps the diff working without depending on
            // (or pinning) the cache entry's lifetime.
            const toolName = chunk.name;
            registerHookCallback(chunk.id, {
              onPostToolUseHook: async (toolUseId, toolInput, toolResponse) => {
                // Both `Edit` and `Write` produce a structuredPatch in their
                // PostToolUse tool_response. For Edit the diff replaces the
                // optimistic content built at tool_use time. For Write the
                // optimistic content (built from `input.content` alone with
                // `oldText: null`) shows "creation" semantics regardless of
                // whether the file existed; the structuredPatch from the
                // hook lets us emit the real diff for `type: "update"`. The
                // helper returns `{}` if the response shape isn't usable.
                const editDiff =
                  toolName === "Edit" || toolName === "Write"
                    ? toolUpdateFromDiffToolResponse(toolResponse)
                    : {};
                const update: SessionNotification["update"] = {
                  _meta: {
                    claudeCode: {
                      toolResponse,
                      toolName,
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

        if (
          toolUse.name === "TaskCreate" ||
          toolUse.name === "TaskUpdate" ||
          toolUse.name === "TaskList" ||
          toolUse.name === "TaskGet"
        ) {
          // Headless/SDK sessions emit Task* tools instead of TodoWrite.
          // TaskCreate / TaskUpdate mutate the accumulated task list; TaskList
          // and TaskGet are read-only so we just suppress their tool_call /
          // tool_result events. The plan update is emitted as a snapshot of
          // the accumulated state, mirroring the legacy TodoWrite behavior.
          const isError = "is_error" in chunk && chunk.is_error;
          if (!isError) {
            if (toolUse.name === "TaskCreate") {
              applyTaskCreate(
                taskState,
                toolUse.input as Parameters<typeof applyTaskCreate>[1],
                parseTaskCreateOutput(chunk.content),
              );
            } else if (toolUse.name === "TaskUpdate") {
              applyTaskUpdate(taskState, toolUse.input as Parameters<typeof applyTaskUpdate>[1]);
            }
          }
          if (!isError && (toolUse.name === "TaskCreate" || toolUse.name === "TaskUpdate")) {
            update = {
              sessionUpdate: "plan",
              entries: taskStateToPlanEntries(taskState),
            };
          }
        } else if (toolUse.name !== "TodoWrite") {
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
        // The tool_use is fully resolved now — drop it so a long session doesn't
        // retain every tool call. The PostToolUse hook (Edit/Write diffs) closes
        // over the tool name and no longer reads the cache, so pruning here is
        // safe regardless of hook/result ordering.
        delete toolUseCache[chunk.tool_use_id];
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
      case "advisor_tool_result":
      case "mid_conv_system":
      case "fallback":
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
      applyMessageId(update, options?.messageId);
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
    taskState?: TaskState;
    messageId?: string;
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
          taskState: options?.taskState,
          messageId: options?.messageId,
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
          taskState: options?.taskState,
          messageId: options?.messageId,
        },
      );
    // No content. `ping` is a Messages-API keep-alive event that the SDK's
    // `BetaRawMessageStreamEvent` union doesn't include even though the
    // wire format emits it; the `as never` cast lets us no-op it here
    // instead of letting it fall through to `unreachable`.
    case "ping" as never:
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
  let agent!: ClaudeAcpAgent;
  const connection = new AgentSideConnection((client) => {
    agent = new ClaudeAcpAgent(client);
    return agent;
  }, stream);
  return { connection, agent };
}

function commonPrefixLength(a: string, b: string) {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) {
    i++;
  }
  return i;
}

/** Best-effort first guess of a model's context window from its ID, used only
 *  as a fallback when the SDK's authoritative `getContextUsage` is unavailable
 *  (and until a `result` message arrives with the `modelUsage` value).
 *  Anthropic 1M-context variants encode "1m" as a distinct token in the SDK
 *  model ID (e.g., "claude-opus-4-6-1m"), which `\b1m\b` catches without also
 *  matching things like "10m" or embedded substrings. */
function inferContextWindowFromModel(model: string): number | null {
  if (/\b1m\b/i.test(model)) return 1_000_000;
  return null;
}

/** Fetch the SDK's authoritative context-window occupancy via the
 *  `getContextUsage` control request. Unlike the per-message API usage numbers
 *  (which only count message tokens), this `totalTokens` includes the system
 *  prompt, tool schemas, MCP tools, and memory-file overhead — the real
 *  occupancy the user sees. Returns `null` on any control-request failure.
 *
 *  Note: we deliberately do NOT use this response's window fields for `size`.
 *  They have been observed to under-report extended (1M) context windows, so
 *  the window keeps coming from `modelUsage` / `inferContextWindowFromModel`,
 *  which handle the 1M variants correctly. */
async function fetchContextUsedTokens(query: Query, logger: Logger): Promise<number | null> {
  try {
    const usage = await query.getContextUsage();
    return usage.totalTokens;
  } catch (error) {
    logger.error("Failed to fetch context usage from SDK:", error);
    return null;
  }
}

/** Translate the legacy `MAX_THINKING_TOKENS` env var into the SDK's `thinking`
 *  option. The `maxThinkingTokens` option it used to feed is deprecated and
 *  reduced to on/off on current models, so map the value to explicit thinking
 *  config instead: unset → `undefined` (SDK default, adaptive on models that
 *  support it); `0` → disabled; a positive integer → a fixed token budget.
 *  Anything else is ignored with a warning. */
function resolveThinkingConfig(
  raw: string | undefined,
  logger: Logger,
): ThinkingConfig | undefined {
  if (raw === undefined) return undefined;
  const parsed = Number.parseInt(raw, 10);
  if (Number.isNaN(parsed) || parsed < 0) {
    logger.error(`Ignoring MAX_THINKING_TOKENS: expected a non-negative integer, got '${raw}'.`);
    return undefined;
  }
  return parsed === 0 ? { type: "disabled" } : { type: "enabled", budgetTokens: parsed };
}

function parseModelConfig(
  raw: string | undefined,
): { modelOverrides?: Record<string, string>; availableModels?: string[] } | undefined {
  if (!raw) return undefined;
  const parsed = JSON.parse(raw);
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("CLAUDE_MODEL_CONFIG must be a JSON object");
  }
  const result: { modelOverrides?: Record<string, string>; availableModels?: string[] } = {};
  if (parsed.modelOverrides !== undefined) result.modelOverrides = parsed.modelOverrides;
  if (parsed.availableModels !== undefined) result.availableModels = parsed.availableModels;
  return Object.keys(result).length > 0 ? result : undefined;
}

function getMatchingModelUsage(modelUsage: Record<string, ModelUsage>, currentModel: string) {
  let bestKey: string | null = null;
  let bestLen = 0;

  for (const key of Object.keys(modelUsage)) {
    const len = commonPrefixLength(key, currentModel);
    if (len > bestLen) {
      bestLen = len;
      bestKey = key;
    }
  }

  if (bestKey) {
    return modelUsage[bestKey];
  }
}
