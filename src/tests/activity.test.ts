import { describe, it, expect, vi } from "vitest";
import { randomUUID } from "node:crypto";

vi.mock("@anthropic-ai/claude-agent-sdk", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@anthropic-ai/claude-agent-sdk")>();
  return {
    ...actual,
    deleteSession: vi.fn(),
    getSessionInfo: vi.fn().mockRejectedValue(new Error("no session file in this test")),
    getSessionMessages: vi.fn(actual.getSessionMessages),
  };
});

import { ClaudeAcpAgent, type AcpClient } from "../acp-agent.js";
import { Pushable } from "../utils.js";
import {
  ACTIVITY_SCHEMA_VERSION,
  newActivityState,
  parseActivityListRequest,
  parseBackgroundOutputFile,
  processWire,
  PROCESS_UPSERTED_TRANSCRIPT_EVENT,
  subagentWire,
  SUBAGENT_UPSERTED_TRANSCRIPT_EVENT,
  upsertProcess,
  upsertSubagent,
} from "../activity.js";
import type { SessionNotification } from "@agentclientprotocol/sdk";

// ---------------------------------------------------------------------------
// Unit tests: the pure helpers in ../activity.ts
// ---------------------------------------------------------------------------

describe("activity.ts: parseBackgroundOutputFile (background-bash feed materialization)", () => {
  // Live-verified phrasing (old fork fix 3fba2f5, still the shape this SDK
  // version's raw tool_result text uses for a backgrounded Bash command).
  it("matches 'Command running in background with ID: <id>. Output is being written to: <path>'", () => {
    const text =
      "Command running in background with ID: bg-1. Output is being written to: /tmp/claude-bg-1.log";
    expect(parseBackgroundOutputFile(text)).toBe("/tmp/claude-bg-1.log");
  });

  it("matches 'logs to <path>' phrasing", () => {
    expect(parseBackgroundOutputFile("Backgrounded; logs to /tmp/out.log")).toBe("/tmp/out.log");
  });

  it("matches the legacy 'output: <path>' / 'output -> <path>' forms", () => {
    expect(parseBackgroundOutputFile("output: /tmp/legacy.log")).toBe("/tmp/legacy.log");
    expect(parseBackgroundOutputFile("output -> /tmp/legacy2.log")).toBe("/tmp/legacy2.log");
  });

  it("strips trailing punctuation from the captured path", () => {
    expect(parseBackgroundOutputFile("Output is being written to: /tmp/out.log.")).toBe(
      "/tmp/out.log",
    );
  });

  it("extracts text from an array-of-blocks tool_result content", () => {
    const content = [{ type: "text", text: "Output is being written to: /tmp/arr.log" }];
    expect(parseBackgroundOutputFile(content)).toBe("/tmp/arr.log");
  });

  it("returns null when no output-file notice is present", () => {
    expect(parseBackgroundOutputFile("just some regular stdout")).toBeNull();
    expect(parseBackgroundOutputFile(null)).toBeNull();
    expect(parseBackgroundOutputFile(undefined)).toBeNull();
  });
});

describe("activity.ts: upsertProcess / upsertSubagent (merge-only-defined-fields)", () => {
  it("creates a process record with the documented defaults, then merges only defined patch fields", () => {
    const state = newActivityState();
    const created = upsertProcess(state, "proc-1", { command: "npm test", status: "running" });
    expect(created).toMatchObject({
      id: "proc-1",
      command: "npm test",
      cwd: null,
      status: "running",
      exitCode: null,
      pid: null,
      endedAtMs: null,
      feed: null,
    });

    // A later partial patch (no `command`, no `cwd`) must not blank out the
    // fields the launch event already set.
    const updated = upsertProcess(state, "proc-1", {
      status: "exited",
      endedAtMs: 1000,
      feed: { transport: "tail_file", path: "/tmp/out.log" },
    });
    expect(updated.command).toBe("npm test");
    expect(updated.status).toBe("exited");
    expect(updated.endedAtMs).toBe(1000);
    expect(updated.feed).toEqual({ transport: "tail_file", path: "/tmp/out.log" });
    // Same map entry, not a second one.
    expect(state.processes.size).toBe(1);
  });

  it("creates a subagent record with the documented defaults (background true by default)", () => {
    const state = newActivityState();
    const created = upsertSubagent(state, "agent-1", {
      agentType: "Explore",
      description: "Investigate",
    });
    expect(created).toMatchObject({
      id: "agent-1",
      agentType: "Explore",
      description: "Investigate",
      model: null,
      background: true,
      status: "running",
      summary: null,
      tokensUsed: null,
      toolCalls: null,
      durationSeconds: null,
      feed: null,
    });
  });

  it("merges flat usage fields onto an existing subagent without touching other fields", () => {
    const state = newActivityState();
    upsertSubagent(state, "agent-1", { agentType: "Explore", background: true });
    const updated = upsertSubagent(state, "agent-1", {
      tokensUsed: 1234,
      toolCalls: 5,
      durationSeconds: 8,
      status: "completed",
    });
    expect(updated).toMatchObject({
      agentType: "Explore",
      background: true,
      status: "completed",
      tokensUsed: 1234,
      toolCalls: 5,
      durationSeconds: 8,
    });
  });
});

describe("activity.ts: processWire / subagentWire (wire projection)", () => {
  it("strips updatedAtMs and keeps exactly the wire.rs field set for a process", () => {
    const state = newActivityState();
    const record = upsertProcess(state, "proc-1", { command: "ls", status: "running" });
    const wire = processWire(record);
    expect(Object.keys(wire).sort()).toEqual(
      [
        "id",
        "command",
        "cwd",
        "status",
        "exitCode",
        "pid",
        "startedAtMs",
        "endedAtMs",
        "feed",
      ].sort(),
    );
    expect((wire as unknown as { updatedAtMs?: unknown }).updatedAtMs).toBeUndefined();
  });

  it("strips updatedAtMs and keeps exactly the wire.rs field set for a subagent", () => {
    const state = newActivityState();
    const record = upsertSubagent(state, "agent-1", { agentType: "Explore" });
    const wire = subagentWire(record);
    expect(Object.keys(wire).sort()).toEqual(
      [
        "id",
        "agentType",
        "description",
        "model",
        "background",
        "status",
        "summary",
        "tokensUsed",
        "toolCalls",
        "durationSeconds",
        "feed",
      ].sort(),
    );
    expect((wire as unknown as { updatedAtMs?: unknown }).updatedAtMs).toBeUndefined();
  });
});

describe("activity.ts: parseActivityListRequest", () => {
  it("accepts a well-formed request", () => {
    expect(parseActivityListRequest({ sessionId: "abc" })).toEqual({ sessionId: "abc" });
  });

  it("rejects a missing/empty/non-string sessionId", () => {
    expect(() => parseActivityListRequest({})).toThrow();
    expect(() => parseActivityListRequest({ sessionId: "" })).toThrow();
    expect(() => parseActivityListRequest({ sessionId: 42 })).toThrow();
    expect(() => parseActivityListRequest(null)).toThrow();
  });
});

// ---------------------------------------------------------------------------
// Integration tests: drive ClaudeAcpAgent's long-lived consumer through a
// mocked SDK message stream, exactly as acp-agent.test.ts's own
// injectGeneratorSession/mockSessionState harness does (duplicated here in
// miniature since those helpers aren't exported from that file).
// ---------------------------------------------------------------------------

/** Build the replayed `user` message the SDK echoes back for a pushed prompt. */
function userEcho(u: any) {
  return {
    type: "user",
    message: u.message,
    parent_tool_use_id: null,
    uuid: u.uuid,
    session_id: "test-session",
    isReplay: true,
  };
}

function wrapQuery(generator: AsyncGenerator<any>) {
  return Object.assign(generator, {
    interrupt: vi.fn(async () => {}),
    close: vi.fn(),
    setModel: vi.fn(async () => {}),
  }) as any;
}

function mockSessionState(overrides: Record<string, any> = {}) {
  return {
    cancelled: false,
    cwd: "/test",
    sessionFingerprint: JSON.stringify({ cwd: "/test", mcpServers: [] }),
    modes: { currentModeId: "default", availableModes: [] },
    models: { currentModelId: "default", availableModels: [] },
    modelInfos: [],
    settingsManager: { dispose: vi.fn() },
    accumulatedUsage: {
      inputTokens: 0,
      outputTokens: 0,
      cachedReadTokens: 0,
      cachedWriteTokens: 0,
    },
    configOptions: [],
    agents: [],
    currentAgent: "default",
    abortController: new AbortController(),
    emitRawSDKMessages: false,
    forwardSubagentText: false,
    contextWindowSize: 200000,
    contextWindowAuthoritative: false,
    providerCacheKey: "default",
    taskState: new Map(),
    toolUseCache: {},
    emittedToolCalls: new Set(),
    liveBackgroundTasks: new Map(),
    activity: newActivityState(),
    emittedAssistantText: false,
    owedTrailingIdles: 0,
    messageIdToUuid: new Map(),
    ...overrides,
  } as any;
}

function injectGeneratorSession(
  agent: ClaudeAcpAgent,
  makeGenerator: (input: Pushable<any>) => AsyncGenerator<any>,
  overrides: Record<string, any> = {},
) {
  const input = new Pushable<any>();
  agent.sessions["test-session"] = mockSessionState({
    query: wrapQuery(makeGenerator(input)),
    input,
    ...overrides,
  });
  return input;
}

/** Replays the prompt's user echo (so the turn activates), then the given
 *  messages, then settles the turn at an idle. */
function makeGenerator(messages: unknown[]) {
  return async function* (input: Pushable<any>) {
    const iter = input[Symbol.asyncIterator]();
    const { value: userMessage, done } = await iter.next();
    if (!done && userMessage) {
      yield userEcho(userMessage);
    }
    yield* messages as any;
    yield { type: "system", subtype: "session_state_changed", state: "idle" };
  };
}

function successResult(overrides: Record<string, any> = {}) {
  return {
    type: "result" as const,
    subtype: "success" as const,
    stop_reason: null,
    is_error: false,
    result: "",
    errors: [],
    duration_ms: 0,
    duration_api_ms: 0,
    num_turns: 1,
    total_cost_usd: 0,
    usage: {
      input_tokens: 10,
      output_tokens: 5,
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
    },
    modelUsage: {},
    permission_denials: [],
    uuid: randomUUID(),
    session_id: "test-session",
    ...overrides,
  };
}

function taskStarted(overrides: Record<string, any>) {
  return {
    type: "system",
    subtype: "task_started",
    task_id: overrides.task_id,
    tool_use_id: overrides.tool_use_id,
    description: overrides.description ?? "Working",
    subagent_type: overrides.subagent_type,
    uuid: randomUUID(),
    session_id: "test-session",
  };
}

function taskProgress(overrides: Record<string, any>) {
  return {
    type: "system",
    subtype: "task_progress",
    task_id: overrides.task_id,
    tool_use_id: overrides.tool_use_id,
    description: overrides.description ?? "Working",
    subagent_type: overrides.subagent_type,
    usage: overrides.usage ?? { total_tokens: 100, tool_uses: 2, duration_ms: 1500 },
    summary: overrides.summary,
    uuid: randomUUID(),
    session_id: "test-session",
  };
}

function taskNotification(overrides: Record<string, any>) {
  return {
    type: "system",
    subtype: "task_notification",
    task_id: overrides.task_id,
    tool_use_id: overrides.tool_use_id,
    status: overrides.status ?? "completed",
    output_file: overrides.output_file ?? "",
    summary: overrides.summary ?? "done",
    usage: overrides.usage,
    uuid: randomUUID(),
    session_id: "test-session",
  };
}

/** A tool_use content block, cached by the streamed-assistant path so later
 *  task_started/tool_result handling can look up its name/input. */
function toolUseBlock(id: string, name: string, input: unknown) {
  return { type: "tool_use", id, name, input };
}

function assistantMessage(content: unknown[]) {
  return {
    type: "assistant",
    parent_tool_use_id: null,
    uuid: randomUUID(),
    session_id: "test-session",
    message: {
      role: "assistant",
      model: "claude-sonnet-4-5",
      stop_reason: null,
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      content,
    },
  };
}

/** A `user` message carrying one tool_result, with an optional message-level
 *  `tool_use_result` (the structured BashOutput/AgentOutput). */
function toolResultMessage(toolUseId: string, content: unknown, toolUseResult?: unknown) {
  return {
    type: "user",
    parent_tool_use_id: null,
    uuid: randomUUID(),
    session_id: "test-session",
    message: {
      role: "user",
      content: [{ type: "tool_result", tool_use_id: toolUseId, content }],
    },
    tool_use_result: toolUseResult,
  };
}

/** Poll a condition across microtask/timer turns (mirrors acp-agent.test.ts's
 *  own local `waitFor` — the long-lived consumer keeps processing messages
 *  after `prompt()` resolves, so post-turn assertions need this). */
async function waitFor(cond: () => boolean) {
  for (let i = 0; i < 200; i++) {
    if (cond()) {
      return;
    }
    await new Promise((r) => setTimeout(r, 0));
  }
  throw new Error("waitFor timed out");
}

function createCapturingAgent() {
  const updates: SessionNotification[] = [];
  const mockClient = {
    sessionUpdate: async (n: SessionNotification) => {
      updates.push(n);
    },
  } as unknown as AcpClient;
  const agent = new ClaudeAcpAgent(mockClient, { log: () => {}, error: () => {} });
  return { agent, updates };
}

/** A client whose `sessionUpdate` rejects for every roster chunk — simulating
 *  the peer winding the connection down while a post-turn background-Bash
 *  settle notification is emitted — but still succeeds for ordinary
 *  (non-roster) updates, so the rest of the turn's delivery is unaffected.
 *  Roster chunks are the `transcriptEvent`-tagged `_meta.anyharness` updates
 *  (the same discriminator the runtime's NON_TRANSCRIPT_CHUNK_EVENTS gate keys
 *  on); tool_call updates also carry `_meta.anyharness` (nativeToolName/
 *  toolKind/parentToolCallId, no transcriptEvent) and must NOT trip this mock —
 *  they flow through the unguarded canonical send path by design. Captures
 *  every `logger.error` call so a test can assert the rejection was logged
 *  rather than silently dropped. */
function createRejectingCapturingAgent() {
  const updates: SessionNotification[] = [];
  const errors: unknown[] = [];
  const mockClient = {
    sessionUpdate: async (n: SessionNotification) => {
      const meta = (n.update as any)?._meta?.anyharness;
      if (meta?.transcriptEvent) {
        throw new Error("simulated peer disconnect during sessionUpdate");
      }
      updates.push(n);
    },
  } as unknown as AcpClient;
  const agent = new ClaudeAcpAgent(mockClient, {
    log: () => {},
    error: (message: unknown) => {
      errors.push(message);
    },
  });
  return { agent, updates, errors };
}

function activityChunks(updates: SessionNotification[], transcriptEvent?: string) {
  return updates.filter((n) => {
    const meta = (n.update as any)?._meta?.anyharness;
    // `transcriptEvent` presence, not `_meta.anyharness` presence, is what
    // makes an update a roster chunk — tool_call updates carry non-roster
    // `_meta.anyharness` stamps too (see createRejectingCapturingAgent's doc).
    return (
      meta?.transcriptEvent !== undefined &&
      (transcriptEvent === undefined || meta.transcriptEvent === transcriptEvent)
    );
  });
}

describe("activity emission: background Bash processes", () => {
  it("emits process_upserted on launch (task_started) with the exact wire shape", async () => {
    const { agent, updates } = createCapturingAgent();
    injectGeneratorSession(
      agent,
      makeGenerator([
        assistantMessage([toolUseBlock("toolu_bash1", "Bash", { command: "sleep 30 &" })]),
        taskStarted({ task_id: "proc-1", tool_use_id: "toolu_bash1" }),
        successResult(),
      ]),
    );

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "go" }] });

    const chunks = activityChunks(updates, PROCESS_UPSERTED_TRANSCRIPT_EVENT);
    expect(chunks).toHaveLength(1);
    const update = chunks[0].update as any;
    expect(update.sessionUpdate).toBe("agent_message_chunk");
    expect(update.content).toEqual({ type: "text", text: "" });
    expect(update._meta.anyharness).toEqual({
      schemaVersion: ACTIVITY_SCHEMA_VERSION,
      transcriptEvent: "process_upserted",
      process: {
        id: "proc-1",
        command: "sleep 30 &",
        cwd: "/test",
        status: "running",
        exitCode: null,
        pid: null,
        startedAtMs: expect.any(Number),
        endedAtMs: null,
        feed: null,
      },
    });
  });

  it("materializes the live-tail feed from the backgrounded Bash tool_result's raw text", async () => {
    const { agent, updates } = createCapturingAgent();
    injectGeneratorSession(
      agent,
      makeGenerator([
        assistantMessage([toolUseBlock("toolu_bash1", "Bash", { command: "sleep 30 &" })]),
        taskStarted({ task_id: "proc-1", tool_use_id: "toolu_bash1" }),
        toolResultMessage(
          "toolu_bash1",
          "Command running in background with ID: proc-1. Output is being written to: /tmp/proc-1.log",
          { stdout: "", stderr: "", interrupted: false, backgroundTaskId: "proc-1" },
        ),
        successResult(),
      ]),
    );

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "go" }] });

    const chunks = activityChunks(updates, PROCESS_UPSERTED_TRANSCRIPT_EVENT);
    const feeds = chunks
      .map((c) => (c.update as any)._meta.anyharness.process.feed)
      .filter(Boolean);
    expect(feeds).toContainEqual({ transport: "tail_file", path: "/tmp/proc-1.log" });
  });

  it("emits an exited process_upserted on task_notification, carrying the settle-time output_file feed", async () => {
    const { agent, updates } = createCapturingAgent();
    injectGeneratorSession(
      agent,
      makeGenerator([
        assistantMessage([toolUseBlock("toolu_bash1", "Bash", { command: "sleep 1 &" })]),
        taskStarted({ task_id: "proc-1", tool_use_id: "toolu_bash1" }),
        successResult(),
        taskNotification({
          task_id: "proc-1",
          output_file: "/tmp/proc-1.log",
          status: "completed",
        }),
      ]),
    );

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "go" }] });
    await agent.sessions["test-session"]?.consumer;

    const chunks = activityChunks(updates, PROCESS_UPSERTED_TRANSCRIPT_EVENT);
    const settled = chunks[chunks.length - 1].update as any;
    expect(settled._meta.anyharness.process).toMatchObject({
      id: "proc-1",
      status: "exited",
      feed: { transport: "tail_file", path: "/tmp/proc-1.log" },
    });
  });

  it("emits the settled process_upserted AFTER prompt() has already resolved (post-turn completion)", async () => {
    const { agent, updates } = createCapturingAgent();
    injectGeneratorSession(agent, (input) => {
      async function* messageGenerator() {
        const iter = input[Symbol.asyncIterator]();
        const { value: userMessage } = await iter.next();
        yield userEcho(userMessage);
        yield assistantMessage([toolUseBlock("toolu_bash1", "Bash", { command: "sleep 30 &" })]);
        yield taskStarted({ task_id: "proc-1", tool_use_id: "toolu_bash1" });
        // The turn settles here — a background Bash process never holds the
        // turn open (only subagents do; see Turn.deferredSettle).
        yield successResult();
        yield { type: "system", subtype: "session_state_changed", state: "idle" };
        // Strictly AFTER the turn's own idle: an unsolicited post-turn update
        // the long-lived consumer keeps draining regardless of prompt()'s
        // response boundary.
        yield taskNotification({ task_id: "proc-1", output_file: "/tmp/proc-1.log" });
      }
      return messageGenerator();
    });

    const beforePostTurnCount = () =>
      activityChunks(updates, PROCESS_UPSERTED_TRANSCRIPT_EVENT).length;

    const response = await agent.prompt({
      sessionId: "test-session",
      prompt: [{ type: "text", text: "go" }],
    });
    expect(response.stopReason).toBeDefined();
    // Exactly the launch emission so far — the settle hasn't streamed yet.
    expect(beforePostTurnCount()).toBe(1);

    await waitFor(() => beforePostTurnCount() === 2);
    const settled = activityChunks(updates, PROCESS_UPSERTED_TRANSCRIPT_EVENT).at(-1)!
      .update as any;
    expect(settled._meta.anyharness.process.status).toBe("exited");
    await agent.sessions["test-session"]?.consumer;
  });
});

describe("activity emission: subagents (Agent/Task tool)", () => {
  it("emits subagent_upserted on launch, progress (flat usage fields), and completion", async () => {
    const { agent, updates } = createCapturingAgent();
    injectGeneratorSession(
      agent,
      makeGenerator([
        assistantMessage([
          toolUseBlock("toolu_agent1", "Task", {
            description: "Explore",
            prompt: "explore the repo",
            subagent_type: "Explore",
            model: "sonnet",
          }),
        ]),
        taskStarted({
          task_id: "agent-1",
          tool_use_id: "toolu_agent1",
          subagent_type: "Explore",
          description: "Explore",
        }),
        taskProgress({
          task_id: "agent-1",
          tool_use_id: "toolu_agent1",
          subagent_type: "Explore",
          usage: { total_tokens: 4200, tool_uses: 7, duration_ms: 3207 },
          summary: "reading files",
        }),
        taskNotification({
          task_id: "agent-1",
          tool_use_id: "toolu_agent1",
          status: "completed",
          output_file: "/tmp/agent-1.jsonl",
          summary: "done exploring",
          usage: { total_tokens: 5000, tool_uses: 9, duration_ms: 4000 },
        }),
        successResult(),
      ]),
    );

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "go" }] });

    const chunks = activityChunks(updates, SUBAGENT_UPSERTED_TRANSCRIPT_EVENT);
    expect(chunks.length).toBeGreaterThanOrEqual(3);

    // Launch.
    expect((chunks[0].update as any)._meta.anyharness.subagent).toMatchObject({
      id: "agent-1",
      agentType: "Explore",
      description: "Explore",
      model: "sonnet",
      background: true,
      status: "running",
    });

    // Progress: flat sibling fields (tokensUsed/toolCalls/durationSeconds),
    // NOT nested under a `usage` object — the exact bug wire.rs's own
    // `opt_i64_lenient` doc calls out for the old fork (fix 5974047).
    const progress = (chunks[1].update as any)._meta.anyharness.subagent;
    expect(progress.tokensUsed).toBe(4200);
    expect(progress.toolCalls).toBe(7);
    expect(progress.durationSeconds).toBe(3); // round(3207 / 1000)
    expect(progress.usage).toBeUndefined();

    // Completion: settled status + flat usage + feed from output_file.
    const completed = (chunks[2].update as any)._meta.anyharness.subagent;
    expect(completed).toMatchObject({
      id: "agent-1",
      status: "completed",
      summary: "done exploring",
      tokensUsed: 5000,
      toolCalls: 9,
      durationSeconds: 4,
      feed: { transport: "tail_file", path: "/tmp/agent-1.jsonl" },
    });
  });

  it("maps a failed/stopped task_notification to the wire contract's 'failed' status", async () => {
    const { agent, updates } = createCapturingAgent();
    injectGeneratorSession(
      agent,
      makeGenerator([
        taskStarted({ task_id: "agent-2", tool_use_id: "toolu_agent2", subagent_type: "Explore" }),
        taskNotification({
          task_id: "agent-2",
          status: "stopped",
          output_file: "/tmp/agent-2.jsonl",
        }),
        successResult(),
      ]),
    );

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "go" }] });

    const chunks = activityChunks(updates, SUBAGENT_UPSERTED_TRANSCRIPT_EVENT);
    expect(chunks.at(-1)!.update as any).toMatchObject({
      _meta: { anyharness: { subagent: { status: "failed" } } },
    });
  });

  it("materializes an async-launched subagent's feed from the Agent tool's structured AgentOutput", async () => {
    const { agent, updates } = createCapturingAgent();
    injectGeneratorSession(
      agent,
      makeGenerator([
        assistantMessage([
          toolUseBlock("toolu_agent3", "Agent", {
            description: "Long research",
            prompt: "research thing",
            subagent_type: "general",
            run_in_background: true,
          }),
        ]),
        taskStarted({
          task_id: "agent-3",
          tool_use_id: "toolu_agent3",
          subagent_type: "general",
          description: "Long research",
        }),
        toolResultMessage("toolu_agent3", "agentId: agent-3 (use SendMessage to check in)", {
          status: "async_launched",
          isAsync: true,
          agentId: "agent-3",
          description: "Long research",
          prompt: "research thing",
          outputFile: "/tmp/agent-3-async.jsonl",
        }),
        successResult(),
      ]),
    );

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "go" }] });

    const chunks = activityChunks(updates, SUBAGENT_UPSERTED_TRANSCRIPT_EVENT);
    const withFeed = chunks.find((c) => (c.update as any)._meta.anyharness.subagent.feed);
    expect(withFeed).toBeDefined();
    expect((withFeed!.update as any)._meta.anyharness.subagent).toMatchObject({
      id: "agent-3",
      background: true,
      status: "running",
      feed: { transport: "tail_file", path: "/tmp/agent-3-async.jsonl" },
    });
  });

  it("maps a synchronous completed subagent's totals onto the flat usage fields, background:false", async () => {
    const { agent, updates } = createCapturingAgent();
    injectGeneratorSession(
      agent,
      makeGenerator([
        assistantMessage([
          toolUseBlock("toolu_agent4", "Agent", {
            description: "Quick lookup",
            prompt: "look this up",
            subagent_type: "general",
            run_in_background: false,
          }),
        ]),
        taskStarted({
          task_id: "agent-4",
          tool_use_id: "toolu_agent4",
          subagent_type: "general",
          description: "Quick lookup",
        }),
        toolResultMessage("toolu_agent4", "the answer", {
          status: "completed",
          agentId: "agent-4",
          agentType: "general",
          content: [{ type: "text", text: "the answer" }],
          resolvedModel: "claude-haiku-4-5",
          totalToolUseCount: 3,
          totalDurationMs: 2500,
          totalTokens: 900,
          usage: {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: null,
            cache_read_input_tokens: null,
            server_tool_use: null,
            service_tier: null,
            cache_creation: null,
          },
          status_: "completed",
          prompt: "look this up",
        }),
        successResult(),
      ]),
    );

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "go" }] });

    const chunks = activityChunks(updates, SUBAGENT_UPSERTED_TRANSCRIPT_EVENT);
    const completed = chunks.find(
      (c) => (c.update as any)._meta.anyharness.subagent.status === "completed",
    );
    expect(completed).toBeDefined();
    expect((completed!.update as any)._meta.anyharness.subagent).toMatchObject({
      id: "agent-4",
      background: false,
      status: "completed",
      model: "claude-haiku-4-5",
      tokensUsed: 900,
      toolCalls: 3,
      durationSeconds: 3, // round(2500 / 1000)
    });
  });
});

describe("activity emission: bypasses sendUpdate's answer-delivery tracking", () => {
  // The consumer's `sendUpdate` chokepoint marks `session.emittedAssistantText
  // = true` for any untagged agent_message_chunk. Our zero-text chunks are
  // tagged (_meta.anyharness) but carry no assistant text and must never be
  // mistaken for the turn's delivered answer (issue #453's fallback depends
  // on this flag staying accurate).
  it("does not set emittedAssistantText for a roster-only chunk", async () => {
    const { agent } = createCapturingAgent();
    injectGeneratorSession(
      agent,
      makeGenerator([
        assistantMessage([toolUseBlock("toolu_bash1", "Bash", { command: "sleep 30 &" })]),
        taskStarted({ task_id: "proc-1", tool_use_id: "toolu_bash1" }),
        successResult(),
      ]),
    );

    await agent.prompt({ sessionId: "test-session", prompt: [{ type: "text", text: "go" }] });

    expect(agent.sessions["test-session"]?.emittedAssistantText).toBe(false);
  });
});

describe("activity emission: `_anyharness/activity/list` (attach-time reconcile pull)", () => {
  it("returns the current in-memory roster for a live session", async () => {
    const { agent } = createCapturingAgent();
    agent.sessions["test-session"] = mockSessionState();
    upsertProcess(agent.sessions["test-session"]!.activity, "proc-1", {
      command: "sleep 30 &",
      status: "running",
    });
    upsertSubagent(agent.sessions["test-session"]!.activity, "agent-1", {
      agentType: "Explore",
      status: "running",
    });

    const result = await agent.activityList({ sessionId: "test-session" });

    expect(result.processes).toHaveLength(1);
    expect(result.processes[0]).toMatchObject({ id: "proc-1", command: "sleep 30 &" });
    expect(result.subagents).toHaveLength(1);
    expect(result.subagents[0]).toMatchObject({ id: "agent-1", agentType: "Explore" });
  });

  it("returns an empty roster for a session that no longer exists (fresh process / reset path)", async () => {
    const { agent } = createCapturingAgent();
    const result = await agent.activityList({ sessionId: "unknown-session" });
    expect(result).toEqual({ processes: [], subagents: [] });
  });
});

describe("activity emission: resilience to a rejecting client.sessionUpdate", () => {
  // Every activity handler (handleActivityTaskStarted/Progress/Notification/
  // TaskUpdatedTerminal/ToolResults) wraps its body in try/catch and logs via
  // this.logger.error rather than letting the rejection propagate — four of
  // those are fired `void`-ed from runConsumer's switch (an unhandled
  // rejection if left unguarded) and the fifth is `await`-ed directly from
  // the main content loop (a drain-killing throw if left unguarded). This
  // test simulates the reviewer's flagged scenario most likely to actually
  // hit it: the post-turn task_notification for a backgrounded Bash task,
  // emitted while the peer is winding the connection down.
  it("survives a rejecting sessionUpdate on the post-turn task_notification without an unhandled rejection", async () => {
    const unhandled: unknown[] = [];
    const onUnhandledRejection = (reason: unknown) => unhandled.push(reason);
    process.on("unhandledRejection", onUnhandledRejection);

    try {
      const { agent, errors } = createRejectingCapturingAgent();
      injectGeneratorSession(agent, (input) => {
        async function* messageGenerator() {
          const iter = input[Symbol.asyncIterator]();
          const { value: userMessage } = await iter.next();
          yield userEcho(userMessage);
          yield assistantMessage([toolUseBlock("toolu_bash1", "Bash", { command: "sleep 30 &" })]);
          yield taskStarted({ task_id: "proc-1", tool_use_id: "toolu_bash1" });
          yield successResult();
          yield { type: "system", subtype: "session_state_changed", state: "idle" };
          // Strictly post-turn, exactly like the earlier "post-turn
          // completion" test — except here `sessionUpdate` throws for this
          // roster chunk (see createRejectingCapturingAgent).
          yield taskNotification({ task_id: "proc-1", output_file: "/tmp/proc-1.log" });
        }
        return messageGenerator();
      });

      const response = await agent.prompt({
        sessionId: "test-session",
        prompt: [{ type: "text", text: "go" }],
      });
      expect(response.stopReason).toBeDefined();

      // Let the post-turn task_notification (and its rejecting sessionUpdate
      // call) drain fully.
      await agent.sessions["test-session"]?.consumer;
      // Give any would-be unhandled rejection a microtask/timer turn to
      // surface before asserting its absence.
      await new Promise((r) => setTimeout(r, 0));

      expect(unhandled).toEqual([]);
      expect(errors.some((e) => String(e).includes("task_notification"))).toBe(true);
    } finally {
      process.off("unhandledRejection", onUnhandledRejection);
    }
  });

  it("survives a rejecting sessionUpdate on a subagent launch (task_started) without an unhandled rejection", async () => {
    const unhandled: unknown[] = [];
    const onUnhandledRejection = (reason: unknown) => unhandled.push(reason);
    process.on("unhandledRejection", onUnhandledRejection);

    try {
      const { agent, errors } = createRejectingCapturingAgent();
      injectGeneratorSession(
        agent,
        makeGenerator([
          taskStarted({
            task_id: "agent-1",
            tool_use_id: "toolu_agent1",
            subagent_type: "Explore",
          }),
          successResult(),
        ]),
      );

      const response = await agent.prompt({
        sessionId: "test-session",
        prompt: [{ type: "text", text: "go" }],
      });
      expect(response.stopReason).toBeDefined();
      await agent.sessions["test-session"]?.consumer;
      await new Promise((r) => setTimeout(r, 0));

      expect(unhandled).toEqual([]);
      expect(errors.some((e) => String(e).includes("task_started"))).toBe(true);
    } finally {
      process.off("unhandledRejection", onUnhandledRejection);
    }
  });
});
