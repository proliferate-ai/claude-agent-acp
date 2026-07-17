import { afterEach, describe, expect, it, vi } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { RequestError, SessionNotification } from "@agentclientprotocol/sdk";
import { ClaudeAcpAgent, type AcpClient } from "../acp-agent.js";
import {
  ANYHARNESS_CAPABILITIES,
  classifyGoalStatus,
  extractGoalStatus,
  newAnyharnessSessionState,
  readLastGoalStatus,
  TranscriptTailer,
} from "../anyharness.js";
import { Pushable } from "../utils.js";

const silentLogger = { log: () => {}, error: () => {} };
const createdAgents: ClaudeAcpAgent[] = [];

afterEach(() => {
  for (const agent of createdAgents.splice(0)) {
    for (const session of Object.values(agent.sessions)) {
      session.anyharness.tailer?.dispose();
    }
  }
});

function resultMsg(inputTokens: number, outputTokens: number): any {
  return {
    type: "result",
    subtype: "success",
    stop_reason: null,
    is_error: false,
    result: "done",
    total_cost_usd: 0,
    modelUsage: {},
    usage: {
      input_tokens: inputTokens,
      output_tokens: outputTokens,
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
    },
  };
}

function goalStatusLine(attachment: Record<string, unknown>): string {
  return (
    JSON.stringify({
      type: "attachment",
      uuid: "00000000-0000-0000-0000-000000000000",
      attachment: { type: "goal_status", ...attachment },
    }) + "\n"
  );
}

function createAgent() {
  const updates: SessionNotification[] = [];
  const client = {
    sessionUpdate: async (notification: SessionNotification) => {
      updates.push(notification);
    },
  } as unknown as AcpClient;
  const agent = new ClaudeAcpAgent(client, silentLogger);
  createdAgents.push(agent);
  return { agent, updates };
}

function injectSession(
  agent: ClaudeAcpAgent,
  sessionId: string,
  options: { transcriptPath?: string; query?: any; input?: Pushable<any> } = {},
) {
  const input = options.input ?? new Pushable<any>();
  const query =
    options.query ??
    ({
      next: vi.fn(() => new Promise<IteratorResult<any>>(() => {})),
      interrupt: vi.fn(async () => undefined),
      close: vi.fn(),
    } as any);
  const anyharness = newAnyharnessSessionState(true);
  anyharness.transcriptPath = options.transcriptPath ?? null;
  agent.sessions[sessionId] = {
    query,
    input,
    cancelled: false,
    cwd: "/test",
    sessionFingerprint: JSON.stringify({ cwd: "/test", mcpServers: [] }),
    modes: { currentModeId: "default", availableModes: [] },
    models: { currentModelId: "default", availableModels: [] },
    modelInfos: [],
    settingsManager: { dispose: vi.fn() } as any,
    accumulatedUsage: {
      inputTokens: 0,
      outputTokens: 0,
      cachedReadTokens: 0,
      cachedWriteTokens: 0,
    },
    configOptions: [],
    agents: [],
    currentAgent: "default",
    fastModeEnabled: false,
    abortController: new AbortController(),
    emitRawSDKMessages: false,
    contextWindowSize: 200000,
    taskState: new Map(),
    toolUseCache: {},
    emittedToolCalls: new Set(),
    liveBackgroundTasks: new Map(),
    emittedAssistantText: false,
    owedTrailingIdles: 0,
    messageIdToUuid: new Map(),
    anyharness,
  } as any;
  return agent.sessions[sessionId]!;
}

function tempTranscript(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "anyharness-ext-"));
  return path.join(dir, "session.jsonl");
}

describe("AnyHarness transcript primitives", () => {
  it("classifies native goal sentinels and evaluations", () => {
    expect(classifyGoalStatus({ type: "goal_status", met: false, sentinel: true })).toBe("armed");
    expect(classifyGoalStatus({ type: "goal_status", met: true, sentinel: true })).toBe("cleared");
    expect(classifyGoalStatus({ type: "goal_status", met: true, reason: "done" })).toBe("met");
    expect(classifyGoalStatus({ type: "goal_status", failed: true })).toBe("failed");
    expect(classifyGoalStatus({ type: "goal_status", met: false })).toBe("progress");
  });

  it("extracts known goal_status transcript shapes", () => {
    expect(
      extractGoalStatus({
        type: "attachment",
        attachment: { type: "goal_status", sentinel: true, condition: "c" },
      }),
    ).toMatchObject({ type: "goal_status", sentinel: true, condition: "c" });
    expect(
      extractGoalStatus({ message: { attachment: { type: "goal_status", met: true } } })?.met,
    ).toBe(true);
    expect(extractGoalStatus({ type: "user" })).toBeNull();
  });

  it("reads the last persisted goal row", () => {
    const file = tempTranscript();
    fs.writeFileSync(
      file,
      goalStatusLine({ met: false, sentinel: true, condition: "first" }) +
        goalStatusLine({ met: false, condition: "first", reason: "not yet" }),
    );
    expect(readLastGoalStatus(file)?.reason).toBe("not yet");
  });

  it("tails complete appended JSONL rows and buffers partial writes", async () => {
    const file = tempTranscript();
    const rows: unknown[] = [];
    const tailer = new TranscriptTailer(file, (row) => rows.push(row), silentLogger, {
      fromStart: true,
    });
    tailer.start();
    try {
      fs.writeFileSync(file, JSON.stringify({ a: 1 }) + "\n");
      await vi.waitFor(() => expect(rows).toHaveLength(1), { timeout: 3000 });
      const partial = JSON.stringify({ b: 2 });
      fs.appendFileSync(file, partial.slice(0, 5));
      await new Promise((resolve) => setTimeout(resolve, 100));
      expect(rows).toHaveLength(1);
      fs.appendFileSync(file, partial.slice(5) + "\n");
      await vi.waitFor(() => expect(rows).toHaveLength(2), { timeout: 3000 });
    } finally {
      tailer.dispose();
    }
  });
});

describe("AnyHarness GoalPort/LoopPort", () => {
  it("advertises wire-contract capabilities", async () => {
    const { agent } = createAgent();
    const response = await agent.initialize({ protocolVersion: 1, clientCapabilities: {} });
    expect(response._meta?.anyharness).toEqual(ANYHARNESS_CAPABILITIES);
  });

  it("dispatches both extension spellings and validates requests", async () => {
    const { agent } = createAgent();
    injectSession(agent, "s1");
    await expect(agent.extMethod("_anyharness/goal/get", { sessionId: "s1" })).resolves.toEqual({
      goal: null,
    });
    await expect(agent.extMethod("anyharness/loop/list", { sessionId: "s1" })).resolves.toEqual({
      loops: [],
    });
    await expect(agent.extMethod("_other/method", { sessionId: "s1" })).rejects.toThrow(
      RequestError,
    );
    await expect(agent.extMethod("_anyharness/goal/get", {})).rejects.toThrow(RequestError);
  });

  it("seeds a resumed native goal before goal/get answers", async () => {
    const { agent } = createAgent();
    const transcriptPath = tempTranscript();
    fs.writeFileSync(
      transcriptPath,
      goalStatusLine({ met: false, sentinel: true, condition: "DONE.txt exists" }),
    );
    const session = injectSession(agent, "s1", { transcriptPath });
    session.anyharness.tailFromStart = false;
    const result = await agent.extMethod("_anyharness/goal/get", { sessionId: "s1" });
    expect(result.goal).toMatchObject({
      objective: "DONE.txt exists",
      status: "active",
      nativeStatus: "armed",
    });
  });

  it("confirms goal/set only after the native sentinel round-trips", async () => {
    const { agent, updates } = createAgent();
    const transcriptPath = tempTranscript();
    const session = injectSession(agent, "s1", { transcriptPath });
    const push = vi.spyOn(session.input, "push");

    const pending = agent.extMethod("_anyharness/goal/set", {
      sessionId: "s1",
      objective: "DONE.txt exists",
    });
    await vi.waitFor(() => expect(push).toHaveBeenCalledOnce());
    expect(push.mock.calls[0][0].message.content).toEqual([
      { type: "text", text: "/goal DONE.txt exists" },
    ]);
    expect(session.anyharness.goal).toBeNull();

    fs.writeFileSync(
      transcriptPath,
      goalStatusLine({ met: false, sentinel: true, condition: "DONE.txt exists" }),
    );
    await expect(pending).resolves.toMatchObject({
      goal: { objective: "DONE.txt exists", status: "active", native: true, tokenBudget: null },
    });
    expect(
      updates.some(
        (update) => (update.update._meta as any)?.anyharness?.transcriptEvent === "goal_updated",
      ),
    ).toBe(true);
  });

  it("mirrors native goal completion and emits goal_met", async () => {
    const { agent, updates } = createAgent();
    const transcriptPath = tempTranscript();
    const session = injectSession(agent, "s1", { transcriptPath });
    const pending = agent.extMethod("_anyharness/goal/set", {
      sessionId: "s1",
      objective: "DONE.txt exists",
    });
    fs.writeFileSync(
      transcriptPath,
      goalStatusLine({ met: false, sentinel: true, condition: "DONE.txt exists" }),
    );
    await pending;
    fs.appendFileSync(
      transcriptPath,
      goalStatusLine({ met: true, condition: "DONE.txt exists", reason: "file is present" }),
    );
    await vi.waitFor(() => expect(session.anyharness.goal?.status).toBe("met"), { timeout: 3000 });
    expect(session.anyharness.goal?.metReason).toBe("file is present");
    expect(
      updates.some(
        (update) => (update.update._meta as any)?.anyharness?.transcriptEvent === "goal_met",
      ),
    ).toBe(true);
  });

  it("returns the native CronCreate id from loop/set", async () => {
    const { agent, updates } = createAgent();
    injectSession(agent, "s1");
    const pending = agent.extMethod("_anyharness/loop/set", {
      sessionId: "s1",
      prompt: "check deploy",
      schedule: { kind: "interval", expr: "5m" },
    });
    await vi.waitFor(() => expect(agent.sessions.s1.anyharness.pendingLoopSets).toHaveLength(1));
    await (agent as any).handleCronTool(
      "s1",
      "CronCreate",
      { prompt: "check deploy", cron: "*/5 * * * *", recurring: true },
      { id: "cron-native-1" },
    );
    await expect(pending).resolves.toMatchObject({
      loop: { loopId: "cron-native-1", prompt: "check deploy", native: true },
    });
    expect(
      updates.some(
        (update) => (update.update._meta as any)?.anyharness?.transcriptEvent === "loop_updated",
      ),
    ).toBe(true);
  });

  it("keeps a native loop wake separate from a queued ACP prompt", async () => {
    const { agent, updates } = createAgent();
    const input = new Pushable<any>();
    async function* messages() {
      yield {
        type: "user",
        message: { role: "user", content: [{ type: "text", text: "check deploy" }] },
        parent_tool_use_id: null,
        uuid: "native-loop-wake",
        session_id: "s1",
      };
      yield resultMsg(3, 4);
      const queued = await input[Symbol.asyncIterator]().next();
      if (!queued.done) {
        yield {
          type: "user",
          message: queued.value.message,
          parent_tool_use_id: null,
          uuid: queued.value.uuid,
          session_id: "s1",
          isReplay: true,
        };
      }
      yield resultMsg(11, 13);
    }
    const generator = messages();
    const query = Object.assign(generator, {
      interrupt: vi.fn(async () => undefined),
      close: vi.fn(),
      applyFlagSettings: vi.fn(),
    });
    const session = injectSession(agent, "s1", { query, input });
    session.anyharness.loops.set("loop-1", {
      loopId: "loop-1",
      prompt: "check deploy",
      schedule: { kind: "interval", expr: "5m" },
      recurring: true,
      status: "active",
      lastFiredAtMs: null,
      fireCount: 0,
      updatedAtMs: Date.now(),
    });

    const response = await agent.prompt({
      sessionId: "s1",
      prompt: [{ type: "text", text: "what happened?" }],
    });
    expect(response.usage).toMatchObject({ inputTokens: 11, outputTokens: 13 });
    expect(session.anyharness.loops.get("loop-1")?.fireCount).toBe(1);
    expect(
      updates.filter(
        (update) => (update.update._meta as any)?.anyharness?.transcriptEvent === "loop_fired",
      ),
    ).toHaveLength(1);
  });
});
