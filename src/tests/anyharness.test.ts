import { afterEach, describe, expect, it, vi } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { RequestError, SessionNotification } from "@agentclientprotocol/sdk";
import { ClaudeAcpAgent, type AcpClient } from "../acp-agent.js";
import {
  ANYHARNESS_CAPABILITIES,
  classifyGoalStatus,
  extractCronFirePrompt,
  extractGoalStatus,
  matchLoopForWake,
  newAnyharnessSessionState,
  parseBackgroundOutputFile,
  parseCronIdFromResult,
  readLastGoalStatus,
  subagentFeedPath,
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

function cronFireRow(prompt: string, timestampMs = Date.now() + 1_000): Record<string, unknown> {
  return {
    type: "user",
    isMeta: true,
    promptSource: "sdk",
    timestamp: new Date(timestampMs).toISOString(),
    message: { role: "user", content: prompt },
  };
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

function anyharnessEvents(updates: SessionNotification[], transcriptEvent: string): any[] {
  return updates
    .map((notification) => (notification.update._meta as any)?.anyharness)
    .filter((event) => event?.transcriptEvent === transcriptEvent);
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

function mainCronHookOrigin(): { promptId: string } {
  // Claude's hook prompt_id is a prompt-grain UUID distinct from the
  // SDKUserMessage/command UUID tracked by loopInjectionUuid.
  return { promptId: "main-prompt-grain" };
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

  it("recognizes only authoritative native-cron transcript rows", () => {
    expect(extractCronFirePrompt(cronFireRow("check deploy", 1_000))).toBe("check deploy");
    expect(
      extractCronFirePrompt({
        type: "user",
        isMeta: true,
        timestamp: new Date(1_000).toISOString(),
        message: { role: "user", content: "check deploy" },
      }),
    ).toBeNull();
    expect(
      extractCronFirePrompt({
        type: "user",
        isMeta: true,
        promptSource: "sdk",
        timestamp: new Date(1_000).toISOString(),
        message: { role: "user", content: [{ type: "text", text: "check deploy" }] },
      }),
    ).toBeNull();
  });

  it("matches loop wakes exactly and refuses wrappers or duplicate prompts", () => {
    const makeLoop = (loopId: string, prompt: string) => ({
      loopId,
      prompt,
      schedule: { kind: "interval" as const, expr: "5m" },
      recurring: true,
      status: "active" as const,
      lastFiredAtMs: null,
      fireCount: 0,
      createdAtMs: 0,
      updatedAtMs: 1,
    });
    const loops = [makeLoop("short", "deploy"), makeLoop("exact", "check deploy")];
    expect(matchLoopForWake(loops, "check deploy")?.loopId).toBe("exact");
    expect(matchLoopForWake(loops, "please check deploy now")).toBeUndefined();
    expect(matchLoopForWake([makeLoop("only", "deploy")], "please deploy now")).toBeUndefined();
    expect(
      matchLoopForWake([makeLoop("one", "same"), makeLoop("two", "same")], "same"),
    ).toBeUndefined();
  });

  it("parses current CronCreate, background-output, and child-feed wire values", () => {
    expect(
      parseCronIdFromResult({
        content: [{ type: "text", text: "Scheduled recurring job dad38e14 (*/5 * * * *)" }],
      }),
    ).toBe("dad38e14");
    expect(
      parseBackgroundOutputFile(
        "Command is running in the background. Output is being written to: /tmp/bash-1.log",
      ),
    ).toBe("/tmp/bash-1.log");
    expect(subagentFeedPath("/tmp/project/session.jsonl", "session-1", "child-7")).toBe(
      "/tmp/project/session-1/subagents/agent-child-7.jsonl",
    );
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
      {
        content: [{ type: "text", text: "Scheduled recurring job dad38e14 (*/5 * * * *)" }],
      },
      mainCronHookOrigin(),
    );
    await expect(pending).resolves.toMatchObject({
      loop: { loopId: "dad38e14", prompt: "check deploy", native: true },
    });
    expect(
      updates.some(
        (update) => (update.update._meta as any)?.anyharness?.transcriptEvent === "loop_upserted",
      ),
    ).toBe(true);
  });

  it("does not count preexisting transcript rows after fast loop reconciliation", async () => {
    vi.useFakeTimers();
    const now = Date.now();
    const transcriptPath = tempTranscript();
    fs.writeFileSync(
      transcriptPath,
      `${JSON.stringify(cronFireRow("check deploy", now - 10_000))}\n`,
    );
    const { agent, updates } = createAgent();
    const session = injectSession(agent, "s1", { transcriptPath });
    try {
      const pending = agent.extMethod("_anyharness/loop/set", {
        sessionId: "s1",
        prompt: "check deploy",
        schedule: { kind: "interval", expr: "5m" },
      });
      expect(session.anyharness.pendingLoopSets).toHaveLength(1);
      await (agent as any).handleCronTool(
        "s1",
        "CronCreate",
        { prompt: "check deploy", cron: "*/5 * * * *", recurring: true },
        "Scheduled recurring job fresh123 (*/5 * * * *)",
        mainCronHookOrigin(),
      );
      await expect(pending).resolves.toMatchObject({ loop: { loopId: "fresh123" } });

      // Force the tailer's initial read only after the real id is matchable.
      // The historical row remains below the loop's creation-time floor.
      await vi.advanceTimersByTimeAsync(60);
      expect(session.anyharness.loops.get("fresh123")?.fireCount).toBe(0);
      expect(anyharnessEvents(updates, "loop_fired")).toHaveLength(0);

      fs.appendFileSync(
        transcriptPath,
        `${JSON.stringify(cronFireRow("check deploy", now + 1_000))}\n`,
      );
      await vi.advanceTimersByTimeAsync(800);
      expect(session.anyharness.loops.get("fresh123")?.fireCount).toBe(1);
      expect(anyharnessEvents(updates, "loop_fired")).toHaveLength(1);
    } finally {
      session.anyharness.tailer?.dispose();
      session.anyharness.tailer = null;
      vi.useRealTimers();
    }
  });

  it("does not attribute an ordinary SDK user replay as a loop fire", async () => {
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
      createdAtMs: 0,
      updatedAtMs: Date.now(),
    });

    const response = await agent.prompt({
      sessionId: "s1",
      prompt: [{ type: "text", text: "what happened?" }],
    });
    expect(response.usage).toMatchObject({ inputTokens: 11, outputTokens: 13 });
    expect(session.anyharness.loops.get("loop-1")?.fireCount).toBe(0);
    expect(anyharnessEvents(updates, "loop_fired")).toHaveLength(0);
  });

  it("attributes transcript cron fires to the unique matching loop", async () => {
    const { agent, updates } = createAgent();
    const session = injectSession(agent, "s1");
    session.anyharness.loops.set("short", {
      loopId: "short",
      prompt: "deploy",
      schedule: { kind: "interval", expr: "5m" },
      recurring: true,
      status: "active",
      lastFiredAtMs: null,
      fireCount: 0,
      createdAtMs: 0,
      updatedAtMs: 1,
    });
    session.anyharness.loops.set("exact", {
      loopId: "exact",
      prompt: "check deploy",
      schedule: { kind: "interval", expr: "5m" },
      recurring: true,
      status: "active",
      lastFiredAtMs: null,
      fireCount: 0,
      createdAtMs: 0,
      updatedAtMs: 1,
    });

    (agent as any).handleTranscriptRow("s1", cronFireRow("check deploy"));
    await vi.waitFor(() => expect(anyharnessEvents(updates, "loop_fired")).toHaveLength(1));

    expect(session.anyharness.loops.get("short")?.fireCount).toBe(0);
    expect(session.anyharness.loops.get("exact")?.fireCount).toBe(1);
    expect(anyharnessEvents(updates, "loop_fired")[0]).toMatchObject({
      loopId: "exact",
      loop: { loopId: "exact", fireCount: 1 },
    });

    session.anyharness.loops.set("one-shot", {
      loopId: "one-shot",
      prompt: "report once",
      schedule: { kind: "cron", expr: "0 9 * * *" },
      recurring: false,
      status: "active",
      lastFiredAtMs: null,
      fireCount: 0,
      createdAtMs: 0,
      updatedAtMs: 1,
    });
    (agent as any).handleTranscriptRow("s1", cronFireRow("report once"));
    await vi.waitFor(() => expect(anyharnessEvents(updates, "loop_removed")).toHaveLength(1));
    expect(session.anyharness.loops.get("one-shot")).toMatchObject({
      status: "cleared",
      fireCount: 1,
    });
    await expect(agent.extMethod("_anyharness/loop/list", { sessionId: "s1" })).resolves.toEqual({
      loops: expect.not.arrayContaining([expect.objectContaining({ loopId: "one-shot" })]),
    });
  });

  it("defers loop/set until idle and reconciles its provisional id", async () => {
    const { agent, updates } = createAgent();
    const transcriptPath = tempTranscript();
    const session = injectSession(agent, "s1", { transcriptPath });
    const push = vi.spyOn(session.input, "push");
    const activeTurn = { settled: false } as any;
    session.activeTurn = activeTurn;
    session.turnQueue = [activeTurn];
    session.consumer = Promise.resolve();
    session.lastSessionState = "running";

    const result = await agent.extMethod("_anyharness/loop/set", {
      sessionId: "s1",
      prompt: "check deploy",
      schedule: { kind: "interval", expr: "5m" },
    });
    const provisionalId = (result.loop as any).loopId as string;
    expect(provisionalId).toMatch(/^provisional-/);
    expect(push).not.toHaveBeenCalled();
    expect(session.anyharness.deferredLoopInjections).toHaveLength(1);
    expect(anyharnessEvents(updates, "loop_upserted")[0]).toMatchObject({
      loopId: provisionalId,
      loop: { loopId: provisionalId },
    });

    (agent as any).handleTranscriptRow("s1", cronFireRow("check deploy"));
    expect(session.anyharness.loops.get(provisionalId)?.fireCount).toBe(0);
    expect(anyharnessEvents(updates, "loop_fired")).toHaveLength(0);
    session.activeTurn = null;
    session.turnQueue = [];
    session.lastSessionState = "idle";
    (agent as any).tryFlushDeferredLoopInjection("s1", session);
    (agent as any).tryFlushDeferredLoopInjection("s1", session);
    expect(push).toHaveBeenCalledOnce();
    expect(push.mock.calls[0][0].message.content).toEqual([
      { type: "text", text: "/loop 5m check deploy" },
    ]);

    await (agent as any).handleCronTool(
      "s1",
      "CronCreate",
      { prompt: "check deploy", cron: "*/5 * * * *", recurring: true },
      "Scheduled recurring job a1b2c3d4 (*/5 * * * *)",
      mainCronHookOrigin(),
    );
    expect(session.anyharness.loops.has(provisionalId)).toBe(false);
    expect(session.anyharness.loopAliases.get(provisionalId)).toBe("a1b2c3d4");
    expect(session.anyharness.loops.get("a1b2c3d4")).toMatchObject({
      loopId: "a1b2c3d4",
      prompt: "check deploy",
      fireCount: 0,
    });
    expect(anyharnessEvents(updates, "loop_removed")).toContainEqual(
      expect.objectContaining({ loopId: provisionalId }),
    );

    (agent as any).handleTranscriptRow("s1", cronFireRow("check deploy"));
    await vi.waitFor(() => expect(session.anyharness.loops.get("a1b2c3d4")?.fireCount).toBe(1));

    await expect(
      agent.extMethod("_anyharness/loop/clear", { sessionId: "s1", loopId: provisionalId }),
    ).resolves.toEqual({ cleared: 1 });
    expect(session.anyharness.deferredLoopInjections.at(-1)?.text).toContain('"a1b2c3d4"');
    expect(session.anyharness.loops.get("a1b2c3d4")?.status).toBe("cleared");
  });

  it("correlates identical deferred sets only after each exact injection runs", async () => {
    const { agent } = createAgent();
    const session = injectSession(agent, "s1", { transcriptPath: tempTranscript() });
    const activeTurn = { settled: false, deferredSettle: { stopReason: "end_turn" } } as any;
    session.activeTurn = activeTurn;
    session.turnQueue = [activeTurn];
    session.consumer = Promise.resolve();
    session.lastSessionState = "running";

    const first = await agent.extMethod("_anyharness/loop/set", {
      sessionId: "s1",
      prompt: "same prompt",
      schedule: { kind: "interval", expr: "5m" },
    });
    const second = await agent.extMethod("_anyharness/loop/set", {
      sessionId: "s1",
      prompt: "same prompt",
      schedule: { kind: "interval", expr: "10m" },
    });
    const firstId = (first.loop as any).loopId as string;
    const secondId = (second.loop as any).loopId as string;

    // A same-prompt CronCreate from the still-active ACP turn is native state,
    // but it must not consume either deferred LoopPort request.
    await (agent as any).handleCronTool(
      "s1",
      "CronCreate",
      { prompt: "same prompt", cron: "*/5 * * * *", recurring: true },
      "Scheduled recurring job external0 (*/5 * * * *)",
      { promptId: "foreground-user" },
    );
    expect(session.anyharness.loopAliases.get(firstId)).toBeUndefined();
    expect(session.anyharness.loopAliases.get(secondId)).toBeUndefined();
    expect(session.anyharness.loops.has(firstId)).toBe(true);
    expect(session.anyharness.loops.has(secondId)).toBe(true);
    expect(session.anyharness.pendingLoopSets).toHaveLength(2);

    session.activeTurn = null;
    session.turnQueue = [];
    session.lastSessionState = "idle";
    (agent as any).tryFlushDeferredLoopInjection("s1", session);
    const firstUuid = session.anyharness.loopInjectionUuid;
    expect(firstUuid).toBe(session.anyharness.pendingLoopSets[0].injectionUuid);

    // A background child can execute CronCreate concurrently with the
    // main-thread injected turn. Its own hook origin must not steal that turn's
    // pending set even when the prompt text is identical.
    await (agent as any).handleCronTool(
      "s1",
      "CronCreate",
      { prompt: "same prompt", cron: "*/7 * * * *", recurring: true },
      "Scheduled recurring job child999 (*/7 * * * *)",
      { promptId: "child-prompt", agentId: "child-1" },
    );
    expect(session.anyharness.pendingLoopSets).toHaveLength(2);
    expect(session.anyharness.loopAliases.get(firstId)).toBeUndefined();

    await (agent as any).handleCronTool(
      "s1",
      "CronCreate",
      { prompt: "same prompt", cron: "*/5 * * * *", recurring: true },
      "Scheduled recurring job 1111aaaa (*/5 * * * *)",
      mainCronHookOrigin(),
    );
    expect(session.anyharness.loopAliases.get(firstId)).toBe("1111aaaa");
    expect(session.anyharness.loops.has(firstId)).toBe(false);
    expect(session.anyharness.pendingLoopSets).toHaveLength(1);
    expect(session.anyharness.pendingLoopSets[0].provisionalLoopId).toBe(secondId);

    (agent as any).finishLoopInjection("s1", session, firstUuid);
    (agent as any).tryFlushDeferredLoopInjection("s1", session);
    const secondUuid = session.anyharness.loopInjectionUuid;
    expect(secondUuid).toBe(session.anyharness.pendingLoopSets[0].injectionUuid);

    await (agent as any).handleCronTool(
      "s1",
      "CronCreate",
      { prompt: "same prompt", cron: "*/10 * * * *", recurring: true },
      "Scheduled recurring job 2222bbbb (*/10 * * * *)",
      mainCronHookOrigin(),
    );
    expect(session.anyharness.loopAliases.get(secondId)).toBe("2222bbbb");
    expect(session.anyharness.pendingLoopSets).toHaveLength(0);
    (agent as any).finishLoopInjection("s1", session, secondUuid);
    expect([...session.anyharness.loops.keys()].sort()).toEqual([
      "1111aaaa",
      "2222bbbb",
      "child999",
      "external0",
    ]);
  });

  it("keeps a cleared provisional loop retired when CronCreate arrives late", async () => {
    const { agent, updates } = createAgent();
    const session = injectSession(agent, "s1", { transcriptPath: tempTranscript() });
    const activeTurn = { settled: false, deferredSettle: { stopReason: "end_turn" } } as any;
    session.activeTurn = activeTurn;
    session.turnQueue = [activeTurn];
    session.consumer = Promise.resolve();
    session.lastSessionState = "running";

    const created = await agent.extMethod("_anyharness/loop/set", {
      sessionId: "s1",
      prompt: "clear during create",
      schedule: { kind: "interval", expr: "5m" },
    });
    const provisionalId = (created.loop as any).loopId as string;
    session.activeTurn = null;
    session.turnQueue = [];
    session.lastSessionState = "idle";
    (agent as any).tryFlushDeferredLoopInjection("s1", session);

    await expect(
      agent.extMethod("_anyharness/loop/clear", { sessionId: "s1", loopId: provisionalId }),
    ).resolves.toEqual({ cleared: 1 });
    expect(session.anyharness.loops.get(provisionalId)?.status).toBe("cleared");

    await (agent as any).handleCronTool(
      "s1",
      "CronCreate",
      { prompt: "clear during create", cron: "*/5 * * * *", recurring: true },
      "Scheduled recurring job late1234 (*/5 * * * *)",
      mainCronHookOrigin(),
    );

    expect(session.anyharness.loopAliases.get(provisionalId)).toBe("late1234");
    expect(session.anyharness.loops.get("late1234")?.status).toBe("cleared");
    expect(session.anyharness.deferredLoopInjections.at(-1)).toMatchObject({
      loopId: "late1234",
      text: expect.stringContaining('"late1234"'),
    });
    expect(
      anyharnessEvents(updates, "loop_upserted").some((event) => event.loopId === "late1234"),
    ).toBe(false);
    await expect(agent.extMethod("_anyharness/loop/list", { sessionId: "s1" })).resolves.toEqual({
      loops: [],
    });
  });

  it("clear-all cancels every unpushed provisional set", async () => {
    const { agent, updates } = createAgent();
    const session = injectSession(agent, "s1", { transcriptPath: tempTranscript() });
    const push = vi.spyOn(session.input, "push");
    const activeTurn = { settled: false, deferredSettle: { stopReason: "end_turn" } } as any;
    session.activeTurn = activeTurn;
    session.turnQueue = [activeTurn];
    session.consumer = Promise.resolve();
    session.lastSessionState = "running";

    const first = await agent.extMethod("_anyharness/loop/set", {
      sessionId: "s1",
      prompt: "first pending",
      schedule: { kind: "interval", expr: "5m" },
    });
    const second = await agent.extMethod("_anyharness/loop/set", {
      sessionId: "s1",
      prompt: "second pending",
      schedule: { kind: "interval", expr: "10m" },
    });
    const ids = [(first.loop as any).loopId, (second.loop as any).loopId];
    expect(session.anyharness.deferredLoopInjections).toHaveLength(2);

    await expect(agent.extMethod("_anyharness/loop/clear", { sessionId: "s1" })).resolves.toEqual({
      cleared: 2,
    });
    expect(session.anyharness.pendingLoopSets).toHaveLength(0);
    expect(session.anyharness.deferredLoopInjections).toHaveLength(0);
    expect(ids.map((id) => session.anyharness.loops.get(id)?.status)).toEqual([
      "cleared",
      "cleared",
    ]);
    expect(new Set(anyharnessEvents(updates, "loop_removed").map((event) => event.loopId))).toEqual(
      new Set(ids),
    );

    session.activeTurn = null;
    session.turnQueue = [];
    session.lastSessionState = "idle";
    (agent as any).tryFlushDeferredLoopInjection("s1", session);
    expect(push).not.toHaveBeenCalled();
    await expect(agent.extMethod("_anyharness/loop/list", { sessionId: "s1" })).resolves.toEqual({
      loops: [],
    });
  });

  it("retires a failed deferred set by its exact injection id before retry", async () => {
    const { agent, updates } = createAgent();
    const session = injectSession(agent, "s1", { transcriptPath: tempTranscript() });
    const activeTurn = { settled: false, deferredSettle: { stopReason: "end_turn" } } as any;
    session.activeTurn = activeTurn;
    session.turnQueue = [activeTurn];
    session.consumer = Promise.resolve();
    session.lastSessionState = "running";

    const failed = await agent.extMethod("_anyharness/loop/set", {
      sessionId: "s1",
      prompt: "retry me",
      schedule: { kind: "interval", expr: "invalid interval" },
    });
    const failedId = (failed.loop as any).loopId as string;
    session.activeTurn = null;
    session.turnQueue = [];
    session.lastSessionState = "idle";
    (agent as any).tryFlushDeferredLoopInjection("s1", session);
    const injectionUuid = session.anyharness.loopInjectionUuid;
    expect(injectionUuid).toBeTypeOf("string");

    (agent as any).finishLoopInjection("s1", session, "unrelated-goal-uuid");
    expect(session.anyharness.loopInjectionUuid).toBe(injectionUuid);
    expect(session.anyharness.pendingLoopSets).toHaveLength(1);

    (agent as any).finishLoopInjection("s1", session, injectionUuid);
    expect(session.anyharness.loopInjectionInFlight).toBe(false);
    expect(session.anyharness.pendingLoopSets).toHaveLength(0);
    expect(session.anyharness.loops.get(failedId)?.status).toBe("cleared");
    expect(anyharnessEvents(updates, "loop_removed")).toContainEqual(
      expect.objectContaining({ loopId: failedId }),
    );

    const retry = agent.extMethod("_anyharness/loop/set", {
      sessionId: "s1",
      prompt: "retry me",
      schedule: { kind: "interval", expr: "5m" },
    });
    await vi.waitFor(() => expect(session.anyharness.pendingLoopSets).toHaveLength(1));
    await (agent as any).handleCronTool(
      "s1",
      "CronCreate",
      { prompt: "retry me", cron: "*/5 * * * *", recurring: true },
      "Scheduled recurring job abcdef12 (*/5 * * * *)",
      mainCronHookOrigin(),
    );
    await expect(retry).resolves.toMatchObject({ loop: { loopId: "abcdef12" } });
    expect(session.anyharness.loops.get(failedId)?.status).toBe("cleared");
    expect(session.anyharness.loops.get("abcdef12")?.status).toBe("active");
  });

  it("does not apply a concrete unknown CronDelete id to the sole tracked loop", async () => {
    const { agent, updates } = createAgent();
    const session = injectSession(agent, "s1");
    session.anyharness.loops.set("tracked-loop", {
      loopId: "tracked-loop",
      prompt: "tracked prompt",
      schedule: { kind: "interval", expr: "5m" },
      recurring: true,
      status: "active",
      lastFiredAtMs: null,
      fireCount: 0,
      createdAtMs: 0,
      updatedAtMs: 1,
    });

    await (agent as any).handleCronTool(
      "s1",
      "CronDelete",
      { id: "untracked-loop" },
      "Deleted cron job untracked-loop",
      { promptId: "child-prompt", agentId: "child-1" },
    );

    expect(session.anyharness.loops.get("tracked-loop")?.status).toBe("active");
    expect(anyharnessEvents(updates, "loop_removed")).toHaveLength(0);
  });

  it("emits and lists canonical process activity while retaining terminal rows", async () => {
    const { agent, updates } = createAgent();
    const session = injectSession(agent, "s1", { transcriptPath: tempTranscript() });
    (agent as any).captureActivityTaskIo("s1", {
      type: "assistant",
      message: {
        content: [
          { type: "tool_use", id: "bash-tool-1", name: "Bash", input: { command: "sleep 1" } },
        ],
      },
    });
    await (agent as any).handleActivityTaskEvent("s1", {
      type: "system",
      subtype: "task_started",
      task_id: "process-1",
      tool_use_id: "bash-tool-1",
      description: "sleep 1",
    });
    const started = anyharnessEvents(updates, "process_upserted")[0].process;
    expect(started).toMatchObject({
      id: "process-1",
      command: "sleep 1",
      cwd: "/test",
      status: "running",
      exitCode: null,
      pid: null,
      endedAtMs: null,
      feed: null,
    });
    expect(Number.isInteger(started.startedAtMs)).toBe(true);
    expect(started).not.toHaveProperty("toolUseId");
    expect(started).not.toHaveProperty("updatedAtMs");
    const startedNotification = updates.find(
      (notification) =>
        (notification.update._meta as any)?.anyharness?.transcriptEvent === "process_upserted",
    )!;
    expect((startedNotification.update as any).content).toEqual({ type: "text", text: "" });
    expect((startedNotification.update._meta as any).anyharness).toEqual({
      schemaVersion: 1,
      transcriptEvent: "process_upserted",
      process: started,
    });

    (agent as any).captureActivityTaskIo("s1", {
      type: "user",
      message: {
        content: [
          {
            type: "tool_result",
            tool_use_id: "bash-tool-1",
            content:
              "Command is running in the background. Output is being written to: /tmp/bash-1.log",
          },
        ],
      },
    });
    await vi.waitFor(() =>
      expect(session.anyharness.processes.get("process-1")?.feed).toEqual({
        kind: "tail_file",
        path: "/tmp/bash-1.log",
      }),
    );
    await (agent as any).handleActivityTaskEvent("s1", {
      type: "system",
      subtype: "task_notification",
      task_id: "process-1",
      tool_use_id: "bash-tool-1",
      status: "completed",
      output_file: "/tmp/bash-1.log",
      summary: "done",
    });

    const listed = await agent.extMethod("_anyharness/activity/list", { sessionId: "s1" });
    expect(Object.keys(listed).sort()).toEqual(["processes", "subagents"]);
    expect(listed.processes).toEqual([
      expect.objectContaining({
        id: "process-1",
        status: "exited",
        feed: { kind: "tail_file", path: "/tmp/bash-1.log" },
      }),
    ]);
    expect((listed.processes as any[])[0].endedAtMs).not.toBeNull();
    expect(anyharnessEvents(updates, "process_upserted").at(-1).process).toMatchObject({
      id: "process-1",
      status: "exited",
      feed: { kind: "tail_file", path: "/tmp/bash-1.log" },
    });
  });

  it("merges canonical subagent activity through terminal and resumed task_updated", async () => {
    const { agent, updates } = createAgent();
    const transcriptPath = tempTranscript();
    injectSession(agent, "s1", { transcriptPath });
    await (agent as any).handleActivityTaskEvent("s1", {
      type: "system",
      subtype: "task_started",
      task_id: "child-1",
      tool_use_id: "agent-tool-1",
      subagent_type: "Explore",
      description: "inspect the repository",
    });
    const startedEvent = anyharnessEvents(updates, "subagent_upserted")[0].subagent;
    expect(startedEvent).toMatchObject({
      id: "child-1",
      agentType: "Explore",
      description: "inspect the repository",
      background: false,
      status: "running",
      feed: {
        kind: "tail_file",
        path: subagentFeedPath(transcriptPath, "s1", "child-1"),
      },
    });

    await (agent as any).handleActivityTaskEvent("s1", {
      type: "system",
      subtype: "task_progress",
      task_id: "child-1",
      summary: "found the entrypoint",
      usage: { total_tokens: 1200, tool_uses: 4, duration_ms: 3207 },
    });
    await (agent as any).reconcileActivityBackgroundFlags("s1", [{ task_id: "child-1" }]);
    await (agent as any).handleActivityTaskEvent("s1", {
      type: "system",
      subtype: "task_updated",
      task_id: "child-1",
      patch: { status: "completed" },
    });

    const listed = await agent.extMethod("_anyharness/activity/list", { sessionId: "s1" });
    expect(listed.processes).toEqual([]);
    expect(listed.subagents).toEqual([
      {
        id: "child-1",
        agentType: "Explore",
        description: "inspect the repository",
        model: null,
        background: true,
        status: "completed",
        summary: "found the entrypoint",
        tokensUsed: 1200,
        toolCalls: 4,
        durationSeconds: 3.207,
        feed: {
          kind: "tail_file",
          path: subagentFeedPath(transcriptPath, "s1", "child-1"),
        },
      },
    ]);
    const finalEvent = anyharnessEvents(updates, "subagent_upserted").at(-1).subagent;
    expect(finalEvent.status).toBe("completed");
    expect(startedEvent).toMatchObject({ status: "running", background: false });
    expect(finalEvent).not.toHaveProperty("prompt");
    expect(finalEvent).not.toHaveProperty("updatedAtMs");
    expect(finalEvent).not.toHaveProperty("usage");

    await (agent as any).handleActivityTaskEvent("s1", {
      type: "system",
      subtype: "task_updated",
      task_id: "child-1",
      patch: {
        status: "running",
        description: "inspect the resumed repository",
        is_backgrounded: false,
      },
    });
    expect(anyharnessEvents(updates, "subagent_upserted").at(-1).subagent).toMatchObject({
      id: "child-1",
      status: "running",
      description: "inspect the resumed repository",
      background: false,
    });
  });

  it("keeps the child transcript feed after a terminal task_notification", async () => {
    const { agent, updates } = createAgent();
    const transcriptPath = tempTranscript();
    injectSession(agent, "s1", { transcriptPath });
    await (agent as any).handleActivityTaskEvent("s1", {
      type: "system",
      subtype: "task_started",
      task_id: "child-notification",
      tool_use_id: "agent-tool-notification",
      subagent_type: "Explore",
      description: "inspect the repository",
    });
    await (agent as any).handleActivityTaskEvent("s1", {
      type: "system",
      subtype: "task_notification",
      task_id: "child-notification",
      tool_use_id: "agent-tool-notification",
      status: "completed",
      output_file: "/tmp/task-result.txt",
      summary: "done",
    });

    expect(anyharnessEvents(updates, "subagent_upserted").at(-1).subagent).toMatchObject({
      id: "child-notification",
      status: "completed",
      feed: {
        kind: "tail_file",
        path: subagentFeedPath(transcriptPath, "s1", "child-notification"),
      },
    });
  });

  it("uses a background level that arrives before subagent start", () => {
    const { agent, updates } = createAgent();
    injectSession(agent, "s1", { transcriptPath: tempTranscript() });

    (agent as any).handleActivityTaskEvent(
      "s1",
      {
        type: "system",
        subtype: "task_started",
        task_id: "child-level-first",
        subagent_type: "Explore",
        description: "inspect in background",
      },
      new Set(["child-level-first"]),
    );

    expect(anyharnessEvents(updates, "subagent_upserted")[0].subagent).toMatchObject({
      id: "child-level-first",
      background: true,
      status: "running",
    });
  });
});
