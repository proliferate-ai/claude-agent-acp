import { describe, expect, it, vi } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { AgentSideConnection, RequestError, SessionNotification } from "@agentclientprotocol/sdk";
import { ClaudeAcpAgent } from "../acp-agent.js";
import {
  classifyGoalStatus,
  extractGoalStatus,
  newAnyharnessSessionState,
  readLastGoalStatus,
  TranscriptTailer,
} from "../anyharness.js";
import { Pushable } from "../utils.js";

const silentLogger = { log: () => {}, error: () => {} };

/** Minimal SDK "result" message carrying only the fields drainTurn reads. */
function resultMsg(inputTokens: number, outputTokens: number): any {
  return {
    type: "result",
    subtype: "success",
    stop_reason: null,
    is_error: false,
    result: "",
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

/** A query stub that yields a fixed list of SDK messages via next(). */
function queryFrom(messages: any[]): any {
  let i = 0;
  return {
    next: async () =>
      i < messages.length
        ? { done: false, value: messages[i++] }
        : { done: true, value: undefined },
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

describe("classifyGoalStatus", () => {
  it("classifies arm sentinels", () => {
    expect(classifyGoalStatus({ type: "goal_status", met: false, sentinel: true })).toBe("armed");
  });

  it("classifies clear sentinels", () => {
    expect(classifyGoalStatus({ type: "goal_status", met: true, sentinel: true })).toBe("cleared");
  });

  it("classifies met evaluations", () => {
    expect(classifyGoalStatus({ type: "goal_status", met: true, reason: "done" })).toBe("met");
  });

  it("classifies failed evaluations", () => {
    expect(classifyGoalStatus({ type: "goal_status", met: false, failed: true })).toBe("failed");
  });

  it("classifies not-met evaluations as progress", () => {
    expect(classifyGoalStatus({ type: "goal_status", met: false, reason: "not yet" })).toBe(
      "progress",
    );
  });
});

describe("extractGoalStatus", () => {
  it("extracts from a transcript attachment row", () => {
    const row = {
      type: "attachment",
      attachment: { type: "goal_status", met: false, sentinel: true, condition: "c" },
    };
    expect(extractGoalStatus(row)).toEqual({
      type: "goal_status",
      met: false,
      sentinel: true,
      condition: "c",
    });
  });

  it("extracts from a message-nested attachment", () => {
    const row = { message: { attachment: { type: "goal_status", met: true } } };
    expect(extractGoalStatus(row)?.met).toBe(true);
  });

  it("extracts a bare goal_status object", () => {
    expect(extractGoalStatus({ type: "goal_status", met: true })?.met).toBe(true);
  });

  it("returns null for non-goal rows", () => {
    expect(extractGoalStatus({ type: "attachment", attachment: { type: "todo" } })).toBeNull();
    expect(extractGoalStatus({ type: "user", message: { content: "hi" } })).toBeNull();
    expect(extractGoalStatus("goal_status")).toBeNull();
    expect(extractGoalStatus(null)).toBeNull();
  });
});

describe("readLastGoalStatus", () => {
  it("returns the last goal_status row in a transcript", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "anyharness-test-"));
    const file = path.join(dir, "session.jsonl");
    fs.writeFileSync(
      file,
      JSON.stringify({ type: "user", message: { content: "hello" } }) +
        "\n" +
        goalStatusLine({ met: false, sentinel: true, condition: "first" }) +
        goalStatusLine({ met: false, condition: "first", reason: "not yet" }),
    );
    const last = readLastGoalStatus(file);
    expect(last?.reason).toBe("not yet");
    expect(classifyGoalStatus(last!)).toBe("progress");
  });

  it("returns null for a missing file", () => {
    expect(readLastGoalStatus("/nonexistent/nope.jsonl")).toBeNull();
  });
});

describe("TranscriptTailer", () => {
  it("parses appended complete lines, buffering partial writes", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "anyharness-tail-"));
    const file = path.join(dir, "t.jsonl");
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
      await new Promise((r) => setTimeout(r, 200));
      expect(rows).toHaveLength(1);

      fs.appendFileSync(file, partial.slice(5) + "\n");
      await vi.waitFor(() => expect(rows).toHaveLength(2), { timeout: 3000 });
      expect(rows[1]).toEqual({ b: 2 });
    } finally {
      tailer.dispose();
    }
  });
});

describe("extMethod dispatch", () => {
  function createAgent() {
    const updates: SessionNotification[] = [];
    const client = {
      sessionUpdate: async (notification: SessionNotification) => {
        updates.push(notification);
      },
    } as unknown as AgentSideConnection;
    const agent = new ClaudeAcpAgent(client, silentLogger);
    return { agent, updates };
  }

  function injectSession(agent: ClaudeAcpAgent, sessionId: string, transcriptPath?: string) {
    const input = new Pushable<any>();
    async function* neverYields() {
      yield await new Promise<never>(() => {});
    }
    const anyharness = newAnyharnessSessionState(true);
    anyharness.transcriptPath = transcriptPath ?? null;
    agent.sessions[sessionId] = {
      query: neverYields() as any,
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
      promptRunning: false,
      pendingMessages: new Map(),
      nextPendingOrder: 0,
      abortController: new AbortController(),
      emitRawSDKMessages: false,
      contextWindowSize: 200000,
      taskState: new Map(),
      toolUseCache: {},
      messageIdToUuid: new Map(),
      anyharness,
    };
    return agent.sessions[sessionId]!;
  }

  function tempTranscript(): string {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "anyharness-ext-"));
    return path.join(dir, "session.jsonl");
  }

  it("answers goal/get from the mirror under both wire spellings", async () => {
    const { agent } = createAgent();
    injectSession(agent, "s1");
    expect(await agent.extMethod("_anyharness/goal/get", { sessionId: "s1" })).toEqual({
      goal: null,
    });
    expect(await agent.extMethod("anyharness/goal/get", { sessionId: "s1" })).toEqual({
      goal: null,
    });
  });

  it("goal/get seeds the mirror from the transcript on a resumed session", async () => {
    const { agent } = createAgent();
    const transcriptPath = tempTranscript();
    // A native goal that survived --resume: the transcript already holds the
    // arm sentinel, but the fresh Session starts with an empty mirror.
    fs.writeFileSync(
      transcriptPath,
      goalStatusLine({ met: false, sentinel: true, condition: "DONE.txt exists" }),
    );
    const session = injectSession(agent, "s1", transcriptPath);
    // Resumed/forked sessions tail from EOF, so goal/get must trigger the
    // synchronous seed itself (an attach reconcile pulls before the CLI's
    // SessionStart hook seeds via the async pump).
    session.anyharness.tailFromStart = false;
    expect(session.anyharness.goal).toBeNull();

    const result = (await agent.extMethod("_anyharness/goal/get", { sessionId: "s1" })) as {
      goal: { objective: string; status: string; nativeStatus: string } | null;
    };
    expect(result.goal).toMatchObject({
      objective: "DONE.txt exists",
      status: "active",
      nativeStatus: "armed",
    });
    // The seeded mirror is non-terminal, so an attach reconcile sees the goal
    // instead of clearing it.
    expect(session.anyharness.goal?.status).toBe("active");
  });

  it("rejects unknown methods, missing sessionId, and unknown sessions", async () => {
    const { agent } = createAgent();
    injectSession(agent, "s1");
    await expect(agent.extMethod("_other/thing", { sessionId: "s1" })).rejects.toThrow(
      RequestError,
    );
    await expect(agent.extMethod("_anyharness/goal/bogus", { sessionId: "s1" })).rejects.toThrow(
      RequestError,
    );
    await expect(agent.extMethod("_anyharness/goal/get", {})).rejects.toThrow(RequestError);
    await expect(agent.extMethod("_anyharness/goal/get", { sessionId: "nope" })).rejects.toThrow(
      RequestError,
    );
  });

  it("clears the shared in-flight query.next() when it rejects so the session recovers", async () => {
    const { agent } = createAgent();
    const session = injectSession(agent, "s1");
    let calls = 0;
    // First next() rejects with a non-process-exit error; a later drain must
    // start a fresh next() instead of re-awaiting the poisoned promise.
    session.query = {
      next: () => {
        calls += 1;
        if (calls === 1) {
          return Promise.reject(new Error("transient transport hiccup"));
        }
        return Promise.resolve({ done: true, value: undefined });
      },
    } as unknown as (typeof session)["query"];

    await expect(
      (agent as unknown as { drainTurn: (p: unknown) => Promise<unknown> }).drainTurn({
        sessionId: "s1",
        session,
        owner: "prompt",
        promptUuid: "p1",
      }),
    ).rejects.toThrow("transient transport hiccup");
    expect(session.pendingQueryNext ?? null).toBeNull();

    // A subsequent drain must call next() fresh, not re-await the rejection.
    const outcome = await (
      agent as unknown as { drainTurn: (p: unknown) => Promise<unknown> }
    ).drainTurn({ sessionId: "s1", session, owner: "prompt", promptUuid: "p2" });
    expect(outcome).toEqual({ kind: "stream_ended" });
    expect(calls).toBe(2);
  });

  it("rejects goal/set with status paused (no native pause on claude)", async () => {
    const { agent } = createAgent();
    const session = injectSession(agent, "s1");
    const push = vi.spyOn(session.input, "push");
    await expect(
      agent.extMethod("_anyharness/goal/set", {
        sessionId: "s1",
        objective: "x",
        status: "paused",
      }),
    ).rejects.toMatchObject({ code: -32602 });
    expect(push).not.toHaveBeenCalled();
  });

  it("rejects an objective-less patch when no goal is active", async () => {
    const { agent } = createAgent();
    injectSession(agent, "s1");
    await expect(
      agent.extMethod("_anyharness/goal/set", { sessionId: "s1", status: "active" }),
    ).rejects.toMatchObject({ code: -32602 });
  });

  it("emits goal_updated for an objective-omitted status patch so anyharness confirms", async () => {
    const { agent, updates } = createAgent();
    const session = injectSession(agent, "s1", tempTranscript());
    session.anyharness.goal = {
      objective: "DONE.txt exists",
      status: "active",
      nativeStatus: "armed",
      metReason: null,
      iterations: null,
      tokensUsed: null,
      timeUsedSeconds: null,
      updatedAtMs: Date.now(),
    };

    // A status-only patch (no objective) is a no-op for claude, but it must
    // still emit the tagged notification the anyharness runtime waits on —
    // otherwise the confirmation wait times out into a 409.
    const result = (await agent.extMethod("_anyharness/goal/set", {
      sessionId: "s1",
      status: "active",
    })) as { goal: { objective: string; status: string } };
    expect(result.goal).toMatchObject({ objective: "DONE.txt exists", status: "active" });

    const events = updates
      .map((u) => (u.update as { _meta?: { anyharness?: { transcriptEvent?: string } } })._meta)
      .map((meta) => meta?.anyharness?.transcriptEvent)
      .filter(Boolean);
    expect(events).toEqual(["goal_updated"]);
  });

  it("confirms goal/set only after the native arm sentinel round-trips", async () => {
    const { agent, updates } = createAgent();
    const transcriptPath = tempTranscript();
    const session = injectSession(agent, "s1", transcriptPath);
    const push = vi.spyOn(session.input, "push");

    const pending = agent.extMethod("_anyharness/goal/set", {
      sessionId: "s1",
      objective: "DONE.txt exists",
    });
    await vi.waitFor(() => expect(push).toHaveBeenCalledOnce(), { timeout: 2000 });
    expect(push.mock.calls[0][0].message.content).toEqual([
      { type: "text", text: "/goal DONE.txt exists" },
    ]);
    expect(session.anyharness.goal).toBeNull();

    fs.writeFileSync(
      transcriptPath,
      goalStatusLine({ met: false, sentinel: true, condition: "DONE.txt exists" }),
    );
    const result = (await pending) as { goal: { status: string; objective: string } };
    expect(result.goal).toMatchObject({
      objective: "DONE.txt exists",
      status: "active",
      nativeStatus: "armed",
      native: true,
      metReason: null,
      tokenBudget: null,
    });

    const tagged = updates
      .map((u) => (u.update as { _meta?: { anyharness?: { transcriptEvent?: string } } })._meta)
      .filter((meta) => meta?.anyharness?.transcriptEvent);
    expect(tagged.map((meta) => meta!.anyharness!.transcriptEvent)).toEqual(["goal_updated"]);
  });

  it("confirms goal/clear via the clear sentinel and emits goal_cleared", async () => {
    const { agent, updates } = createAgent();
    const transcriptPath = tempTranscript();
    const session = injectSession(agent, "s1", transcriptPath);

    const set = agent.extMethod("_anyharness/goal/set", {
      sessionId: "s1",
      objective: "DONE.txt exists",
    });
    await vi.waitFor(() => expect(session.anyharness.tailer).not.toBeNull());
    fs.writeFileSync(
      transcriptPath,
      goalStatusLine({ met: false, sentinel: true, condition: "DONE.txt exists" }),
    );
    await set;

    const clear = agent.extMethod("_anyharness/goal/clear", { sessionId: "s1" });
    fs.appendFileSync(
      transcriptPath,
      goalStatusLine({ met: true, sentinel: true, condition: "DONE.txt exists" }),
    );
    expect(await clear).toEqual({ cleared: true });
    expect(session.anyharness.goal?.status).toBe("cleared");

    const events = updates
      .map(
        (u) =>
          (u.update as { _meta?: { anyharness?: { transcriptEvent?: string } } })._meta?.anyharness
            ?.transcriptEvent,
      )
      .filter(Boolean);
    expect(events).toEqual(["goal_updated", "goal_cleared"]);
  });

  it("returns cleared:false immediately when no goal is active but still sends the native clear", async () => {
    const { agent } = createAgent();
    const session = injectSession(agent, "s1", tempTranscript());
    const push = vi.spyOn(session.input, "push");
    expect(await agent.extMethod("_anyharness/goal/clear", { sessionId: "s1" })).toEqual({
      cleared: false,
    });
    expect(push.mock.calls[0][0].message.content).toEqual([{ type: "text", text: "/goal clear" }]);
  });

  it("mirrors a met evaluation row as goal_met with the evaluator's reason", async () => {
    const { agent, updates } = createAgent();
    const transcriptPath = tempTranscript();
    const session = injectSession(agent, "s1", transcriptPath);

    const set = agent.extMethod("_anyharness/goal/set", {
      sessionId: "s1",
      objective: "DONE.txt exists",
    });
    await vi.waitFor(() => expect(session.anyharness.tailer).not.toBeNull());
    fs.writeFileSync(
      transcriptPath,
      goalStatusLine({ met: false, sentinel: true, condition: "DONE.txt exists" }),
    );
    await set;

    fs.appendFileSync(
      transcriptPath,
      goalStatusLine({ met: true, condition: "DONE.txt exists", reason: "file is present" }),
    );
    await vi.waitFor(() => expect(session.anyharness.goal?.status).toBe("met"), { timeout: 3000 });
    expect(session.anyharness.goal?.metReason).toBe("file is present");

    const metUpdate = updates
      .map(
        (u) =>
          (
            u.update as {
              _meta?: { anyharness?: { transcriptEvent?: string; goal?: { metReason?: string } } };
            }
          )._meta?.anyharness,
      )
      .find((meta) => meta?.transcriptEvent === "goal_met");
    expect(metUpdate?.goal?.metReason).toBe("file is present");
  });

  function armLoop(session: ReturnType<typeof injectSession>, prompt: string) {
    session.anyharness.loops.set("loop-1", {
      loopId: "loop-1",
      prompt,
      schedule: { kind: "interval", expr: "5m" },
      recurring: true,
      status: "active",
      lastFiredAtMs: null,
      fireCount: 0,
      updatedAtMs: Date.now(),
    });
  }

  const drainTurn = (agent: ClaudeAcpAgent, params: unknown): Promise<unknown> =>
    (agent as unknown as { drainTurn: (p: unknown) => Promise<unknown> }).drainTurn(params);

  it("a prompt drain handed a cron-wake pre-turn emits loop_fired and ends on its own turn", async () => {
    const { agent, updates } = createAgent();
    const session = injectSession(agent, "s1");
    armLoop(session, "check the build");
    const promptUuid = "prompt-uuid-1";

    session.query = queryFrom([
      // Spontaneous cron-wake pre-turn (handed off from the interrupted pump).
      { type: "system", subtype: "init", session_id: "s1" },
      {
        type: "user",
        uuid: "wake-uuid",
        parent_tool_use_id: null,
        session_id: "s1",
        message: { role: "user", content: "check the build" },
      },
      resultMsg(100, 50),
      { type: "system", subtype: "session_state_changed", state: "idle", session_id: "s1" },
      // The prompt's own message replay + its own turn.
      { type: "system", subtype: "init", session_id: "s1" },
      {
        type: "user",
        uuid: promptUuid,
        parent_tool_use_id: null,
        session_id: "s1",
        message: { role: "user", content: "hello" },
      },
      resultMsg(7, 3),
      { type: "system", subtype: "session_state_changed", state: "idle", session_id: "s1" },
    ]);
    // Simulate the pump handoff: the in-flight next() is left on the session,
    // which is also the signal (pendingQueryNext != null) that this drain was
    // handed the stream mid-flight rather than owning it from the start.
    session.pendingQueryNext = session.query.next();

    const outcome = await drainTurn(agent, {
      sessionId: "s1",
      session,
      owner: "prompt",
      promptUuid,
    });
    // It ends on ITS OWN turn's idle, not the wake pre-turn's idle boundary.
    expect(outcome).toEqual({ kind: "turn_ended", stopReason: "end_turn" });

    // loop_fired is emitted for the wake even though a prompt drain saw it.
    const loopFired = updates
      .map(
        (u) =>
          (u.update as { _meta?: { anyharness?: { transcriptEvent?: string; loopId?: string } } })
            ._meta?.anyharness,
      )
      .filter((m) => m?.transcriptEvent === "loop_fired");
    expect(loopFired).toHaveLength(1);
    expect(loopFired[0]!.loopId).toBe("loop-1");
    expect(session.anyharness.loops.get("loop-1")?.fireCount).toBe(1);

    // The wake pre-turn's 100/50 usage must NOT pollute the prompt's own 7/3.
    expect(session.accumulatedUsage).toEqual({
      inputTokens: 7,
      outputTokens: 3,
      cachedReadTokens: 0,
      cachedWriteTokens: 0,
    });
  });

  it("does not fold an idle-pump turn's usage into accumulatedUsage", async () => {
    const { agent } = createAgent();
    const session = injectSession(agent, "s1");
    session.query = queryFrom([
      { type: "system", subtype: "init", session_id: "s1" },
      resultMsg(100, 50),
      { type: "system", subtype: "session_state_changed", state: "idle", session_id: "s1" },
    ]);

    const outcome = await drainTurn(agent, { sessionId: "s1", session, owner: "pump" });
    expect(outcome).toEqual({ kind: "turn_ended", stopReason: "end_turn" });
    expect(session.accumulatedUsage).toEqual({
      inputTokens: 0,
      outputTokens: 0,
      cachedReadTokens: 0,
      cachedWriteTokens: 0,
    });
  });
});
