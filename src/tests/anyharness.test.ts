import { describe, expect, it, vi } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { AgentSideConnection, RequestError, SessionNotification } from "@agentclientprotocol/sdk";
import { ClaudeAcpAgent } from "../acp-agent.js";
import {
  AnyharnessSessionState,
  classifyGoalStatus,
  extractCronId,
  extractGoalStatus,
  isSyntheticLoopId,
  LoopState,
  matchLoopForWake,
  newAnyharnessSessionState,
  parseBackgroundOutputFile,
  parseCronIdFromResult,
  readLastGoalStatus,
  reconcileSessionCrons,
  subagentFeedPath,
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

function activeLoop(overrides: Partial<LoopState> = {}): LoopState {
  return {
    loopId: "loop-1",
    prompt: "ping",
    schedule: { kind: "cron", expr: "*/1 * * * *" },
    recurring: true,
    status: "active",
    lastFiredAtMs: null,
    fireCount: 0,
    updatedAtMs: 0,
    ...overrides,
  };
}

describe("reconcileSessionCrons", () => {
  function state(): AnyharnessSessionState {
    return newAnyharnessSessionState(true);
  }

  it("upserts a cron present in the snapshot but absent from the mirror", () => {
    const s = state();
    const result = reconcileSessionCrons(
      s,
      [{ id: "cron-real-1", cron: "*/5 * * * *", prompt: "ping PING.log", recurring: true }],
      1000,
    );
    expect(result.upserted.map((l) => l.loopId)).toEqual(["cron-real-1"]);
    expect(result.removed).toEqual([]);
    const loop = s.loops.get("cron-real-1");
    expect(loop).toMatchObject({
      prompt: "ping PING.log",
      schedule: { kind: "cron", expr: "*/5 * * * *" },
      status: "active",
    });
  });

  it("is a no-op when the snapshot matches the mirror (steady state)", () => {
    const s = state();
    s.loops.set(
      "cron-real-1",
      activeLoop({
        loopId: "cron-real-1",
        prompt: "p",
        schedule: { kind: "cron", expr: "* * * * *" },
      }),
    );
    // NOTE: loopId "cron-real-1" is treated as a real id below only because it is
    // returned verbatim in the snapshot; the synthetic check is by prefix.
    const result = reconcileSessionCrons(
      s,
      [{ id: "cron-real-1", cron: "* * * * *", prompt: "p", recurring: true }],
      2000,
    );
    expect(result.upserted).toEqual([]);
    expect(result.removed).toEqual([]);
  });

  it("removes a real-id active loop that vanished from the snapshot", () => {
    const s = state();
    s.loops.set("job_abc", activeLoop({ loopId: "job_abc" }));
    const result = reconcileSessionCrons(s, [], 3000);
    expect(result.removed).toEqual(["job_abc"]);
    expect(s.loops.get("job_abc")?.status).toBe("cleared");
  });

  it("leaves synthetic-id loops untouched when absent from the snapshot", () => {
    const s = state();
    s.loops.set("provisional-xyz", activeLoop({ loopId: "provisional-xyz" }));
    s.loops.set("cron-abc", activeLoop({ loopId: "cron-abc" }));
    const result = reconcileSessionCrons(s, [], 4000);
    // Neither synthetic id is a confirmed external deletion.
    expect(result.removed).toEqual([]);
    expect(s.loops.get("provisional-xyz")?.status).toBe("active");
    expect(s.loops.get("cron-abc")?.status).toBe("active");
  });

  it("upgrades a synthetic loop to its real cron id by prompt match, carrying fire bookkeeping", () => {
    const s = state();
    s.loops.set(
      "provisional-xyz",
      activeLoop({
        loopId: "provisional-xyz",
        prompt: "ping PING.log",
        fireCount: 3,
        lastFiredAtMs: 999,
      }),
    );
    const result = reconcileSessionCrons(
      s,
      [{ jobId: "job_real", cron: "*/1 * * * *", prompt: "ping PING.log" }],
      5000,
    );
    expect(result.removed).toEqual(["provisional-xyz"]);
    expect(result.upserted.map((l) => l.loopId)).toEqual(["job_real"]);
    expect(s.loops.has("provisional-xyz")).toBe(false);
    expect(s.loops.get("job_real")).toMatchObject({ fireCount: 3, lastFiredAtMs: 999 });
  });

  it("ignores non-array snapshots and crons without an id", () => {
    const s = state();
    expect(reconcileSessionCrons(s, undefined, 0)).toEqual({ upserted: [], removed: [] });
    expect(reconcileSessionCrons(s, [{ prompt: "no id here" }], 0)).toEqual({
      upserted: [],
      removed: [],
    });
    expect(s.loops.size).toBe(0);
  });
});

describe("isSyntheticLoopId / extractCronId", () => {
  it("classifies our synthesized placeholder ids", () => {
    expect(isSyntheticLoopId("provisional-abcd")).toBe(true);
    expect(isSyntheticLoopId("cron-abcd")).toBe(true);
    expect(isSyntheticLoopId("job_real_123")).toBe(false);
  });

  it("extracts a cron id from varied IO shapes", () => {
    expect(extractCronId({ id: "j1" })).toBe("j1");
    expect(extractCronId({ jobId: 42 })).toBe("42");
    expect(extractCronId({ result: { cron_id: "c9" } })).toBe("c9");
    expect(extractCronId({ nothing: true })).toBeUndefined();
  });
});

describe("matchLoopForWake", () => {
  it("matches a wake to the loop whose prompt it replays exactly", () => {
    const a = activeLoop({ loopId: "a", prompt: "check the build" });
    const b = activeLoop({ loopId: "b", prompt: "poll the queue" });
    expect(matchLoopForWake([a, b], "poll the queue")?.loopId).toBe("b");
  });

  it("prefers an exact match over a containment match (no cross-loop credit)", () => {
    // "poll" is a substring of the other loop's prompt, but the exact match wins.
    const a = activeLoop({ loopId: "a", prompt: "poll the queue" });
    const b = activeLoop({ loopId: "b", prompt: "poll" });
    expect(matchLoopForWake([a, b], "poll")?.loopId).toBe("b");
  });

  it("refuses to guess when the text ambiguously contains multiple loop prompts", () => {
    const a = activeLoop({ loopId: "a", prompt: "build" });
    const b = activeLoop({ loopId: "b", prompt: "test" });
    // A wrapped wake prompt that contains BOTH loop prompts — attributing to
    // either would corrupt the wrong loop's bookkeeping, so refuse.
    expect(matchLoopForWake([a, b], "build and test the project")).toBeUndefined();
  });

  it("returns undefined when no loop prompt is related to the text", () => {
    const a = activeLoop({ loopId: "a", prompt: "check the build" });
    expect(matchLoopForWake([a], "unrelated goal continuation")).toBeUndefined();
  });
});

describe("parseCronIdFromResult", () => {
  // The exact CronCreate result string captured live from Claude Code 2.1.199.
  const liveResult =
    "Scheduled recurring job dad38e14 (Every minute). Session-only (not written to disk, " +
    "dies when Claude exits). Auto-expires after 7 days. Use CronDelete to cancel sooner.";

  it("parses the real cron id out of the live CronCreate prose result", () => {
    expect(parseCronIdFromResult(liveResult)).toBe("dad38e14");
  });

  it("parses from a content-block array and a wrapped object", () => {
    expect(parseCronIdFromResult([{ type: "text", text: liveResult }])).toBe("dad38e14");
    expect(parseCronIdFromResult({ content: liveResult })).toBe("dad38e14");
  });

  it("returns null when the result has no job id", () => {
    expect(parseCronIdFromResult("nothing scheduled")).toBeNull();
    expect(parseCronIdFromResult(null)).toBeNull();
  });
});

describe("parseBackgroundOutputFile", () => {
  it("parses the output-file path from a background-bash tool_result string", () => {
    expect(
      parseBackgroundOutputFile("Command running in background with ID: t1, output → /tmp/out.log"),
    ).toBe("/tmp/out.log");
    expect(parseBackgroundOutputFile("logs — output: /var/log/x.txt.")).toBe("/var/log/x.txt");
  });

  it("parses from an array-of-blocks tool_result body", () => {
    expect(
      parseBackgroundOutputFile([{ type: "text", text: "running, output -> /tmp/a.log" }]),
    ).toBe("/tmp/a.log");
  });

  it("returns null when no output path is present", () => {
    expect(parseBackgroundOutputFile("done")).toBeNull();
    expect(parseBackgroundOutputFile(null)).toBeNull();
  });
});

describe("subagentFeedPath", () => {
  it("derives the per-agent transcript path from the parent transcript", () => {
    expect(subagentFeedPath("/home/u/.claude/projects/proj/sess.jsonl", "sess", "task9")).toBe(
      "/home/u/.claude/projects/proj/sess/subagents/agent-task9.jsonl",
    );
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
      modes: { currentModeId: "default", availableModes: [] },
      models: { currentModelId: "default", availableModels: [] },
      modelCapabilitiesById: {},
      liveSettings: {},
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

  // Resolves the drain (500ms budget) or reports the wedge so the assertion
  // shows a clear diff instead of the whole suite timing out.
  const drainOrHang = (agent: ClaudeAcpAgent, params: unknown): Promise<unknown> =>
    Promise.race([
      drainTurn(agent, params),
      new Promise((resolve) => setTimeout(() => resolve("HANG"), 500)),
    ]);

  it("resolves the drain when an interrupt lands mid prompt-turn (no wedge, stopReason cancelled)", async () => {
    const { agent } = createAgent();
    const session = injectSession(agent, "s1");
    const promptUuid = "prompt-uuid-1";

    // Faithful ordering of the live wedge (goal met -> idle pump blocked on
    // next() -> user prompt "nice" arrives -> pump hands off its in-flight
    // next(), so this prompt drain starts in pre-turn mode, inOwnTurn=false):
    //   0: assistant streams a first chunk ("An").
    //   1: a client cancel lands here — cancel() sets session.cancelled and
    //      calls query.interrupt(); the CLI records "[Request interrupted by
    //      user]".
    //   1: the prompt's own user-message replay. A cancelled turn short-circuits
    //      before the uuid match, so inOwnTurn never flips to true.
    //   2: result (interrupted).
    //   3: session_state idle — the CLI is now idle and emits nothing further.
    const messages: any[] = [
      {
        type: "assistant",
        parent_tool_use_id: null,
        session_id: "s1",
        message: { role: "assistant", model: "claude", content: [{ type: "text", text: "An" }] },
      },
      {
        type: "user",
        uuid: promptUuid,
        parent_tool_use_id: null,
        session_id: "s1",
        message: { role: "user", content: "nice" },
      },
      resultMsg(5, 2),
      { type: "system", subtype: "session_state_changed", state: "idle", session_id: "s1" },
    ];
    let i = 0;
    const interrupt = vi.fn(async () => {});
    session.query = {
      next: async () => {
        // The interrupt lands right after the first chunk streamed, before the
        // prompt's own replay is drained.
        if (i === 1) session.cancelled = true;
        if (i < messages.length) return { done: false, value: messages[i++] };
        // After acknowledging the interrupt the CLI is idle: next() never
        // resolves again — exactly what wedged prompt() for 135s in the wild.
        return new Promise<never>(() => {});
      },
      interrupt,
    } as any;
    // Idle-pump handoff: the in-flight next() sits on the session, which is the
    // signal that puts the prompt drain into pre-turn mode (inOwnTurn=false).
    session.pendingQueryNext = session.query.next();

    const outcome = await drainOrHang(agent, {
      sessionId: "s1",
      session,
      owner: "prompt",
      promptUuid,
    });
    expect(outcome).not.toBe("HANG");
    expect(outcome).toEqual({ kind: "turn_ended", stopReason: "cancelled" });
  });

  it("ends a cancelled prompt drain on the next idle even with no trailing result", async () => {
    const { agent } = createAgent();
    const session = injectSession(agent, "s1");
    session.cancelled = true;
    let served = false;
    session.query = {
      next: async () => {
        if (!served) {
          served = true;
          return {
            done: false,
            value: {
              type: "system",
              subtype: "session_state_changed",
              state: "idle",
              session_id: "s1",
            },
          };
        }
        return new Promise<never>(() => {});
      },
    } as any;
    // Pre-turn mode + already cancelled: the interrupt was acknowledged and the
    // CLI emits only idle (no trailing result). The drain must resolve rather
    // than block forever on the next next().
    session.pendingQueryNext = session.query.next();

    const outcome = await drainOrHang(agent, {
      sessionId: "s1",
      session,
      owner: "prompt",
      promptUuid: "p1",
    });
    expect(outcome).toEqual({ kind: "turn_ended", stopReason: "cancelled" });
  });

  it("a normal prompt handed off by the idle pump streams to completion without a self-interrupt", async () => {
    const { agent } = createAgent();
    const session = injectSession(agent, "s1");
    const promptUuid = "prompt-uuid-1";
    const interrupt = vi.fn(async () => {});
    // No cron wake ahead: the handed-off stream is the prompt's OWN turn.
    const base = queryFrom([
      {
        type: "user",
        uuid: promptUuid,
        parent_tool_use_id: null,
        session_id: "s1",
        message: { role: "user", content: "nice" },
      },
      {
        type: "assistant",
        parent_tool_use_id: null,
        session_id: "s1",
        message: { role: "assistant", model: "claude", content: [{ type: "text", text: "An" }] },
      },
      resultMsg(5, 2),
      { type: "system", subtype: "session_state_changed", state: "idle", session_id: "s1" },
    ]);
    session.query = { next: base.next, interrupt } as any;
    // Idle-pump handoff: pre-turn mode (inOwnTurn=false).
    session.pendingQueryNext = session.query.next();

    const outcome = await drainOrHang(agent, {
      sessionId: "s1",
      session,
      owner: "prompt",
      promptUuid,
    });
    expect(outcome).toEqual({ kind: "turn_ended", stopReason: "end_turn" });
    // A normal prompt turn must never trigger an interrupt of its own turn.
    expect(interrupt).not.toHaveBeenCalled();
    expect(session.accumulatedUsage).toEqual({
      inputTokens: 5,
      outputTokens: 2,
      cachedReadTokens: 0,
      cachedWriteTokens: 0,
    });
  });

  const loopFiredEvents = (updates: SessionNotification[]) =>
    updates
      .map(
        (u) =>
          (u.update as { _meta?: { anyharness?: { transcriptEvent?: string; loopId?: string } } })
            ._meta?.anyharness,
      )
      .filter((m) => m?.transcriptEvent === "loop_fired");

  it("does not fire a loop for a spontaneous assistant turn with no matching wake prompt", async () => {
    // A goal continuation (or a background-task wake) drains as a bare
    // spontaneous assistant turn while a loop is armed. It carries no wake
    // user-prompt, so it must NOT be miscounted as a loop fire.
    const { agent, updates } = createAgent();
    const session = injectSession(agent, "s1");
    armLoop(session, "check the build");

    session.query = queryFrom([
      { type: "system", subtype: "init", session_id: "s1" },
      {
        type: "assistant",
        parent_tool_use_id: null,
        session_id: "s1",
        message: {
          role: "assistant",
          model: "claude",
          content: [{ type: "text", text: "continuing toward the goal" }],
        },
      },
      resultMsg(40, 20),
      { type: "system", subtype: "session_state_changed", state: "idle", session_id: "s1" },
    ]);

    const outcome = await drainTurn(agent, { sessionId: "s1", session, owner: "pump" });
    expect(outcome).toEqual({ kind: "turn_ended", stopReason: "end_turn" });
    expect(loopFiredEvents(updates)).toHaveLength(0);
    expect(session.anyharness.loops.get("loop-1")?.fireCount).toBe(0);
  });

  it("credits a cron wake to the loop whose prompt it replays, not loops[0]", async () => {
    // Two loops armed: a genuine wake for loop B must move loop B's bookkeeping,
    // never loop A's (the old loops[0] fallback credited the wrong loop).
    const { agent, updates } = createAgent();
    const session = injectSession(agent, "s1");
    session.anyharness.loops.set("loop-a", activeLoop({ loopId: "loop-a", prompt: "watch A" }));
    session.anyharness.loops.set("loop-b", activeLoop({ loopId: "loop-b", prompt: "watch B" }));

    session.query = queryFrom([
      { type: "system", subtype: "init", session_id: "s1" },
      {
        type: "user",
        uuid: "wake-b",
        parent_tool_use_id: null,
        session_id: "s1",
        message: { role: "user", content: "watch B" },
      },
      {
        type: "assistant",
        parent_tool_use_id: null,
        session_id: "s1",
        message: { role: "assistant", model: "claude", content: [{ type: "text", text: "on it" }] },
      },
      resultMsg(10, 5),
      { type: "system", subtype: "session_state_changed", state: "idle", session_id: "s1" },
    ]);

    const outcome = await drainTurn(agent, { sessionId: "s1", session, owner: "pump" });
    expect(outcome).toEqual({ kind: "turn_ended", stopReason: "end_turn" });

    const fired = loopFiredEvents(updates);
    expect(fired.map((m) => m!.loopId)).toEqual(["loop-b"]);
    expect(session.anyharness.loops.get("loop-b")?.fireCount).toBe(1);
    expect(session.anyharness.loops.get("loop-a")?.fireCount).toBe(0);
  });

  // --- Helpers to reach the private roster / reconcile / injection paths. ---
  const call = (agent: ClaudeAcpAgent, method: string, ...args: unknown[]): unknown =>
    (agent as unknown as Record<string, (...a: unknown[]) => unknown>)[method](...args);

  function anyharnessEvents(updates: SessionNotification[]): {
    transcriptEvent?: string;
    loopId?: string;
    loop?: { loopId?: string };
    process?: Record<string, unknown>;
    subagent?: Record<string, unknown>;
  }[] {
    return updates
      .map(
        (u) => (u.update as { _meta?: { anyharness?: Record<string, unknown> } })._meta?.anyharness,
      )
      .filter((m): m is NonNullable<typeof m> => !!m && !!m.transcriptEvent) as any;
  }

  describe("session_crons reconcile emission", () => {
    it("emits loop_upserted for a newly-observed cron then loop_removed when it vanishes", async () => {
      const { agent, updates } = createAgent();
      const session = injectSession(agent, "s1");

      await call(agent, "handleSessionCrons", "s1", [
        { id: "job_1", cron: "*/1 * * * *", prompt: "append ping", recurring: true },
      ]);
      expect(session.anyharness.loops.get("job_1")?.status).toBe("active");
      let events = anyharnessEvents(updates);
      expect(events.map((e) => e.transcriptEvent)).toEqual(["loop_upserted"]);
      expect(events[0].loop?.loopId).toBe("job_1");

      // A steady-state snapshot emits nothing.
      await call(agent, "handleSessionCrons", "s1", [
        { id: "job_1", cron: "*/1 * * * *", prompt: "append ping", recurring: true },
      ]);
      expect(anyharnessEvents(updates)).toHaveLength(1);

      // The cron disappears (deleted out of band) → loop_removed.
      await call(agent, "handleSessionCrons", "s1", []);
      events = anyharnessEvents(updates);
      expect(events.map((e) => e.transcriptEvent)).toEqual(["loop_upserted", "loop_removed"]);
      expect(events[1].loopId).toBe("job_1");
      expect(session.anyharness.loops.get("job_1")?.status).toBe("cleared");
    });
  });

  describe("CronCreate observation", () => {
    it("mirrors the real cron id parsed from the live result string and emits loop_upserted", async () => {
      const { agent, updates } = createAgent();
      const session = injectSession(agent, "s1");
      // A loop/set is awaiting its CronCreate.
      let resolved: LoopState | null = null;
      session.anyharness.pendingLoopSets.push({
        prompt: "append the word ping to PING.log",
        schedule: { kind: "interval", expr: "1m" },
        recurring: true,
        requestedAtMs: Date.now(),
        resolve: (loop) => {
          resolved = loop;
        },
      });

      await call(
        agent,
        "handleCronTool",
        "s1",
        "CronCreate",
        { cron: "*/1 * * * *", prompt: "append the word ping to PING.log", recurring: true },
        "Scheduled recurring job dad38e14 (Every minute). Session-only. Use CronDelete to cancel.",
      );

      // The real harness id — not a synthesized "cron-…" placeholder.
      expect(session.anyharness.loops.has("dad38e14")).toBe(true);
      expect(resolved).not.toBeNull();
      expect(resolved!.loopId).toBe("dad38e14");
      const events = anyharnessEvents(updates);
      expect(events.map((e) => e.transcriptEvent)).toEqual(["loop_upserted"]);
      expect(events[0].loop?.loopId).toBe("dad38e14");
    });
  });

  describe("activity roster emission", () => {
    it("task_started (local_bash) emits process_upserted with the captured command", async () => {
      const { agent, updates } = createAgent();
      const session = injectSession(agent, "s1", tempTranscript());

      // The spawning assistant tool_use carries the command; captureTaskIo files it.
      call(agent, "captureTaskIo", "s1", {
        type: "assistant",
        message: {
          role: "assistant",
          content: [
            {
              type: "tool_use",
              id: "tu1",
              name: "Bash",
              input: { command: "sleep 30 && echo OK" },
            },
          ],
        },
      });

      await call(agent, "handleTaskEvent", "s1", {
        type: "system",
        subtype: "task_started",
        task_id: "t1",
        tool_use_id: "tu1",
        task_type: "local_bash",
        description: "run sleep",
      });

      const proc = session.anyharness.processes.get("t1");
      expect(proc).toMatchObject({ id: "t1", command: "sleep 30 && echo OK", status: "running" });
      const events = anyharnessEvents(updates);
      expect(events.map((e) => e.transcriptEvent)).toEqual(["process_upserted"]);
      expect(events[0].process).toMatchObject({
        id: "t1",
        status: "running",
        command: "sleep 30 && echo OK",
      });
    });

    it("captures the output file from the tool_result and opens a live feed on the running process", async () => {
      const { agent, updates } = createAgent();
      const session = injectSession(agent, "s1", tempTranscript());
      await call(agent, "handleTaskEvent", "s1", {
        type: "system",
        subtype: "task_started",
        task_id: "t1",
        tool_use_id: "tu1",
        task_type: "local_bash",
        description: "run",
      });
      expect(session.anyharness.processes.get("t1")?.feed).toBeNull();

      call(agent, "captureTaskIo", "s1", {
        type: "user",
        message: {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "tu1",
              content: "Command running in background with ID: t1, output → /tmp/out.log",
            },
          ],
        },
      });

      expect(session.anyharness.processes.get("t1")?.feed).toEqual({
        transport: "tail_file",
        path: "/tmp/out.log",
      });
      // The feed discovery re-emits process_upserted so the runtime can attach.
      const feedEvents = anyharnessEvents(updates).filter(
        (e) => e.transcriptEvent === "process_upserted" && e.process?.feed,
      );
      expect(feedEvents).toHaveLength(1);
    });

    it("task_notification flips the process to exited and attaches the output-file feed", async () => {
      const { agent, updates } = createAgent();
      const session = injectSession(agent, "s1", tempTranscript());
      await call(agent, "handleTaskEvent", "s1", {
        type: "system",
        subtype: "task_started",
        task_id: "t1",
        tool_use_id: "tu1",
        task_type: "local_bash",
        description: "run",
      });
      await call(agent, "handleTaskEvent", "s1", {
        type: "system",
        subtype: "task_notification",
        task_id: "t1",
        status: "completed",
        output_file: "/tmp/out.log",
        summary: "done",
      });
      const proc = session.anyharness.processes.get("t1");
      expect(proc).toMatchObject({ status: "exited", exitCode: null });
      expect(proc?.endedAtMs).not.toBeNull();
      expect(proc?.feed).toEqual({ transport: "tail_file", path: "/tmp/out.log" });
      const allEvents = anyharnessEvents(updates);
      const last = allEvents[allEvents.length - 1];
      expect(last?.transcriptEvent).toBe("process_upserted");
      expect(last?.process).toMatchObject({ status: "exited" });
    });

    it("task_started (local_agent) emits subagent_upserted with a derived transcript feed", async () => {
      const { agent, updates } = createAgent();
      const session = injectSession(agent, "s1", "/home/u/.claude/projects/proj/s1.jsonl");
      await call(agent, "handleTaskEvent", "s1", {
        type: "system",
        subtype: "task_started",
        task_id: "a1",
        tool_use_id: "tu2",
        task_type: "local_agent",
        subagent_type: "code-reviewer",
        description: "review the diff",
        prompt: "please review",
      });
      const sub = session.anyharness.subagents.get("a1");
      expect(sub).toMatchObject({
        id: "a1",
        agentType: "code-reviewer",
        prompt: "please review",
        status: "running",
        background: true,
      });
      expect(sub?.feed).toEqual({
        transport: "tail_file",
        path: "/home/u/.claude/projects/proj/s1/subagents/agent-a1.jsonl",
      });
      const events = anyharnessEvents(updates);
      expect(events.map((e) => e.transcriptEvent)).toEqual(["subagent_upserted"]);
    });

    it("task_progress updates subagent usage; task_notification failed flips it to failed", async () => {
      const { agent, updates } = createAgent();
      const session = injectSession(agent, "s1", tempTranscript());
      await call(agent, "handleTaskEvent", "s1", {
        type: "system",
        subtype: "task_started",
        task_id: "a1",
        task_type: "local_agent",
        subagent_type: "general-purpose",
      });
      await call(agent, "handleTaskEvent", "s1", {
        type: "system",
        subtype: "task_progress",
        task_id: "a1",
        usage: { total_tokens: 1200, tool_uses: 4, duration_ms: 8000 },
      });
      // Usage is stored as FLAT sibling fields (seconds, not ms) — the shape the
      // ActivitySubagentWire contract deserializes. A nested `usage` object would
      // make the runtime read them as absent.
      const stored = session.anyharness.subagents.get("a1") as unknown as Record<string, unknown>;
      expect(stored.tokensUsed).toBe(1200);
      expect(stored.toolCalls).toBe(4);
      expect(stored.durationSeconds).toBe(8);
      expect("usage" in stored).toBe(false);
      // The emitted wire payload carries the same flat fields.
      const progressEvents = anyharnessEvents(updates);
      const progressEvent = progressEvents[progressEvents.length - 1];
      expect(progressEvent.subagent).toMatchObject({
        tokensUsed: 1200,
        toolCalls: 4,
        durationSeconds: 8,
      });
      expect("usage" in (progressEvent.subagent ?? {})).toBe(false);
      await call(agent, "handleTaskEvent", "s1", {
        type: "system",
        subtype: "task_notification",
        task_id: "a1",
        status: "failed",
        output_file: "/x/agent-a1.jsonl",
        summary: "blew up",
      });
      const sub = session.anyharness.subagents.get("a1");
      expect(sub).toMatchObject({ status: "failed", summary: "blew up" });
      const events = anyharnessEvents(updates).map((e) => e.transcriptEvent);
      expect(events).toEqual(["subagent_upserted", "subagent_upserted", "subagent_upserted"]);
    });

    it("activity/list serves the whole mirror from tracked state", async () => {
      const { agent } = createAgent();
      const session = injectSession(agent, "s1", tempTranscript());
      await call(agent, "handleTaskEvent", "s1", {
        type: "system",
        subtype: "task_started",
        task_id: "t1",
        tool_use_id: "tu1",
        task_type: "local_bash",
        description: "run",
      });
      await call(agent, "handleSessionCrons", "s1", [
        { id: "job_1", cron: "* * * * *", prompt: "p" },
      ]);
      const result = (await agent.extMethod("_anyharness/activity/list", { sessionId: "s1" })) as {
        loops: unknown[];
        processes: { id: string }[];
        subagents: unknown[];
        goal: unknown;
      };
      expect(result.processes.map((p) => p.id)).toEqual(["t1"]);
      expect(result.loops).toHaveLength(1);
      expect(session.anyharness.processes.size).toBe(1);
    });
  });

  describe("goal deferral behind a streaming turn", () => {
    it("defers a mid-turn goal/set to the turn boundary and returns a provisional pending goal", async () => {
      const { agent } = createAgent();
      const session = injectSession(agent, "s1", tempTranscript());
      // A turn is streaming — a `/goal` now would degrade to a queued command.
      session.promptRunning = true;
      const push = vi.spyOn(session.input, "push");

      const result = (await agent.extMethod("_anyharness/goal/set", {
        sessionId: "s1",
        objective: "DONE.txt exists",
      })) as { goal: { objective: string; status: string; nativeStatus: string; native: boolean } };

      // Returns immediately (no 30s block) with a provisional pending goal.
      expect(result.goal).toMatchObject({
        objective: "DONE.txt exists",
        status: "active",
        nativeStatus: "pending_injection",
        native: true,
      });
      // Nothing injected yet, and the mirror is untouched (no optimistic state).
      expect(push).not.toHaveBeenCalled();
      expect(session.anyharness.deferredInjections).toHaveLength(1);
      expect(session.anyharness.goal).toBeNull();

      // The turn ends → flush injects the deferred /goal at the boundary.
      session.promptRunning = false;
      session.anyharness.turnActive = false;
      call(agent, "tryFlushDeferredInjections", "s1");
      expect(push).toHaveBeenCalledOnce();
      expect(push.mock.calls[0][0].message.content).toEqual([
        { type: "text", text: "/goal DONE.txt exists" },
      ]);
      expect(session.anyharness.deferredInjections).toHaveLength(0);
    });

    it("injects immediately when the session is idle (unchanged idle path)", async () => {
      const { agent } = createAgent();
      const session = injectSession(agent, "s1", tempTranscript());
      const push = vi.spyOn(session.input, "push");
      // Fire and forget: the idle path blocks on the arm sentinel we never write.
      void agent.extMethod("_anyharness/goal/set", { sessionId: "s1", objective: "x" });
      await vi.waitFor(() => expect(push).toHaveBeenCalledOnce(), { timeout: 2000 });
      expect(push.mock.calls[0][0].message.content).toEqual([{ type: "text", text: "/goal x" }]);
      expect(session.anyharness.deferredInjections).toHaveLength(0);
    });
  });

  describe("loop/set deferral behind a streaming turn", () => {
    it("defers a mid-turn loop/set and returns a provisional loop without minting a duplicate", async () => {
      const { agent } = createAgent();
      const session = injectSession(agent, "s1", tempTranscript());
      // A turn is streaming — a `/loop` now would degrade to a queued command,
      // and (before the fix) the 60s CronCreate race would time out and mint a
      // phantom loop that the later real CronCreate would duplicate.
      session.promptRunning = true;
      const push = vi.spyOn(session.input, "push");

      // Resolves immediately (no 60s block) with a provisional loop.
      const result = (await agent.extMethod("_anyharness/loop/set", {
        sessionId: "s1",
        prompt: "append ping to PING.log",
        schedule: { kind: "interval", expr: "1m" },
      })) as { loop: { loopId: string; prompt: string; status: string; native: boolean } };
      expect(result.loop).toMatchObject({
        prompt: "append ping to PING.log",
        status: "active",
        native: true,
      });
      expect(isSyntheticLoopId(result.loop.loopId)).toBe(true);

      // Nothing injected yet; the mirror is untouched (no optimistic loop), but a
      // pending set is registered so the deferred /loop's CronCreate attributes here.
      expect(push).not.toHaveBeenCalled();
      expect(session.anyharness.loops.size).toBe(0);
      expect(session.anyharness.deferredInjections).toHaveLength(1);
      expect(session.anyharness.pendingLoopSets).toHaveLength(1);

      // The turn ends → flush injects the deferred /loop at the boundary.
      session.promptRunning = false;
      session.anyharness.turnActive = false;
      call(agent, "tryFlushDeferredInjections", "s1");
      expect(push).toHaveBeenCalledOnce();
      expect(push.mock.calls[0][0].message.content).toEqual([
        { type: "text", text: "/loop 1m append ping to PING.log" },
      ]);

      // The /loop runs and its CronCreate is observed — one real loop, no duplicate.
      await call(
        agent,
        "handleCronTool",
        "s1",
        "CronCreate",
        { cron: "*/1 * * * *", prompt: "append ping to PING.log", recurring: true },
        "Scheduled recurring job realjob42 (Every minute). Session-only. Use CronDelete to cancel.",
      );
      expect([...session.anyharness.loops.keys()]).toEqual(["realjob42"]);
      expect(session.anyharness.pendingLoopSets).toHaveLength(0);
    });

    it("collapses an idle-path provisional loop into the real cron when its CronCreate lands late", async () => {
      const { agent, updates } = createAgent();
      const session = injectSession(agent, "s1", tempTranscript());
      // Simulate the idle-path RPC timeout: a "provisional-" loop is already in
      // the mirror (with accrued fire bookkeeping) and its pending set is gone.
      session.anyharness.loops.set(
        "provisional-abc",
        activeLoop({
          loopId: "provisional-abc",
          prompt: "append ping to PING.log",
          fireCount: 2,
          lastFiredAtMs: 111,
        }),
      );

      await call(
        agent,
        "handleCronTool",
        "s1",
        "CronCreate",
        { cron: "*/1 * * * *", prompt: "append ping to PING.log", recurring: true },
        "Scheduled recurring job realjob99 (Every minute). Session-only. Use CronDelete to cancel.",
      );

      // Exactly one loop — the real id — carrying the placeholder's fire count.
      expect([...session.anyharness.loops.keys()]).toEqual(["realjob99"]);
      const real = session.anyharness.loops.get("realjob99")!;
      expect(real.fireCount).toBe(2);
      expect(real.lastFiredAtMs).toBe(111);
      // The client is told to add the real loop and drop the placeholder.
      const events = anyharnessEvents(updates);
      expect(events.map((e) => e.transcriptEvent)).toEqual(["loop_upserted", "loop_removed"]);
      expect(events[0].loop?.loopId).toBe("realjob99");
      expect(events[1].loopId).toBe("provisional-abc");
    });
  });
});
