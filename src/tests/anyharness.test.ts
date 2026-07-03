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
});
