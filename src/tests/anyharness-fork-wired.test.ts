import { describe, it, expect, beforeEach, vi } from "vitest";
import { randomUUID } from "node:crypto";
import type { Options } from "@anthropic-ai/claude-agent-sdk";
import type { SessionNotification } from "@agentclientprotocol/sdk";
import {
  ANYHARNESS_FORK_EXTENSION_VERSION,
  type AnyharnessForkSessionCapabilityMeta,
} from "../anyharness-fork.js";
import { Pushable } from "../utils.js";

// ---------------------------------------------------------------------------
// Mock seam: capture every `query()` call's options (the same options object
// that ends up carrying `resume`/`forkSession`/`resumeSessionAt`, since
// createSession spreads `creationOpts` directly onto the SDK `Options` — see
// acp-agent.ts's `const options: Options = { ..., ...creationOpts, ... }`),
// and let each test queue a per-call message generator so the parent session
// can stream messages (populating `messageIdToUuid`) while the fork call
// itself only needs to be captured, never iterated. Modeled on
// create-session-options.test.ts's capturedOptions seam plus
// activity.test.ts's makeGenerator/userEcho harness.
// ---------------------------------------------------------------------------

let capturedOptionsList: Options[] = [];
/** One entry consumed per `query()` call, in order (parent creation, then the
 *  fork's own `createSession`, etc). Falls back to an empty generator when the
 *  queue is exhausted (e.g. the fork call, which is never iterated here). */
let generatorQueue: Array<(input: Pushable<unknown>) => AsyncGenerator<unknown>> = [];

vi.mock("@anthropic-ai/claude-agent-sdk", async () => {
  const actual = await vi.importActual<typeof import("@anthropic-ai/claude-agent-sdk")>(
    "@anthropic-ai/claude-agent-sdk",
  );
  const { makeMockQuery, DEFAULT_CONTEXT_USAGE } = await import("./helpers.js");
  return {
    ...actual,
    deleteSession: vi.fn(),
    getSessionInfo: vi.fn().mockRejectedValue(new Error("no session file in this test")),
    getSessionMessages: vi.fn(),
    query: (args: { prompt: Pushable<unknown>; options: Options }) => {
      capturedOptionsList.push(args.options);
      const makeGenerator = generatorQueue.shift();
      const generator = makeGenerator ? makeGenerator(args.prompt) : (async function* () {})();
      // The query object IS iterated directly via `.next()` (runConsumer calls
      // `session.query.next()`, not `session.query[Symbol.asyncIterator]()`),
      // so the mock generator itself must carry the extra Query methods —
      // mirrors activity.test.ts's wrapQuery, not makeMockQuery's default
      // (non-generator) asyncIterator stub.
      const base = makeMockQuery({
        initializationResult: async () => ({
          models: [
            {
              value: "claude-sonnet-4-6",
              displayName: "Claude Sonnet",
              description: "Fast",
              supportsAutoMode: true,
            },
          ],
        }),
        getContextUsage: () => Promise.resolve(DEFAULT_CONTEXT_USAGE),
      });
      // Deliberately drop `base`'s own [Symbol.asyncIterator] stub — the
      // generator's is the one that must survive so `.next()` actually drains
      // the queued messages.
      delete (base as { [Symbol.asyncIterator]?: unknown })[Symbol.asyncIterator];
      return Object.assign(generator, base, {
        interrupt: vi.fn(async () => {}),
        close: vi.fn(),
        setModel: vi.fn(async () => {}),
      });
    },
  };
});

vi.mock("../tools.js", async () => {
  const actual = await vi.importActual<typeof import("../tools.js")>("../tools.js");
  return {
    ...actual,
    registerHookCallback: vi.fn(),
  };
});

/** Build the replayed `user` message the SDK echoes back for a pushed prompt,
 *  exactly like activity.test.ts's own helper of the same name. */
function userEcho(u: any) {
  return {
    type: "user",
    message: u.message,
    parent_tool_use_id: null,
    uuid: u.uuid,
    session_id: "irrelevant-not-filtered-on",
    isReplay: true,
  };
}

/** An assistant message carrying a distinct Anthropic API message id (`msg.id`)
 *  in addition to its SDK `uuid` — the case where the ACP anchor id
 *  (`messageIdForGrouping` returns the API id for assistant messages) and the
 *  SDK uuid the runtime keys `resumeSessionAt` on genuinely differ. */
function assistantMessageWithApiId(apiId: string) {
  return {
    type: "assistant",
    parent_tool_use_id: null,
    uuid: randomUUID(),
    session_id: "irrelevant-not-filtered-on",
    message: {
      id: apiId,
      role: "assistant",
      model: "claude-sonnet-4-5",
      stop_reason: null,
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        cache_read_input_tokens: 0,
        cache_creation_input_tokens: 0,
      },
      content: [{ type: "text", text: "hi" }],
    },
  };
}

function successResult() {
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
    session_id: "irrelevant-not-filtered-on",
  };
}

/** Replays the prompt's user echo (activating the turn), then the given
 *  messages, then settles at idle — mirrors activity.test.ts's makeGenerator. */
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

function createMockClient() {
  const updates: SessionNotification[] = [];
  return {
    client: {
      sessionUpdate: async (n: SessionNotification) => {
        updates.push(n);
      },
      requestPermission: async () => ({ outcome: { outcome: "cancelled" } }),
      readTextFile: async () => ({ content: "" }),
      writeTextFile: async () => ({}),
    } as any,
    updates,
  };
}

describe("fork session capability advertisement (initialize)", () => {
  let ClaudeAcpAgent: typeof import("../acp-agent.js").ClaudeAcpAgent;

  beforeEach(async () => {
    capturedOptionsList = [];
    generatorQueue = [];
    vi.resetModules();
    const acpAgent = await import("../acp-agent.js");
    ClaudeAcpAgent = acpAgent.ClaudeAcpAgent;
  });

  /** Mirrors `has_anyharness_targeted_fork_extension` in the AnyHarness
   *  runtime: the runtime probes exactly this location/shape on the fork
   *  session capability's own `_meta`, never on the top-level `_meta.anyharness`
   *  discovery block (which stays additive/human-facing). */
  function hasAnyharnessTargetedForkExtension(meta: unknown): meta is {
    anyharness: AnyharnessForkSessionCapabilityMeta["anyharness"];
  } {
    if (!meta || typeof meta !== "object") return false;
    const anyharness = (meta as Record<string, unknown>).anyharness;
    if (!anyharness || typeof anyharness !== "object") return false;
    const a = anyharness as Record<string, unknown>;
    if (typeof a.schemaVersion !== "number" || a.schemaVersion !== 1) return false;
    if (!a.targetedFork || typeof a.targetedFork !== "object") return false;
    const tf = a.targetedFork as Record<string, unknown>;
    if (tf.fileEffects !== "none") return false;
    if (tf.target !== "message_id" && tf.target !== "user_message_index") return false;
    return true;
  }

  it("advertises a fork session capability _meta the runtime probe accepts, targeting message_id", async () => {
    const { client } = createMockClient();
    const agent = new ClaudeAcpAgent(client, { log: () => {}, error: () => {} });

    const result = await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
    });

    const forkMeta = result.agentCapabilities?.sessionCapabilities?.fork?._meta;
    expect(hasAnyharnessTargetedForkExtension(forkMeta)).toBe(true);
    expect((forkMeta as any).anyharness.targetedFork.target).toBe("message_id");
  });

  it("keeps the top-level discovery block additive alongside the strict session-capability meta", async () => {
    const { client } = createMockClient();
    const agent = new ClaudeAcpAgent(client, { log: () => {}, error: () => {} });

    const result = await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
    });

    expect((result._meta as any).anyharness.fork).toEqual({
      version: ANYHARNESS_FORK_EXTENSION_VERSION,
      anchor: "upToMessageId",
    });
  });
});

describe("session/fork wired end-to-end (Design ADR A §3.4)", () => {
  let ClaudeAcpAgent: typeof import("../acp-agent.js").ClaudeAcpAgent;
  let getSessionMessages: ReturnType<typeof vi.fn>;

  beforeEach(async () => {
    capturedOptionsList = [];
    generatorQueue = [];
    vi.resetModules();
    const acpAgent = await import("../acp-agent.js");
    ClaudeAcpAgent = acpAgent.ClaudeAcpAgent;
    const sdk = await import("@anthropic-ai/claude-agent-sdk");
    getSessionMessages = vi.mocked(sdk.getSessionMessages);
    getSessionMessages.mockReset();
  });

  function sessionFor(agent: InstanceType<typeof ClaudeAcpAgent>, sessionId: string) {
    return (agent as unknown as { sessions: Record<string, any> }).sessions[sessionId];
  }

  it("anchored fork happy path: resolves the ACP message id to the mapped SDK uuid and forwards resume/forkSession/resumeSessionAt", async () => {
    const { client } = createMockClient();
    const agent = new ClaudeAcpAgent(client, { log: () => {}, error: () => {} });

    const apiMessageId = "msg_api_anchor_1";
    generatorQueue.push(makeGenerator([assistantMessageWithApiId(apiMessageId), successResult()]));

    const parent = await agent.newSession({ cwd: process.cwd(), mcpServers: [] });
    await agent.prompt({
      sessionId: parent.sessionId,
      prompt: [{ type: "text", text: "go" }],
    });

    const parentSession = sessionFor(agent, parent.sessionId);
    const mappedUuid = parentSession.messageIdToUuid.get(apiMessageId);
    expect(mappedUuid).toEqual(expect.any(String));

    // The fork's own createSession call: query() is invoked again, captured
    // but never iterated (the consumer only starts on a `prompt()` against the
    // forked session, which this test never issues).
    generatorQueue.push(makeGenerator([]));

    const forkResponse = await agent.unstable_forkSession({
      sessionId: parent.sessionId,
      cwd: process.cwd(),
      mcpServers: [],
      _meta: { anyharness: { upToMessageId: apiMessageId } },
    });

    expect(forkResponse.sessionId).not.toBe(parent.sessionId);
    const forkOptions = capturedOptionsList.at(-1)!;
    expect(forkOptions.resume).toBe(parent.sessionId);
    expect(forkOptions.forkSession).toBe(true);
    expect(forkOptions.resumeSessionAt).toBe(mappedUuid);
  });

  it("tip fork: no anchor meta forwards forkSession without resumeSessionAt", async () => {
    const { client } = createMockClient();
    const agent = new ClaudeAcpAgent(client, { log: () => {}, error: () => {} });

    generatorQueue.push(makeGenerator([]));
    const parent = await agent.newSession({ cwd: process.cwd(), mcpServers: [] });

    generatorQueue.push(makeGenerator([]));
    const forkResponse = await agent.unstable_forkSession({
      sessionId: parent.sessionId,
      cwd: process.cwd(),
      mcpServers: [],
    });

    expect(forkResponse.sessionId).not.toBe(parent.sessionId);
    const forkOptions = capturedOptionsList.at(-1)!;
    expect(forkOptions.resume).toBe(parent.sessionId);
    expect(forkOptions.forkSession).toBe(true);
    expect(forkOptions.resumeSessionAt).toBeUndefined();
  });

  it("malformed anchor (empty string) rejects invalidParams and never calls query for the fork", async () => {
    const { client } = createMockClient();
    const agent = new ClaudeAcpAgent(client, { log: () => {}, error: () => {} });

    generatorQueue.push(makeGenerator([]));
    const parent = await agent.newSession({ cwd: process.cwd(), mcpServers: [] });
    const callsBeforeFork = capturedOptionsList.length;

    await expect(
      agent.unstable_forkSession({
        sessionId: parent.sessionId,
        cwd: process.cwd(),
        mcpServers: [],
        _meta: { anyharness: { upToMessageId: "" } },
      }),
    ).rejects.toMatchObject({ code: -32602 });

    expect(capturedOptionsList.length).toBe(callsBeforeFork);
  });

  it("malformed anchor (non-string) rejects invalidParams and never calls query for the fork", async () => {
    const { client } = createMockClient();
    const agent = new ClaudeAcpAgent(client, { log: () => {}, error: () => {} });

    generatorQueue.push(makeGenerator([]));
    const parent = await agent.newSession({ cwd: process.cwd(), mcpServers: [] });
    const callsBeforeFork = capturedOptionsList.length;

    await expect(
      agent.unstable_forkSession({
        sessionId: parent.sessionId,
        cwd: process.cwd(),
        mcpServers: [],
        _meta: { anyharness: { upToMessageId: 42 } },
      }),
    ).rejects.toMatchObject({ code: -32602 });

    expect(capturedOptionsList.length).toBe(callsBeforeFork);
  });

  it("unresolvable anchor (well-formed but unknown id) rejects invalidParams and never calls query for the fork", async () => {
    const { client } = createMockClient();
    const agent = new ClaudeAcpAgent(client, { log: () => {}, error: () => {} });

    generatorQueue.push(makeGenerator([]));
    const parent = await agent.newSession({ cwd: process.cwd(), mcpServers: [] });
    const callsBeforeFork = capturedOptionsList.length;

    await expect(
      agent.unstable_forkSession({
        sessionId: parent.sessionId,
        cwd: process.cwd(),
        mcpServers: [],
        _meta: { anyharness: { upToMessageId: "msg_never_seen" } },
      }),
    ).rejects.toMatchObject({ code: -32602 });

    expect(capturedOptionsList.length).toBe(callsBeforeFork);
  });

  it("non-resident source session rejects invalidParams and never calls query for the fork", async () => {
    const { client } = createMockClient();
    const agent = new ClaudeAcpAgent(client, { log: () => {}, error: () => {} });

    const callsBeforeFork = capturedOptionsList.length;

    await expect(
      agent.unstable_forkSession({
        sessionId: "session-not-resident",
        cwd: process.cwd(),
        mcpServers: [],
        _meta: { anyharness: { upToMessageId: "msg_1" } },
      }),
    ).rejects.toMatchObject({ code: -32602 });

    expect(capturedOptionsList.length).toBe(callsBeforeFork);
  });

  // -------------------------------------------------------------------------
  // Q-A3 residency qualification fixture: a session loaded/resumed via
  // session/load replays history (replaySessionHistory -> getSessionMessages)
  // and populates messageIdToUuid BEFORE any anchored session/fork is issued —
  // proving the map is safe to drive from a child actor that loads first and
  // forks second, not just from live prompt traffic.
  // -------------------------------------------------------------------------
  it("a replayed (session/load) transcript populates messageIdToUuid, and an anchored fork against a negative control (never-replayed session) still fails invalidParams", async () => {
    const { client } = createMockClient();
    const agent = new ClaudeAcpAgent(client, { log: () => {}, error: () => {} });

    const replayedSessionId = "replayed-session-1";
    const replayedApiMessageId = "msg_replayed_1";
    const replayedUuid = randomUUID();

    getSessionMessages.mockResolvedValueOnce([
      {
        type: "assistant",
        uuid: replayedUuid,
        message: {
          id: replayedApiMessageId,
          role: "assistant",
          content: [{ type: "text", text: "replayed answer" }],
        },
      },
    ] as any);

    // session/load: createSession(resume: replayedSessionId) calls query()
    // once (never iterated — replay comes from getSessionMessages, not the
    // live stream), then replaySessionHistory backfills messageIdToUuid.
    generatorQueue.push(makeGenerator([]));
    await agent.loadSession({
      sessionId: replayedSessionId,
      cwd: process.cwd(),
      mcpServers: [],
    });

    const loadedSession = sessionFor(agent, replayedSessionId);
    expect(loadedSession.messageIdToUuid.get(replayedApiMessageId)).toBe(replayedUuid);

    // Positive: the anchored fork against the replay-populated id resolves.
    generatorQueue.push(makeGenerator([]));
    const forkResponse = await agent.unstable_forkSession({
      sessionId: replayedSessionId,
      cwd: process.cwd(),
      mcpServers: [],
      _meta: { anyharness: { upToMessageId: replayedApiMessageId } },
    });
    expect(forkResponse.sessionId).not.toBe(replayedSessionId);
    const forkOptions = capturedOptionsList.at(-1)!;
    expect(forkOptions.resume).toBe(replayedSessionId);
    expect(forkOptions.forkSession).toBe(true);
    expect(forkOptions.resumeSessionAt).toBe(replayedUuid);

    // Negative control: a fresh session that has NOT replayed anything has an
    // empty map, so the same anchor id (well-formed, but unknown here) fails.
    generatorQueue.push(makeGenerator([]));
    const fresh = await agent.newSession({ cwd: process.cwd(), mcpServers: [] });
    const callsBeforeNegativeFork = capturedOptionsList.length;

    await expect(
      agent.unstable_forkSession({
        sessionId: fresh.sessionId,
        cwd: process.cwd(),
        mcpServers: [],
        _meta: { anyharness: { upToMessageId: replayedApiMessageId } },
      }),
    ).rejects.toMatchObject({ code: -32602 });

    expect(capturedOptionsList.length).toBe(callsBeforeNegativeFork);
  });
});
