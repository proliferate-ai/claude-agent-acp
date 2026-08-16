import { RequestError } from "@agentclientprotocol/sdk";

/** The Proliferate ("AnyHarness") thin delta carried on top of canonical
 *  `claude-agent-acp`. Two capabilities, both offered upstream:
 *
 *   1. An INCLUSIVE fork anchor (`upToMessageId`) threaded onto the existing
 *      `session/fork` request via `_meta.anyharness.upToMessageId`. This
 *      deliberately extends the standard method rather than introducing a
 *      proprietary `_anyharness/fork/at` RPC — the runtime bridge the ADR
 *      names `_anyharness/fork/at` maps ONTO `session/fork` carrying this meta,
 *      keeping the wire addition upstreamable (a new optional field, not a new
 *      method). The anchor maps to the SDK's `resumeSessionAt` option, whose
 *      contract is "resume messages up to AND INCLUDING the message with this
 *      UUID" — the inclusive boundary the Forks ADR boundary model needs.
 *
 *   2. A labeled, provider-PARTIAL file rewind (`_anyharness/rewindFiles`),
 *      surfacing the SDK `Query.rewindFiles(userMessageId)` fast path. Its
 *      scope is ONLY Write/Edit/NotebookEdit-tracked changes — never Bash,
 *      manual, or external changes — and it requires file checkpointing to have
 *      been enabled at session creation (opt-in via
 *      `_meta.anyharness.enableFileCheckpointing`). It is NOT a complete
 *      restore; the ADR's complete file restore is the runtime-owned checkpoint
 *      layer (rung 7). This exposure exists so that layer can offer the labeled
 *      fast path where available.
 *
 *  Cardinal rule (ADR §5, "the cardinal sin"): a REQUESTED anchor must never
 *  silently degrade to an unanchored tip fork. A present-but-malformed anchor
 *  is a hard `invalidParams`; an anchor that cannot be resolved to an SDK uuid
 *  is a hard error at the call site — never a tip fork. */

export const ANYHARNESS_FORK_EXTENSION_VERSION = 1 as const;

/** Namespace key under a request/response `_meta` for every field in this
 *  extension, sibling to canonical's `_meta.goal` / `_meta.steering`. */
export const ANYHARNESS_META_NAMESPACE = "anyharness";

/** The labeled provider-partial file-rewind request method. */
export const REWIND_FILES_METHOD = "_anyharness/rewindFiles";

export type AnyharnessForkCapability = {
  version: typeof ANYHARNESS_FORK_EXTENSION_VERSION;
  /** Fork copies the transcript up to AND INCLUDING the anchored message. */
  anchor: "upToMessageId";
};

export type AnyharnessRewindFilesCapability = {
  version: typeof ANYHARNESS_FORK_EXTENSION_VERSION;
  controlMethod: typeof REWIND_FILES_METHOD;
  /** Honest scope label (ADR §4.7): tracked Write/Edit/NotebookEdit changes
   *  only, never Bash/manual/external changes. */
  scope: "writeEditNotebookEdit";
  /** True: rewind only functions when the session opted into checkpointing at
   *  creation, so clients know a session created without opt-in will report
   *  `canRewind:false` rather than silently succeeding. */
  requiresCheckpointingOptIn: true;
};

export type AnyharnessCapabilities = {
  fork: AnyharnessForkCapability;
  rewindFiles: AnyharnessRewindFilesCapability;
};

/** The `_meta.anyharness` capability block advertised at initialize. */
export function anyharnessCapabilities(): AnyharnessCapabilities {
  return {
    fork: {
      version: ANYHARNESS_FORK_EXTENSION_VERSION,
      anchor: "upToMessageId",
    },
    rewindFiles: {
      version: ANYHARNESS_FORK_EXTENSION_VERSION,
      controlMethod: REWIND_FILES_METHOD,
      scope: "writeEditNotebookEdit",
      requiresCheckpointingOptIn: true,
    },
  };
}

function metaNamespace(meta: unknown): Record<string, unknown> | undefined {
  if (!meta || typeof meta !== "object") return undefined;
  const ns = (meta as Record<string, unknown>)[ANYHARNESS_META_NAMESPACE];
  if (!ns || typeof ns !== "object") return undefined;
  return ns as Record<string, unknown>;
}

/** Read the inclusive fork anchor from a `session/fork` request `_meta`.
 *
 *  - Absent anchor  → `undefined` (an unanchored tip fork, canonical behavior).
 *  - Present anchor → the non-empty string message id.
 *  - Present but non-string/empty → hard `invalidParams`. NEVER returns
 *    `undefined` for a malformed anchor: a requested anchor that fell through to
 *    a tip fork is the cardinal sin this guard exists to prevent. */
export function forkAnchorFromMeta(meta: unknown): string | undefined {
  const ns = metaNamespace(meta);
  if (!ns) return undefined;
  const anchor = ns.upToMessageId;
  if (anchor === undefined || anchor === null) return undefined;
  if (typeof anchor !== "string" || anchor.length === 0) {
    throw RequestError.invalidParams(
      undefined,
      "_meta.anyharness.upToMessageId must be a non-empty string when present",
    );
  }
  return anchor;
}

/** Whether the session opted into SDK file checkpointing (so the labeled
 *  `_anyharness/rewindFiles` fast path can function). Default false: opting in
 *  has a disk/perf cost, so a client that does not ask for it pays nothing and
 *  gets a truthful `canRewind:false` from rewind. */
export function fileCheckpointingFromMeta(meta: unknown): boolean {
  const ns = metaNamespace(meta);
  return ns?.enableFileCheckpointing === true;
}

export type RewindFilesRequest = {
  sessionId: string;
  /** The ACP message id (as exposed to clients) to rewind file state to. */
  upToMessageId: string;
  /** Preview the change set without mutating files. */
  dryRun?: boolean;
};

/** Mirror of the SDK `RewindFilesResult`, re-declared so the wire response
 *  shape is owned here rather than leaking an SDK type across the ACP boundary. */
export type RewindFilesResponse = {
  canRewind: boolean;
  error?: string;
  filesChanged?: string[];
  insertions?: number;
  deletions?: number;
  skippedLinks?: number;
};

export function parseRewindFilesRequest(params: unknown): RewindFilesRequest {
  if (!params || typeof params !== "object") {
    throw RequestError.invalidParams(undefined, "rewindFiles params must be an object");
  }
  const { sessionId, upToMessageId, dryRun } = params as Record<string, unknown>;
  if (typeof sessionId !== "string" || sessionId.length === 0) {
    throw RequestError.invalidParams(undefined, "rewindFiles params require a non-empty sessionId");
  }
  if (typeof upToMessageId !== "string" || upToMessageId.length === 0) {
    throw RequestError.invalidParams(
      undefined,
      "rewindFiles params require a non-empty upToMessageId",
    );
  }
  if (dryRun !== undefined && typeof dryRun !== "boolean") {
    throw RequestError.invalidParams(undefined, "rewindFiles dryRun must be a boolean when present");
  }
  return { sessionId, upToMessageId, dryRun: dryRun as boolean | undefined };
}
