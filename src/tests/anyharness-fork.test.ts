import { describe, it, expect } from "vitest";
import { RequestError } from "@agentclientprotocol/sdk";
import {
  ANYHARNESS_FORK_EXTENSION_VERSION,
  REWIND_FILES_METHOD,
  anyharnessCapabilities,
  fileCheckpointingFromMeta,
  forkAnchorFromMeta,
  parseRewindFilesRequest,
} from "../anyharness-fork.js";

describe("forkAnchorFromMeta", () => {
  it("returns undefined for absent meta / namespace / anchor (unanchored tip fork)", () => {
    expect(forkAnchorFromMeta(undefined)).toBeUndefined();
    expect(forkAnchorFromMeta(null)).toBeUndefined();
    expect(forkAnchorFromMeta({})).toBeUndefined();
    expect(forkAnchorFromMeta({ anyharness: {} })).toBeUndefined();
    expect(forkAnchorFromMeta({ anyharness: { upToMessageId: null } })).toBeUndefined();
  });

  it("returns the anchor string when present and well-formed", () => {
    expect(forkAnchorFromMeta({ anyharness: { upToMessageId: "msg_123" } })).toBe("msg_123");
  });

  // CARDINAL SIN regression pin (ADR §5): a present-but-malformed anchor must
  // be a hard error, NEVER silently dropped to undefined (which would become a
  // tip fork downstream).
  it("throws invalidParams on a present-but-empty anchor (never a silent tip fork)", () => {
    expect(() => forkAnchorFromMeta({ anyharness: { upToMessageId: "" } })).toThrow(RequestError);
  });

  it("throws invalidParams on a present-but-non-string anchor", () => {
    expect(() => forkAnchorFromMeta({ anyharness: { upToMessageId: 42 } })).toThrow(RequestError);
  });
});

describe("fileCheckpointingFromMeta", () => {
  it("defaults to false unless explicitly opted in", () => {
    expect(fileCheckpointingFromMeta(undefined)).toBe(false);
    expect(fileCheckpointingFromMeta({ anyharness: {} })).toBe(false);
    expect(fileCheckpointingFromMeta({ anyharness: { enableFileCheckpointing: "yes" } })).toBe(false);
  });

  it("is true only for the exact boolean true", () => {
    expect(fileCheckpointingFromMeta({ anyharness: { enableFileCheckpointing: true } })).toBe(true);
  });
});

describe("parseRewindFilesRequest", () => {
  it("accepts a well-formed request", () => {
    expect(parseRewindFilesRequest({ sessionId: "s1", upToMessageId: "msg_1", dryRun: true })).toEqual({
      sessionId: "s1",
      upToMessageId: "msg_1",
      dryRun: true,
    });
  });

  it("rejects missing/empty sessionId and upToMessageId, and non-boolean dryRun", () => {
    expect(() => parseRewindFilesRequest(null)).toThrow(RequestError);
    expect(() => parseRewindFilesRequest({ upToMessageId: "m" })).toThrow(RequestError);
    expect(() => parseRewindFilesRequest({ sessionId: "s", upToMessageId: "" })).toThrow(RequestError);
    expect(() =>
      parseRewindFilesRequest({ sessionId: "s", upToMessageId: "m", dryRun: "no" }),
    ).toThrow(RequestError);
  });
});

describe("anyharnessCapabilities", () => {
  it("advertises the inclusive fork anchor and the labeled partial rewind scope", () => {
    const caps = anyharnessCapabilities();
    expect(caps.fork).toEqual({
      version: ANYHARNESS_FORK_EXTENSION_VERSION,
      anchor: "upToMessageId",
    });
    expect(caps.rewindFiles.controlMethod).toBe(REWIND_FILES_METHOD);
    expect(caps.rewindFiles.scope).toBe("writeEditNotebookEdit");
    expect(caps.rewindFiles.requiresCheckpointingOptIn).toBe(true);
  });
});
