// Export the main agent class and utilities for library usage
export {
  ClaudeAcpAgent,
  isLocalCommandMetadata,
  stripLocalCommandMetadata,
  runAcp,
  toAcpNotifications,
  streamEventToAcpNotifications,
  type ToolUpdateMeta,
  type NewSessionMeta,
  type SDKMessageFilter,
} from "./acp-agent.js";
export { nodeToWebReadable, nodeToWebWritable, Pushable, unreachable } from "./utils.js";
export {
  toolInfoFromToolUse,
  toDisplayPath,
  planEntries,
  toolUpdateFromToolResult,
} from "./tools.js";
export { SettingsManager, type SettingsManagerOptions } from "./settings.js";

// Export types
export type { ClaudePlanEntry } from "./tools.js";

// anyharness GoalPort/LoopPort extension (wire contract v1)
export {
  ANYHARNESS_CAPABILITIES,
  ANYHARNESS_SCHEMA_VERSION,
  type AnyharnessTranscriptEvent,
  type GoalWire,
  type LoopSchedule,
  type LoopWire,
} from "./anyharness.js";
