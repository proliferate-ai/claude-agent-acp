// Export the main agent class and utilities for library usage
export {
  ClaudeAcpAgent,
  runAcp,
  toAcpNotifications,
  streamEventToAcpNotifications,
  type ToolUpdateMeta,
  type NewSessionMeta,
} from "./acp-agent.js";
export {
  loadManagedSettings,
  applyEnvironmentSettings,
  nodeToWebReadable,
  nodeToWebWritable,
  Pushable,
  unreachable,
} from "./utils.js";
export {
  toolInfoFromToolUse,
  toDisplayPath,
  planEntries,
  toolUpdateFromToolResult,
} from "./tools.js";
export {
  SettingsManager,
  type ClaudeCodeSettings,
  type SettingsManagerOptions,
} from "./settings.js";

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
