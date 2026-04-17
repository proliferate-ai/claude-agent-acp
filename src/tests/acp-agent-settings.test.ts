import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";
import type { ClaudeAcpAgent as ClaudeAcpAgentType } from "../acp-agent.js";

const { querySpy } = vi.hoisted(() => ({
  querySpy: vi.fn(),
}));

vi.mock("@anthropic-ai/claude-agent-sdk", async () => {
  const actual = await vi.importActual<any>("@anthropic-ai/claude-agent-sdk");
  return {
    ...actual,
    query: querySpy,
  };
});

describe("ClaudeAcpAgent settings", () => {
  let tempDir: string;
  let originalClaudeConfigDir: string | undefined;

  function createMockClient(): AgentSideConnection {
    return {
      sessionUpdate: async () => {},
      requestPermission: async () => ({ outcome: { outcome: "cancelled" } }),
      readTextFile: async () => ({ content: "" }),
      writeTextFile: async () => ({}),
    } as unknown as AgentSideConnection;
  }

  function mockQuery(params?: {
    models?: any[];
    initialSettings?: Record<string, unknown>;
    settingsResponseShape?: "flat" | "effective";
    fastModeState?: "off" | "cooldown" | "on";
  }) {
    let capturedOptions: any;
    let currentSettings = { ...(params?.initialSettings ?? {}) };
    const settingsResponseShape = params?.settingsResponseShape ?? "flat";
    const setModelSpy = vi.fn();
    const applyFlagSettingsSpy = vi.fn(async (settingsPatch: Record<string, unknown>) => {
      currentSettings = {
        ...currentSettings,
        ...settingsPatch,
      };
    });
    const getSettingsSpy = vi.fn(async () =>
      settingsResponseShape === "effective"
        ? { effective: { ...currentSettings } }
        : { ...currentSettings },
    );
    querySpy.mockImplementation(({ options }: any) => {
      capturedOptions = options;
      return {
        initializationResult: async () => ({
          models: params?.models ?? [
            {
              value: "claude-sonnet-4-5",
              displayName: "Claude Sonnet 4.5",
              description: "Default",
            },
          ],
          ...(params?.fastModeState !== undefined
            ? { fast_mode_state: params.fastModeState }
            : {}),
        }),
        setModel: setModelSpy,
        applyFlagSettings: applyFlagSettingsSpy,
        getSettings: getSettingsSpy,
        supportedCommands: async () => [],
      } as any;
    });
    return {
      getCapturedOptions: () => capturedOptions,
      setModelSpy,
      applyFlagSettingsSpy,
      getSettingsSpy,
      getCurrentSettings: () => ({ ...currentSettings }),
    };
  }

  beforeEach(async () => {
    tempDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), "acp-agent-settings-"));
    originalClaudeConfigDir = process.env.CLAUDE_CONFIG_DIR;
    process.env.CLAUDE_CONFIG_DIR = tempDir;
    querySpy.mockReset();
    vi.resetModules();
  });

  afterEach(async () => {
    if (originalClaudeConfigDir) {
      process.env.CLAUDE_CONFIG_DIR = originalClaudeConfigDir;
    } else {
      delete process.env.CLAUDE_CONFIG_DIR;
    }
    await fs.promises.rm(tempDir, { recursive: true, force: true });
  });

  it("uses permissions.defaultMode for new sessions", async () => {
    await fs.promises.writeFile(
      path.join(tempDir, "settings.json"),
      JSON.stringify({
        permissions: {
          defaultMode: "dontAsk",
        },
      }),
    );

    const projectDir = path.join(tempDir, "project");
    await fs.promises.mkdir(projectDir, { recursive: true });

    const { getCapturedOptions } = mockQuery();

    const { ClaudeAcpAgent } = await import("../acp-agent.js");
    const agent: ClaudeAcpAgentType = new ClaudeAcpAgent(createMockClient());

    const response = await (agent as any).createSession({
      cwd: projectDir,
      mcpServers: [],
      _meta: { disableBuiltInTools: true },
    });

    expect(getCapturedOptions().permissionMode).toBe("dontAsk");
    expect(getCapturedOptions().settingSources).toEqual(["user", "project", "local"]);
    expect(response.modes.currentModeId).toBe("dontAsk");
  });

  it("supports acceptEdits mode defaults", async () => {
    await fs.promises.writeFile(
      path.join(tempDir, "settings.json"),
      JSON.stringify({
        permissions: {
          defaultMode: "acceptEdits",
        },
      }),
    );

    const projectDir = path.join(tempDir, "project");
    await fs.promises.mkdir(projectDir, { recursive: true });

    const { getCapturedOptions } = mockQuery();

    const { ClaudeAcpAgent } = await import("../acp-agent.js");
    const agent: ClaudeAcpAgentType = new ClaudeAcpAgent(createMockClient());

    const response = await (agent as any).createSession({
      cwd: projectDir,
      mcpServers: [],
      _meta: { disableBuiltInTools: true },
    });

    expect(getCapturedOptions().permissionMode).toBe("acceptEdits");
    expect(response.modes.currentModeId).toBe("acceptEdits");
  });

  it("defaults to 'default' when no permissions.defaultMode is set", async () => {
    const projectDir = path.join(tempDir, "project");
    await fs.promises.mkdir(projectDir, { recursive: true });

    const { getCapturedOptions } = mockQuery();

    const { ClaudeAcpAgent } = await import("../acp-agent.js");
    const agent: ClaudeAcpAgentType = new ClaudeAcpAgent(createMockClient());

    const response = await (agent as any).createSession({
      cwd: projectDir,
      mcpServers: [],
      _meta: { disableBuiltInTools: true },
    });

    expect(getCapturedOptions().permissionMode).toBe("default");
    expect(response.modes.currentModeId).toBe("default");
  });

  it("throws when permissions.defaultMode is not a string", async () => {
    await fs.promises.writeFile(
      path.join(tempDir, "settings.json"),
      JSON.stringify({
        permissions: {
          defaultMode: 123,
        },
      }),
    );

    const projectDir = path.join(tempDir, "project");
    await fs.promises.mkdir(projectDir, { recursive: true });

    mockQuery();

    const { ClaudeAcpAgent } = await import("../acp-agent.js");
    const agent: ClaudeAcpAgentType = new ClaudeAcpAgent(createMockClient());

    await expect(
      (agent as any).createSession({
        cwd: projectDir,
        mcpServers: [],
        _meta: { disableBuiltInTools: true },
      }),
    ).rejects.toThrow("Invalid permissions.defaultMode");
  });

  it("resolves model aliases like opus[1m] to the correct model", async () => {
    await fs.promises.writeFile(
      path.join(tempDir, "settings.json"),
      JSON.stringify({
        model: "opus[1m]",
      }),
    );

    const projectDir = path.join(tempDir, "project");
    await fs.promises.mkdir(projectDir, { recursive: true });

    const setModelSpy = vi.fn();
    querySpy.mockImplementation(({ options: _options }: any) => {
      return {
        initializationResult: async () => ({
          models: [
            {
              value: "claude-opus-4-6",
              displayName: "Claude Opus 4.6",
              description: "Base",
            },
            {
              value: "claude-opus-4-6-1m",
              displayName: "Claude Opus 4.6 (1M)",
              description: "Long context",
            },
          ],
        }),
        setModel: setModelSpy,
        supportedCommands: async () => [],
      } as any;
    });

    const { ClaudeAcpAgent } = await import("../acp-agent.js");
    const agent: ClaudeAcpAgentType = new ClaudeAcpAgent(createMockClient());

    const response = await (agent as any).createSession({
      cwd: projectDir,
      mcpServers: [],
      _meta: { disableBuiltInTools: true },
    });

    expect(setModelSpy).toHaveBeenCalledWith("claude-opus-4-6-1m");
    expect(response.models.currentModelId).toBe("claude-opus-4-6-1m");
  });

  it("includes effort and fast mode config options and normalizes thinking on", async () => {
    const projectDir = path.join(tempDir, "project");
    await fs.promises.mkdir(projectDir, { recursive: true });

    const { applyFlagSettingsSpy, getCurrentSettings } = mockQuery({
      models: [
        {
          value: "claude-sonnet-4-5",
          displayName: "Claude Sonnet 4.5",
          description: "Default",
          supportsEffort: true,
          supportedEffortLevels: ["low", "medium", "high", "xhigh", "max"],
          supportsAdaptiveThinking: true,
          supportsFastMode: true,
        },
      ],
      initialSettings: {
        alwaysThinkingEnabled: false,
        effortLevel: "medium",
        fastMode: true,
      },
    });

    const { ClaudeAcpAgent } = await import("../acp-agent.js");
    const agent: ClaudeAcpAgentType = new ClaudeAcpAgent(createMockClient());

    const response = await (agent as any).createSession({
      cwd: projectDir,
      mcpServers: [],
      _meta: { disableBuiltInTools: true },
    });

    expect(response.configOptions.map((option: any) => option.id)).toEqual([
      "mode",
      "model",
      "effort",
      "fast_mode",
    ]);
    expect(response.configOptions.find((option: any) => option.id === "effort")?.options).toEqual([
      { value: "low", name: "Low" },
      { value: "medium", name: "Medium" },
      { value: "high", name: "High" },
      { value: "xhigh", name: "X High" },
    ]);
    expect(response.configOptions.find((option: any) => option.id === "effort")?.currentValue).toBe(
      "medium",
    );
    expect(
      response.configOptions.find((option: any) => option.id === "fast_mode")?.currentValue,
    ).toBe("on");
    expect(applyFlagSettingsSpy).toHaveBeenCalledWith({
      alwaysThinkingEnabled: true,
    });
    expect(getCurrentSettings().alwaysThinkingEnabled).toBe(true);
  });

  it("includes fast mode when the SDK advertises session-level fast mode state", async () => {
    const projectDir = path.join(tempDir, "project");
    await fs.promises.mkdir(projectDir, { recursive: true });

    mockQuery({
      models: [
        {
          value: "claude-sonnet-4-5",
          displayName: "Claude Sonnet 4.5",
          description: "Default",
          supportsEffort: true,
          supportedEffortLevels: ["low", "medium", "high"],
          supportsAdaptiveThinking: true,
          supportsFastMode: false,
        },
      ],
      initialSettings: {
        alwaysThinkingEnabled: true,
        effortLevel: "medium",
      },
      fastModeState: "off",
    });

    const { ClaudeAcpAgent } = await import("../acp-agent.js");
    const agent: ClaudeAcpAgentType = new ClaudeAcpAgent(createMockClient());

    const response = await (agent as any).createSession({
      cwd: projectDir,
      mcpServers: [],
      _meta: { disableBuiltInTools: true },
    });

    expect(response.configOptions.map((option: any) => option.id)).toEqual([
      "mode",
      "model",
      "effort",
      "fast_mode",
    ]);
    expect(
      response.configOptions.find((option: any) => option.id === "fast_mode")?.currentValue,
    ).toBe("off");
  });

  it("uses enabled session-level fast mode state as the current fast mode value", async () => {
    const projectDir = path.join(tempDir, "project");
    await fs.promises.mkdir(projectDir, { recursive: true });

    mockQuery({
      models: [
        {
          value: "claude-sonnet-4-5",
          displayName: "Claude Sonnet 4.5",
          description: "Default",
          supportsEffort: true,
          supportedEffortLevels: ["low", "medium", "high"],
          supportsAdaptiveThinking: true,
          supportsFastMode: false,
        },
      ],
      initialSettings: {
        alwaysThinkingEnabled: true,
        effortLevel: "medium",
      },
      fastModeState: "on",
    });

    const { ClaudeAcpAgent } = await import("../acp-agent.js");
    const agent: ClaudeAcpAgentType = new ClaudeAcpAgent(createMockClient());

    const response = await (agent as any).createSession({
      cwd: projectDir,
      mcpServers: [],
      _meta: { disableBuiltInTools: true },
    });

    expect(
      response.configOptions.find((option: any) => option.id === "fast_mode")?.currentValue,
    ).toBe("on");
  });

  it("reads effort and fast mode from wrapped effective settings responses", async () => {
    const projectDir = path.join(tempDir, "project");
    await fs.promises.mkdir(projectDir, { recursive: true });

    const { applyFlagSettingsSpy } = mockQuery({
      models: [
        {
          value: "claude-sonnet-4-5",
          displayName: "Claude Sonnet 4.5",
          description: "Default",
          supportsEffort: true,
          supportedEffortLevels: ["low", "medium", "high"],
          supportsAdaptiveThinking: true,
          supportsFastMode: true,
        },
      ],
      initialSettings: {
        alwaysThinkingEnabled: false,
        effortLevel: "medium",
        fastMode: true,
      },
      settingsResponseShape: "effective",
    });

    const { ClaudeAcpAgent } = await import("../acp-agent.js");
    const agent: ClaudeAcpAgentType = new ClaudeAcpAgent(createMockClient());

    const response = await (agent as any).createSession({
      cwd: projectDir,
      mcpServers: [],
      _meta: { disableBuiltInTools: true },
    });

    expect(response.configOptions.find((option: any) => option.id === "effort")?.currentValue).toBe(
      "medium",
    );
    expect(
      response.configOptions.find((option: any) => option.id === "fast_mode")?.currentValue,
    ).toBe("on");
    expect(applyFlagSettingsSpy).toHaveBeenCalledWith({
      alwaysThinkingEnabled: true,
    });
  });
});
