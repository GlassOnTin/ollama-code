/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from 'fs';
import * as path from 'path';
import { homedir } from 'os';
import { MCPServerConfig } from '@tcsenpai/ollama-code';

export enum MCPServerScope {
  USER = 'user',
  PROJECT = 'project',
  LOCAL = 'local'
}

export interface MCPServerDefinition {
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  cwd?: string;
  url?: string;
  httpUrl?: string;
  headers?: Record<string, string>;
  tcp?: string;
  timeout?: number;
  trust?: boolean;
  description?: string;
  includeTools?: string[];
  excludeTools?: string[];
  // Transport type for Claude Code compatibility
  type?: 'stdio' | 'sse' | 'http';
}

export interface MCPUserConfig {
  mcpServers: Record<string, MCPServerDefinition>;
}

export const USER_MCP_CONFIG_DIR = path.join(homedir(), '.config', 'ollama-code');
export const USER_MCP_CONFIG_PATH = path.join(USER_MCP_CONFIG_DIR, 'mcp.json');

/**
 * Migrates legacy MCP server configurations from settings.json to user scope mcp.json
 */
export function migrateLegacyMcpConfig(legacyConfig: Record<string, MCPServerConfig>): void {
  const userConfig = loadUserMcpConfig();
  
  for (const [serverName, config] of Object.entries(legacyConfig)) {
    // Skip if already exists in user config
    if (userConfig.mcpServers[serverName]) {
      continue;
    }
    
    // Convert MCPServerConfig to MCPServerDefinition
    userConfig.mcpServers[serverName] = {
      command: config.command,
      args: config.args,
      env: config.env,
      cwd: config.cwd,
      url: config.url,
      httpUrl: config.httpUrl,
      headers: config.headers,
      tcp: config.tcp,
      timeout: config.timeout,
      trust: config.trust,
      description: config.description,
      includeTools: config.includeTools,
      excludeTools: config.excludeTools,
      // Determine transport type based on configuration
      type: determineTransportType(config)
    };
  }
  
  saveUserMcpConfig(userConfig);
}

function determineTransportType(config: MCPServerConfig): 'stdio' | 'sse' | 'http' {
  if (config.command) return 'stdio';
  if (config.url) return 'sse';
  if (config.httpUrl) return 'http';
  return 'stdio'; // default
}

/**
 * Loads user scope MCP configuration from ~/.config/ollama-code/mcp.json
 */
export function loadUserMcpConfig(): MCPUserConfig {
  try {
    if (fs.existsSync(USER_MCP_CONFIG_PATH)) {
      const content = fs.readFileSync(USER_MCP_CONFIG_PATH, 'utf-8');
      const config = JSON.parse(content) as MCPUserConfig;
      
      // Ensure mcpServers object exists
      if (!config.mcpServers) {
        config.mcpServers = {};
      }
      
      return config;
    }
  } catch (error) {
    console.warn('Failed to load user MCP config:', error);
  }
  
  return { mcpServers: {} };
}

/**
 * Saves user scope MCP configuration to ~/.config/ollama-code/mcp.json
 */
export function saveUserMcpConfig(config: MCPUserConfig): void {
  try {
    // Ensure directory exists
    if (!fs.existsSync(USER_MCP_CONFIG_DIR)) {
      fs.mkdirSync(USER_MCP_CONFIG_DIR, { recursive: true });
    }
    
    fs.writeFileSync(
      USER_MCP_CONFIG_PATH,
      JSON.stringify(config, null, 2),
      'utf-8'
    );
  } catch (error) {
    console.error('Failed to save user MCP config:', error);
    throw new Error(`Failed to save user MCP configuration: ${error}`);
  }
}

/**
 * Adds a user-scoped MCP server
 */
export function addUserMcpServer(
  serverName: string,
  serverConfig: MCPServerDefinition
): void {
  const config = loadUserMcpConfig();
  config.mcpServers[serverName] = serverConfig;
  saveUserMcpConfig(config);
}

/**
 * Removes a user-scoped MCP server
 */
export function removeUserMcpServer(serverName: string): boolean {
  const config = loadUserMcpConfig();
  if (config.mcpServers[serverName]) {
    delete config.mcpServers[serverName];
    saveUserMcpConfig(config);
    return true;
  }
  return false;
}

/**
 * Lists all user-scoped MCP servers
 */
export function listUserMcpServers(): Record<string, MCPServerDefinition> {
  const config = loadUserMcpConfig();
  return config.mcpServers;
}

/**
 * Converts user MCP config to MCPServerConfig format for compatibility
 */
export function convertToMCPServerConfig(
  definition: MCPServerDefinition
): MCPServerConfig {
  return new MCPServerConfig(
    definition.command,
    definition.args,
    definition.env,
    definition.cwd,
    definition.url,
    definition.httpUrl,
    definition.headers,
    definition.tcp,
    definition.timeout,
    definition.trust,
    definition.description,
    definition.includeTools,
    definition.excludeTools
  );}

/**
 * Gets all MCP servers from user scope, converted to MCPServerConfig format
 */
export function getUserMcpServersAsConfig(): Record<string, MCPServerConfig> {
  const userConfig = loadUserMcpConfig();
  const result: Record<string, MCPServerConfig> = {};
  
  for (const [serverName, definition] of Object.entries(userConfig.mcpServers)) {
    result[serverName] = convertToMCPServerConfig(definition);
  }
  
  return result;
}

/**
 * Environment variable expansion for MCP configuration
 */
export function expandEnvVarsInConfig(
  config: MCPUserConfig
): MCPUserConfig {
  const expanded = JSON.parse(JSON.stringify(config)); // Deep clone
  
  for (const [serverName, definition] of Object.entries(expanded.mcpServers)) {
    const def = definition as MCPServerDefinition;
    // Expand environment variables in command
    if (def.command) {
      def.command = expandEnvVars(def.command);
    }
    
    // Expand environment variables in args
    if (def.args) {
      def.args = def.args.map((arg: string) => expandEnvVars(arg));
    }
    
    // Expand environment variables in env values
    if (def.env) {
      for (const [key, value] of Object.entries(def.env)) {
        def.env[key] = expandEnvVars(value);
      }
    }
    
    // Expand environment variables in URLs
    if (def.url) {
      def.url = expandEnvVars(def.url);
    }
    
    if (def.httpUrl) {
      def.httpUrl = expandEnvVars(def.httpUrl);
    }
    
    // Expand environment variables in headers
    if (def.headers) {
      for (const [key, value] of Object.entries(def.headers)) {
        def.headers[key] = expandEnvVars(value);
      }
    }
    
    // Expand environment variables in cwd
    if (def.cwd) {
      def.cwd = expandEnvVars(def.cwd);
    }
  }
  
  return expanded;
}

function expandEnvVars(value: string): string {
  const envVarRegex = /\$(?:(\w+)|{([^}]+)})/g; // Find $VAR_NAME or ${VAR_NAME}
  return value.replace(envVarRegex, (match, varName1, varName2) => {
    const varName = varName1 || varName2;
    if (process && process.env && typeof process.env[varName] === 'string') {
      return process.env[varName]!;
    }
    return match;
  });
}