# MCP Configuration Migration Guide

## Overview
This guide helps you migrate your existing Claude Code user scope MCP configuration to ollama-code.

## Claude Code User MCP Configuration Location
- **File**: `~/.config/claude/mcp.json`
- **Format**: JSON with `mcpServers` object

## Ollama Code User MCP Configuration Location
- **File**: `~/.ollama/settings.json`
- **Format**: JSON with `mcpServers` object (same structure)

## Migration Steps

### 1. Check Your Claude Code MCP Configuration
First, view your current Claude Code MCP configuration:
```bash
cat ~/.config/claude/mcp.json
```

### 2. Backup Your Ollama Code Settings
Before making changes, backup your current ollama-code settings:
```bash
cp ~/.ollama/settings.json ~/.ollama/settings.json.backup
```

### 3. Migrate the Configuration
The MCP server configuration structure is compatible between Claude Code and ollama-code. You just need to copy the `mcpServers` section from your Claude Code configuration to your ollama-code settings.

#### Example Claude Code MCP Configuration:
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@github/github-mcp-server"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token-here"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
    }
  }
}
```

#### Example Ollama Code Settings After Migration:
```json
{
  "theme": "Ollama Dark",
  "ollama": {
    "model": "kimi-k2:1t-cloud"
  },
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@github/github-mcp-server"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token-here"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
    }
  }
}
```

### 4. Manual Migration Command
You can use this command to automatically migrate your MCP configuration:

```bash
# Create a temporary file with the merged configuration
jq -s '.[0] * .[1]' ~/.ollama/settings.json ~/.config/claude/mcp.json > /tmp/merged_settings.json

# Review the merged configuration
cat /tmp/merged_settings.json

# If it looks correct, replace your ollama-code settings
mv /tmp/merged_settings.json ~/.ollama/settings.json
```

### 5. Verify the Migration
Check that your ollama-code settings now include the MCP servers:
```bash
cat ~/.ollama/settings.json
```

### 6. Test the Configuration
Start ollama-code and use the `/mcp` command to verify your MCP servers are loaded:
```bash
# In ollama-code
/mcp
```

## Configuration Format Reference

Both Claude Code and ollama-code use the same MCP server configuration format:

```json
{
  "mcpServers": {
    "server-name": {
      "command": "command-to-execute",
      "args": ["arg1", "arg2"],
      "env": {
        "ENV_VAR": "value"
      },
      "cwd": "/working/directory",
      "timeout": 30000,
      "trust": false
    }
  }
}
```

### Supported Transport Types
- **stdio**: Local process-based servers (default)
- **sse**: Server-Sent Events for remote access
- **http**: HTTP-based servers

### Environment Variables
Both systems support environment variable expansion in the configuration using `${VAR_NAME}` syntax.

## Troubleshooting

### MCP Servers Not Loading
1. Check the configuration syntax with: `jq . ~/.ollama/settings.json`
2. Verify the MCP server commands are executable
3. Check ollama-code logs for MCP-related errors

### Authentication Issues
- Ensure API keys and tokens are correctly set in the `env` section
- Check that environment variables are properly expanded

### Port Conflicts
- MCP servers will automatically find available ports
- Check the `/mcp` command output for server status

## Additional Notes
- The migration preserves all your existing MCP server configurations
- You can continue using the same servers you had in Claude Code
- The configuration structure is 100% compatible between both systems