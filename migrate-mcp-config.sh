#!/bin/bash

# MCP Configuration Migration Script
# Migrates user scope MCP configuration from Claude Code to ollama-code

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🔧 MCP Configuration Migration Tool"
echo "===================================="

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is required but not installed.${NC}"
    echo "Please install jq: https://stedolan.github.io/jq/download/"
    exit 1
fi

# Define file paths
CLAUDE_MCP_CONFIG="$HOME/.config/claude/mcp.json"
OLLAMA_SETTINGS="$HOME/.ollama/settings.json"
BACKUP_FILE="$HOME/.ollama/settings.json.backup.$(date +%Y%m%d_%H%M%S)"

# Function to check if file exists
check_file() {
    if [ -f "$1" ]; then
        return 0
    else
        return 1
    fi
}

# Function to validate JSON
validate_json() {
    if jq empty "$1" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

echo -e "\n📋 Checking configuration files..."

# Check if Claude Code MCP config exists
if check_file "$CLAUDE_MCP_CONFIG"; then
    echo -e "${GREEN}✓${NC} Found Claude Code MCP config: $CLAUDE_MCP_CONFIG"
    
    # Validate Claude config
    if validate_json "$CLAUDE_MCP_CONFIG"; then
        echo -e "${GREEN}✓${NC} Claude Code MCP config is valid JSON"
    else
        echo -e "${RED}✗${NC} Claude Code MCP config is invalid JSON"
        exit 1
    fi
else
    echo -e "${RED}✗${NC} Claude Code MCP config not found: $CLAUDE_MCP_CONFIG"
    echo "Please ensure Claude Code is installed and has MCP servers configured."
    exit 1
fi

# Check if ollama-code settings exist
if check_file "$OLLAMA_SETTINGS"; then
    echo -e "${GREEN}✓${NC} Found ollama-code settings: $OLLAMA_SETTINGS"
    
    # Validate ollama settings
    if validate_json "$OLLAMA_SETTINGS"; then
        echo -e "${GREEN}✓${NC} ollama-code settings are valid JSON"
    else
        echo -e "${RED}✗${NC} ollama-code settings are invalid JSON"
        echo "Please fix your ollama-code settings file first."
        exit 1
    fi
else
    echo -e "${YELLOW}!${NC} ollama-code settings not found: $OLLAMA_SETTINGS"
    echo "Creating new settings file..."
    mkdir -p "$(dirname "$OLLAMA_SETTINGS")"
    echo "{}" > "$OLLAMA_SETTINGS"
fi

echo -e "\n💾 Creating backup..."
cp "$OLLAMA_SETTINGS" "$BACKUP_FILE"
echo -e "${GREEN}✓${NC} Backup created: $BACKUP_FILE"

echo -e "\n🔄 Migrating MCP configuration..."

# Extract MCP servers from Claude config
MCP_SERVERS_FOUND=false

# Check for both "mcpServers" and "servers" formats
if jq -e '.mcpServers' "$CLAUDE_MCP_CONFIG" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Found MCP servers in Claude Code configuration (mcpServers format)"
    MCP_SERVERS_FOUND=true
    MCP_KEY="mcpServers"
elif jq -e '.servers' "$CLAUDE_MCP_CONFIG" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Found MCP servers in Claude Code configuration (servers format)"
    MCP_SERVERS_FOUND=true
    MCP_KEY="servers"
fi

if [ "$MCP_SERVERS_FOUND" = true ]; then
    # Create temporary files for merging
    TEMP_CLAUDE=$(mktemp)
    TEMP_OLLAMA=$(mktemp)
    TEMP_MERGED=$(mktemp)
    
    # Clean up temp files on exit
    trap "rm -f $TEMP_CLAUDE $TEMP_OLLAMA $TEMP_MERGED" EXIT
    
    # Prepare Claude config (only mcpServers section, normalized to mcpServers key)
    jq --arg key "$MCP_KEY" '{mcpServers: .[$key]}' "$CLAUDE_MCP_CONFIG" > "$TEMP_CLAUDE"
    
    # Prepare ollama config (remove any existing mcpServers to avoid conflicts)
    jq 'del(.mcpServers)' "$OLLAMA_SETTINGS" > "$TEMP_OLLAMA"
    
    # Merge configurations
    jq -s '.[0] * .[1]' "$TEMP_OLLAMA" "$TEMP_CLAUDE" > "$TEMP_MERGED"
    
    # Show what will be migrated
echo -e "\n📊 MCP Servers to be migrated:"
    jq -r '.mcpServers | keys[]' "$TEMP_CLAUDE" | while read -r server; do
        echo -e "  ${GREEN}•${NC} $server"
    done
    
    # Apply the merged configuration
    mv "$TEMP_MERGED" "$OLLAMA_SETTINGS"
    
    echo -e "\n${GREEN}✓${NC} MCP configuration migrated successfully!"
    
else
    echo -e "${YELLOW}!${NC} No MCP servers found in Claude Code configuration"
    echo "Nothing to migrate."
    exit 0
fi

echo -e "\n🔍 Verifying migration..."

# Verify the migration
if validate_json "$OLLAMA_SETTINGS"; then
    echo -e "${GREEN}✓${NC} ollama-code settings are valid JSON"
    
    # Show migrated MCP servers
    if jq -e '.mcpServers' "$OLLAMA_SETTINGS" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} MCP servers successfully added to ollama-code settings"
        echo -e "\n📋 Migrated MCP Servers:"
        jq -r '.mcpServers | keys[]' "$OLLAMA_SETTINGS" | while read -r server; do
            echo -e "  ${GREEN}•${NC} $server"
        done
    else
        echo -e "${RED}✗${NC} MCP servers not found in migrated configuration"
        exit 1
    fi
else
    echo -e "${RED}✗${NC} Migration resulted in invalid JSON"
    echo "Restoring backup..."
    mv "$BACKUP_FILE" "$OLLAMA_SETTINGS"
    exit 1
fi

echo -e "\n🎉 Migration completed successfully!"
echo -e "\n📋 Next steps:"
echo -e "  1. Start ollama-code"
echo -e "  2. Use the ${YELLOW}/mcp${NC} command to verify your MCP servers are loaded"
echo -e "  3. Test your MCP tools to ensure they work correctly"
echo -e "\n💡 Backup saved at: $BACKUP_FILE"
echo -e "   You can restore it if needed with: cp $BACKUP_FILE $OLLAMA_SETTINGS"