/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * This module provides utilities for converting between Google's Gemini API format
 * and OpenAI's chat completion format. It handles the translation of messages,
 * tools, and responses between the different API schemas.
 * 
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  GenerateContentResponse,
  GenerateContentParameters,
  FinishReason,
  Part,
  ToolListUnion,
  FunctionCall,
  FunctionResponse,
} from '@google/genai';
import type {
  ChatCompletion,
  ChatCompletionChunk,
} from 'openai/resources/chat/index.js';

// OpenAI API type definitions for conversions
export interface OpenAIToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

export interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
}

export interface OpenAIRequestFormat {
  model: string;
  messages: OpenAIMessage[];
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  tools?: unknown[];
}

/**
 * Utility class for converting between OpenAI and Gemini API formats with robust error handling.
 * 
 * This class provides static methods for translating between Google's Gemini API format and
 * OpenAI's chat completion format. It includes advanced error handling for streaming responses,
 * malformed JSON parsing, and message consolidation to ensure reliable API integration.
 * 
 * The main features include:
 * - Safe JSON parsing with fallback mechanisms for malformed data
 * - Message consolidation for streaming responses
 * - Tool call validation and cleanup
 * - Type conversion between different AI API schemas
 * 
 * @example
 * ```typescript
 * // Convert Gemini request to OpenAI format
 * const geminiRequest = { contents: [{ role: "user", parts: [{ text: "Hello" }] }] };
 * const openaiRequest = OpenAIFormatConverter.convertGeminiParametersToOpenAI(geminiRequest, "gpt-4");
 * ```
 * 
 * @example
 * ```typescript
 * // Safely parse potentially malformed JSON
 * const malformed = '{"name": "test", "value": 123';
 * const result = OpenAIFormatConverter.safeJsonParse(malformed);
 * // Result: { name: "test", value: 123 }
 * ```
 */
export class OpenAIFormatConverter {
  /**
   * Convert Gemini tools to OpenAI format
   */
  /**
   * Converts a collection of Gemini tools to OpenAI format for API compatibility.
   * 
   * This method takes Gemini tool definitions and transforms them into the format
   * expected by OpenAI's chat completions API, specifically handling function declarations
   * which are the primary tool type supported by both platforms.
   * 
   * @param geminiTools - Array of Gemini tool definitions to convert
   * @returns Promise resolving to array of OpenAI-compatible tool definitions
   * 
   * @example
   * ```typescript
   * const geminiTools = [{ functionDeclarations: [{ name: "search", description: "Search for info" }] }];
   * const openaiTools = await OpenAIFormatConverter.convertGeminiToolsToOpenAI(geminiTools);
   * ```
   */
  static async convertGeminiToolsToOpenAI(
    geminiTools: ToolListUnion,
  ): Promise<unknown[]> {
    const openAITools: unknown[] = [];

    for (const toolUnion of geminiTools) {
      if ('functionDeclarations' in toolUnion) {
        for (const func of toolUnion.functionDeclarations || []) {
          const openAITool = {
            type: 'function',
            function: {
              name: func.name,
              description: func.description,
              parameters: func.parameters,
            },
          };
          openAITools.push(openAITool);
        }
      }
    }

    return openAITools;
  }

  /**
   * Convert Gemini parameters to OpenAI request format
   */
  static convertGeminiParametersToOpenAI(
    request: GenerateContentParameters,
    model: string,
  ): OpenAIRequestFormat {
    const messages: OpenAIMessage[] = [];

    // Convert Gemini contents to OpenAI messages
    const contents = Array.isArray(request.contents) ? request.contents : [request.contents];
    for (const content of contents) {
      // Type guard to ensure content has the expected properties
      if (typeof content === 'object' && content !== null && 'role' in content && 'parts' in content) {
        if (content.role === 'user' || content.role === 'model') {
          const role = content.role === 'model' ? 'assistant' : content.role;
        
          // Handle text parts
          const textParts = (content.parts || [])
            .filter((part): part is { text: string } => 'text' in part)
            .map((part) => part.text)
            .join('\n');

          if (textParts) {
            messages.push({
              role: role as 'user' | 'assistant',
              content: textParts,
            });
          }

          // Handle function calls and responses
          const functionCalls = (content.parts || []).filter(
            (part): part is { functionCall: FunctionCall } => 'functionCall' in part
          );

          const functionResponses = (content.parts || []).filter(
            (part): part is { functionResponse: FunctionResponse } => 'functionResponse' in part
          );

          if (functionCalls.length > 0) {
            const tool_calls = functionCalls.map((fc, index) => ({
              id: `call_${Date.now()}_${index}`,
              type: 'function' as const,
              function: {
                name: fc.functionCall?.name || 'unknown',
                arguments: JSON.stringify(fc.functionCall?.args || {}),
              },
            }));

            messages.push({
              role: 'assistant',
              content: null,
              tool_calls,
            });
          }

          if (functionResponses.length > 0) {
            for (const fr of functionResponses) {
              messages.push({
                role: 'tool',
                content: JSON.stringify(fr.functionResponse?.response || {}),
                tool_call_id: `call_${fr.functionResponse?.name || 'unknown'}`,
              });
            }
          }
        }
      }
    }

    const openAIRequest: OpenAIRequestFormat = {
      model,
      messages,
    };

    // Add configuration options
    if (request.config?.temperature !== undefined) {
      openAIRequest.temperature = request.config.temperature;
    }
    if (request.config?.topP !== undefined) {
      openAIRequest.top_p = request.config.topP;
    }
    if (request.config?.maxOutputTokens !== undefined) {
      openAIRequest.max_tokens = request.config.maxOutputTokens;
    }

    return openAIRequest;
  }

  /**
   * Convert OpenAI response to Gemini format
   */
  static convertToGeminiFormat(completion: ChatCompletion): GenerateContentResponse {
    const choice = completion.choices[0];
    if (!choice) {
      throw new Error('No choices in OpenAI response');
    }

    const parts: Part[] = [];

    // Handle text content
    if (choice.message.content) {
      parts.push({ text: choice.message.content });
    }

    // Handle tool calls
    if (choice.message.tool_calls) {
      for (const toolCall of choice.message.tool_calls) {
        if (toolCall.type === 'function') {
          parts.push({
            functionCall: {
              name: toolCall.function.name,
              args: this.safeJsonParse(toolCall.function.arguments || '{}'),
            },
          });
        }
      }
    }

    const finishReason = this.mapFinishReason(choice.finish_reason);

    return {
      candidates: [
        {
          content: {
            parts,
            role: 'model',
          },
          finishReason,
        },
      ],
      usageMetadata: completion.usage
        ? {
            promptTokenCount: completion.usage.prompt_tokens,
            candidatesTokenCount: completion.usage.completion_tokens,
            totalTokenCount: completion.usage.total_tokens,
          }
        : undefined,
    } as GenerateContentResponse;
  }

  /**
   * Convert OpenAI stream chunk to Gemini format
   */
  static convertStreamChunkToGeminiFormat(
    chunk: ChatCompletionChunk,
  ): GenerateContentResponse {
    const choice = chunk.choices[0];
    if (!choice) {
      return { candidates: [] } as unknown as GenerateContentResponse;
    }

    const parts: Part[] = [];

    // Handle text delta
    if (choice.delta.content) {
      parts.push({ text: choice.delta.content });
    }

    // Handle tool call deltas
    if (choice.delta.tool_calls) {
      for (const toolCall of choice.delta.tool_calls) {
        if (toolCall.type === 'function' && toolCall.function) {
          parts.push({
            functionCall: {
              name: toolCall.function.name || '',
              args: toolCall.function.arguments
                ? this.safeJsonParse(toolCall.function.arguments)
                : {},
            },
          });
        }
      }
    }

    const finishReason = choice.finish_reason
      ? this.mapFinishReason(choice.finish_reason)
      : undefined;

    return {
      candidates: [
        {
          content: {
            parts,
            role: 'model',
          },
          finishReason,
        },
      ],
    } as GenerateContentResponse;
  }

  /**
   * Map OpenAI finish reason to Gemini finish reason
   */
  private static mapFinishReason(openAIReason: string): FinishReason {
    switch (openAIReason) {
      case 'stop':
        return FinishReason.STOP;
      case 'length':
        return FinishReason.MAX_TOKENS;
      case 'tool_calls':
        return FinishReason.STOP;
      case 'content_filter':
        return FinishReason.SAFETY;
      default:
        return FinishReason.OTHER;
    }
  }

  /**
   * Clean orphaned tool calls from message history
   */
  static cleanOrphanedToolCalls(messages: OpenAIMessage[]): OpenAIMessage[] {
    const cleanedMessages: OpenAIMessage[] = [];
    const toolCallIds = new Set<string>();

    // First pass: collect all tool call IDs
    for (const message of messages) {
      if (message.role === 'assistant' && message.tool_calls) {
        for (const toolCall of message.tool_calls) {
          toolCallIds.add(toolCall.id);
        }
      }
    }

    // Second pass: filter messages and clean orphaned tool calls
    for (const message of messages) {
      if (message.role === 'tool') {
        // Only include tool messages that have corresponding tool calls
        if (message.tool_call_id && toolCallIds.has(message.tool_call_id)) {
          cleanedMessages.push(message);
        }
      } else if (message.role === 'assistant' && message.tool_calls) {
        // Filter out tool calls that don't have responses
        const validToolCalls = message.tool_calls.filter(toolCall =>
          messages.some(
            m => m.role === 'tool' && m.tool_call_id === toolCall.id
          )
        );

        cleanedMessages.push({
          ...message,
          tool_calls: validToolCalls.length > 0 ? validToolCalls : undefined,
        });
      } else {
        cleanedMessages.push(message);
      }
    }

    return cleanedMessages;
  }

  /**
   * Merge consecutive assistant messages
   */
  static mergeConsecutiveAssistantMessages(messages: OpenAIMessage[]): OpenAIMessage[] {
    const mergedMessages: OpenAIMessage[] = [];
    let currentAssistantMessage: OpenAIMessage | null = null;

    for (const message of messages) {
      if (message.role === 'assistant') {
        if (currentAssistantMessage) {
          // Merge with previous assistant message
          const mergedContent = [
            currentAssistantMessage.content,
            message.content,
          ]
            .filter(Boolean)
            .join('\n');

          currentAssistantMessage = {
            ...(currentAssistantMessage as OpenAIMessage),
            content: mergedContent || null,
            role: 'assistant' as const,
            tool_calls: [
              ...(currentAssistantMessage?.tool_calls || []),
              ...(message.tool_calls || []),
            ],
          };
        } else {
          currentAssistantMessage = { ...message };
        }
      } else {
        // Non-assistant message, push any pending assistant message
        if (currentAssistantMessage) {
          mergedMessages.push(currentAssistantMessage);
          currentAssistantMessage = null;
        }
        mergedMessages.push(message);
      }
    }

    // Don't forget the last assistant message
    if (currentAssistantMessage) {
      mergedMessages.push(currentAssistantMessage);
    }

    return mergedMessages;
  }

  /**
   * Safely parse JSON with fallback handling for malformed input
   */
  /**
   * Provides a safe interface for JSON parsing with graceful error handling.
   * 
   * This method serves as the main entry point for safe JSON parsing operations,
   * providing a fail-safe alternative to direct JSON.parse(). It automatically
   * falls back to partial JSON extraction when standard parsing fails.
   * 
   * This is particularly useful in contexts where:
   * - JSON data comes from external sources (APIs, streaming)
   * - Data may be incomplete or corrupted during transmission
   * - Robust error handling is required without throwing exceptions
   * 
   * The method internally uses {@link extractPartialJson} for fallback parsing,
   * which can handle various forms of malformed JSON including:
   * - Missing closing braces or brackets
   * - Trailing commas
   * - Incomplete string values
   * - Partial object properties
   * 
   * @param input - The JSON string to parse safely
   * @returns A parsed JavaScript object, or an empty object if parsing fails
   * 
   * @example
   * ```typescript
   * // Standard valid JSON
   * const valid = '{"status": "success", "count": 42}';
   * const result = this.safeJsonParse(valid);
   * console.log(result); // { status: "success", count: 42 }
   * ```
   * 
   * @example
   * ```typescript
   * // Malformed JSON that would normally throw
   * const malformed = '{"status": "success", "count": 42,}';
   * const result = this.safeJsonParse(malformed);
   * console.log(result); // { status: "success", count: 42 }
   * ```
   * 
   * @see {@link extractPartialJson} for the underlying partial parsing implementation
   */
  private static safeJsonParse(input: string): Record<string, unknown> {
    try {
      return JSON.parse(input);
    } catch {
      // If that fails, try to find valid JSON patterns
      return this.extractPartialJson(input);
    }
  }

  /**
   * Safely extracts valid JSON from potentially malformed or incomplete JSON strings.
   * 
   * This method is designed to handle cases where streaming chunks contain incomplete JSON,
   * such as during OpenAI API streaming responses where tool call arguments may be split
   * across multiple chunks. It employs a multi-stage parsing strategy to maximize success rate.
   * 
   * The parsing strategy includes:
   * 1. Direct JSON parsing for already valid JSON
   * 2. Character-by-character parsing to find valid object boundaries
   * 3. Manual key-value extraction for severely malformed cases
   * 4. Graceful fallback to empty object on complete failure
   * 
   * This is particularly useful when dealing with:
   * - Streaming API responses that may be interrupted
   * - Network issues that cause partial data transmission
   * - Malformed JSON from third-party services
   * - Valid JSON followed by extra text (e.g., "{\"key\":\"value\"} extra text")
   * 
   * @param input - The potentially malformed JSON string to parse
   * @returns A parsed JavaScript object, or an empty object if parsing completely fails
   * 
   * @example
   * ```typescript
   * // Handle streaming chunk with incomplete JSON
   * const chunk = '{"name": "test", "value": 123';
   * const result = this.extractPartialJson(chunk);
   * console.log(result); // { name: "test", value: 123 }
   * ```
   * 
   * @example
   * ```typescript
   * // Handle valid JSON followed by extra text
   * const chunk = '{"name": "test"} some extra text';
   * const result = this.extractPartialJson(chunk);
   * console.log(result); // { name: "test" }
   * ```
   * 
   * @example
   * ```typescript
   * // Handle completely malformed input gracefully
   * const bad = 'this is not json at all';
   * const result = this.extractPartialJson(bad);
   * console.log(result); // {}
   * ```
   * 
   * @see {@link safeJsonParse} for the public interface that calls this method
   */
  private static extractPartialJson(input: string): Record<string, unknown> {
    if (!input || typeof input !== 'string') {
      return {};
    }

    const trimmed = input.trim();

    // First try to parse the entire string
    try {
      return JSON.parse(trimmed);
    } catch {
      // If that fails, try to find valid JSON patterns
    }

    // Handle case where there's valid JSON followed by extra text
    // This is the specific case causing "Unexpected non-whitespace character after JSON"
    const braceStart = trimmed.indexOf('{');
    const braceEnd = trimmed.lastIndexOf('}');
    
    if (braceStart !== -1 && braceEnd !== -1 && braceEnd > braceStart) {
      // Extract the JSON portion between the first { and last }
      const jsonPart = trimmed.substring(braceStart, braceEnd + 1);
      try {
        return JSON.parse(jsonPart);
      } catch {
        // If the extracted JSON still fails, continue with other methods
      }
    }

    // Try to find a complete JSON object in the string
    // Check if it looks like a complete object
    if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
      // Try to parse character by character to find where it breaks
      let braceCount = 0;
      let inString = false;
      let escapeNext = false;
      let lastValidPos = -1;
      
      for (let i = 0; i < trimmed.length; i++) {
        const char = trimmed[i];
        
        if (escapeNext) {
          escapeNext = false;
          continue;
        }
        
        if (char === '\\') {
          escapeNext = true;
          continue;
        }
        
        if (char === '"' && !escapeNext) {
          inString = !inString;
          continue;
        }
        
        if (!inString) {
          if (char === '{') braceCount++;
          else if (char === '}') braceCount--;
          
          // Track position of balanced braces
          if (braceCount === 0) {
            lastValidPos = i;
          }
        }
      }
      
      // If we found a valid complete JSON object, try parsing it
      if (lastValidPos > 0) {
        const validJson = trimmed.substring(0, lastValidPos + 1);
        try {
          return JSON.parse(validJson);
        } catch {
          // Still not valid, continue
        }
      }
    }

    // Try to extract key-value pairs manually for simple cases
    const keyValuePattern = /"([^"]+)"\s*:\s*("([^"]*)"|([0-9.]+)|(true|false)|(null))/g;
    const matches = [...trimmed.matchAll(keyValuePattern)];
    
    if (matches.length > 0) {
      const result: Record<string, unknown> = {};
      
      for (const match of matches) {
        const key = match[1];
        let value: unknown;
        
        if (match[3] !== undefined) {
          // String value
          value = match[3];
        } else if (match[4] !== undefined) {
          // Number value
          value = parseFloat(match[4]);
          if (Number.isNaN(value)) {
            value = match[4];
          }
        } else if (match[5] !== undefined) {
          // Boolean value
          value = match[5] === 'true';
        } else if (match[6] !== undefined) {
          // Null value
          value = null;
        }
        
        result[key] = value;
      }
      
      return result;
    }

    // Try to handle simple comma-separated key-value pairs without quotes
    const simplePattern = /([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^,}\]]+)/g;
    const simpleMatches = [...trimmed.matchAll(simplePattern)];
    
    if (simpleMatches.length > 0) {
      const result: Record<string, unknown> = {};
      
      for (const match of simpleMatches) {
        const key = match[1];
        const rawValue = match[2].trim();
        let value: unknown;
        
        if (rawValue === 'true' || rawValue === 'false') {
          value = rawValue === 'true';
        } else if (!isNaN(Number(rawValue)) && rawValue !== '') {
          value = Number(rawValue);
        } else if (rawValue === 'null') {
          value = null;
        } else {
          value = rawValue.replace(/^"|"$/g, ''); // Remove quotes if present
        }
        
        result[key] = value;
      }
      
      if (Object.keys(result).length > 0) {
        return result;
      }
    }

    // Last resort: return an empty object rather than throwing
    console.warn('Could not extract valid JSON from:', input);
    return {};
  }
}