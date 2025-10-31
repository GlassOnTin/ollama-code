/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  GenerateContentResponse,
  GenerateContentParameters,
  FinishReason,
  Part,
  FunctionCall,
  FunctionResponse,
} from '@google/genai';
import type {
  ChatCompletion,
  ChatCompletionChunk,
  OpenAIRequestFormat,
} from 'openai/resources/chat/index.js';

/**
 * Utility class for converting between OpenAI and Gemini API formats
 */
export class OpenAIFormatConverter {
  /**
   * Convert Gemini tools to OpenAI format
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
            (part): part is FunctionCall => 'functionCall' in part
          );

          const functionResponses = (content.parts || []).filter(
            (part): part is FunctionResponse => 'functionResponse' in part
          );

          if (functionCalls.length > 0) {
            const tool_calls = functionCalls.map((fc, index) => ({
              id: `call_${Date.now()}_${index}`,
              type: 'function' as const,
              function: {
                name: fc.functionCall.name,
                arguments: JSON.stringify(fc.functionCall.args || {}),
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
  private static safeJsonParse(input: string): Record<string, unknown> {
    try {
      return JSON.parse(input);
    } catch {
      // If that fails, try to find valid JSON patterns
      return this.extractPartialJson(input);
    }
  }

  /**
   * Extract valid JSON from potentially partial JSON string
   * This handles cases where streaming chunks contain incomplete JSON
   */
  private static extractPartialJson(input: string): Record<string, unknown> {
    if (!input || typeof input !== 'string') {
      return {};
    }

    // First try to parse the entire string
    try {
      return JSON.parse(input);
    } catch {
      // If that fails, try to find valid JSON patterns
    }

    // Try to find a complete JSON object in the string
    const trimmed = input.trim();
    
    // Check if it looks like a complete object
    if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
      // Try to parse character by character to find where it breaks
      let braceCount = 0;
      let inString = false;
      let escapeNext = false;
      
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
          
          // If we have balanced braces and are at the end, try parsing
          if (braceCount === 0 && i === trimmed.length - 1) {
            try {
              return JSON.parse(trimmed);
            } catch {
              // Still not valid, continue
            }
          }
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

    // Last resort: return an empty object rather than throwing
    console.warn('Could not extract valid JSON from:', input);
    return {};
  }
}