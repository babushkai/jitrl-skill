#!/usr/bin/env node
/**
 * JitRL MCP Server
 *
 * Experience-based learning for Claude Code via Model Context Protocol.
 *
 * ## Paper Reference
 * "JitRL: Just-in-Time Reinforcement Learning for Real-World LLM Agents"
 * https://arxiv.org/abs/2601.18510
 *
 * ## Overview
 *
 * JitRL is a training-free framework that enables test-time policy
 * optimization without any gradient updates. The core idea is to:
 *
 * 1. Maintain a dynamic memory of experiences: $\mathcal{M} = \{(s_i, a_i, G_i)\}$
 * 2. Retrieve relevant trajectories via similarity: $N(s) = \text{Top-}k(J(s, s_i))$
 * 3. Estimate action advantages on-the-fly: $\hat{A}(s,a) = \hat{Q}(s,a) - \hat{V}(s)$
 * 4. Adjust policy via closed-form solution: $\pi^*(a|s) \propto \pi_\theta(a|s) \exp(\beta A(s,a))$
 *
 * Since Claude Code doesn't expose logprobs, we approximate step 4 via
 * context injection of advantage-weighted recommendations.
 *
 * ## MCP Tools Provided
 *
 * - `jitrl_search`: Retrieve similar experiences with advantages
 * - `jitrl_store`: Store new $(s, a, G)$ triplets
 * - `jitrl_inject`: Get formatted context injection text
 * - `jitrl_stats`: View memory statistics
 * - `jitrl_clear`: Reset memory
 * - `jitrl_increment_episode`: Update episode counter
 *
 * Runs as persistent process - no startup overhead per request.
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

import { ExperienceStore } from "./store.js";
import { generateInjection } from "./injection.js";

const store = new ExperienceStore();

const server = new Server(
  {
    name: "jitrl",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "jitrl_search",
        description: "Search for similar past experiences based on current context. Returns relevant experiences with similarity scores and advantages.",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Current context or problem description to search for",
            },
            k: {
              type: "number",
              description: "Number of results to return (default: 5)",
              default: 5,
            },
            step_count: {
              type: "number",
              description: "Current step in session (affects threshold)",
              default: 0,
            },
          },
          required: ["query"],
        },
      },
      {
        name: "jitrl_store",
        description: "Store a new experience (state, action, outcome triplet)",
        inputSchema: {
          type: "object",
          properties: {
            state: {
              type: "object",
              description: "Current state/context when action was taken",
            },
            action: {
              type: "object",
              description: "Action taken (tool_name, summary, parameters)",
              properties: {
                tool_name: { type: "string" },
                summary: { type: "string" },
                parameters: { type: "object" },
              },
            },
            outcome: {
              type: "object",
              description: "Result of the action",
              properties: {
                success: { type: "boolean" },
                error_summary: { type: "string" },
                user_feedback: { type: "string" },
              },
            },
            trajectory_context: {
              type: "string",
              description: "Summary of history leading to this point",
            },
            future_rewards: {
              type: "array",
              items: { type: "number" },
              description: "Future rewards for discounted return calculation",
            },
          },
          required: ["state", "action", "outcome"],
        },
      },
      {
        name: "jitrl_inject",
        description: "Get context injection text for the current prompt. This provides past experience insights formatted for Claude.",
        inputSchema: {
          type: "object",
          properties: {
            context: {
              type: "string",
              description: "Current prompt/context to find relevant experiences for",
            },
            step_count: {
              type: "number",
              description: "Current step in session",
              default: 0,
            },
          },
          required: ["context"],
        },
      },
      {
        name: "jitrl_stats",
        description: "Get statistics about the experience store",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
      {
        name: "jitrl_clear",
        description: "Clear all experiences for the current project",
        inputSchema: {
          type: "object",
          properties: {
            confirm: {
              type: "boolean",
              description: "Must be true to confirm deletion",
            },
          },
          required: ["confirm"],
        },
      },
      {
        name: "jitrl_increment_episode",
        description: "Increment episode counter (call at session end)",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "jitrl_search": {
        const query = args?.query as string;
        const k = (args?.k as number) || 5;
        const stepCount = (args?.step_count as number) || 0;

        const results = await store.search(query, k, stepCount);
        const advantages = store.getAdvantages(results);

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ results, advantages }, null, 2),
            },
          ],
        };
      }

      case "jitrl_store": {
        const experience = await store.add(
          args?.state as Record<string, unknown>,
          args?.action as { tool_name?: string; summary?: string; parameters?: Record<string, unknown> },
          args?.outcome as { success?: boolean; error_summary?: string; user_feedback?: string },
          args?.trajectory_context as string,
          args?.future_rewards as number[]
        );

        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ stored: true, score: experience.score }),
            },
          ],
        };
      }

      case "jitrl_inject": {
        const context = args?.context as string;
        const stepCount = (args?.step_count as number) || 0;

        const results = await store.search(context, 5, stepCount);
        const advantages = store.getAdvantages(results);
        const injection = generateInjection(results, advantages);

        return {
          content: [
            {
              type: "text",
              text: injection || "No relevant past experiences found.",
            },
          ],
        };
      }

      case "jitrl_stats": {
        const stats = store.getStats();
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(stats, null, 2),
            },
          ],
        };
      }

      case "jitrl_clear": {
        if (args?.confirm !== true) {
          return {
            content: [
              {
                type: "text",
                text: "Error: Must set confirm=true to clear experiences",
              },
            ],
          };
        }
        store.clear();
        return {
          content: [
            {
              type: "text",
              text: "All experiences cleared.",
            },
          ],
        };
      }

      case "jitrl_increment_episode": {
        store.incrementEpisode();
        return {
          content: [
            {
              type: "text",
              text: `Episode count: ${store.episodeCount}`,
            },
          ],
        };
      }

      default:
        return {
          content: [
            {
              type: "text",
              text: `Unknown tool: ${name}`,
            },
          ],
          isError: true,
        };
    }
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: `Error: ${error instanceof Error ? error.message : String(error)}`,
        },
      ],
      isError: true,
    };
  }
});

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("JitRL MCP Server running");
}

main().catch(console.error);
