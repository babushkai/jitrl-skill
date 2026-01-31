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
export {};
