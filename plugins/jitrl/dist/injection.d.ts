/**
 * Context injection generator for JitRL
 *
 * Since Claude Code doesn't expose logprobs, we use context injection
 * as an approximation of the paper's logit adjustment.
 */
import { SearchResult } from "./store.js";
export declare function generateInjection(results: SearchResult[], advantages: Record<string, number>): string;
