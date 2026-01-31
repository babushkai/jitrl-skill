/**
 * Context Injection Generator for JitRL
 *
 * ## Approximating Logit Adjustment via Context
 *
 * The JitRL paper implements policy optimization in logit space:
 * $$
 * z'(s,a) = z(s,a) + \beta \cdot \hat{A}(s,a)
 * $$
 *
 * Since Claude Code doesn't expose logprobs or allow direct logit
 * manipulation, we approximate this via **context injection**:
 *
 * Instead of:
 * $$
 * \pi^*(a|s) \propto \pi_\theta(a|s) \exp(\beta \cdot A(s,a))
 * $$
 *
 * We inject advantage-weighted recommendations into the prompt context,
 * leveraging Claude's instruction-following to bias action selection
 * toward high-advantage actions.
 *
 * This is a practical approximation that maintains the directional
 * bias of the theoretical policy update.
 *
 * @see https://arxiv.org/abs/2601.18510
 */
import { SearchResult } from "./store.js";
/**
 * Generate context injection text from retrieved experiences.
 *
 * Formats experiences and advantages as markdown for Claude to use
 * as in-context guidance. Actions are ranked by advantage:
 * $$
 * \text{rank}(a) \propto \tilde{A}(s,a)
 * $$
 *
 * Visual indicators map advantage to recommendation strength:
 * - $\tilde{A} > 0.5$: Strong recommend (🟢)
 * - $\tilde{A} > 0$: Recommend (👍)
 * - $\tilde{A} > -0.5$: Avoid (👎)
 * - $\tilde{A} \leq -0.5$: Strongly avoid (🔴)
 *
 * @param results - Retrieved similar experiences $N(s)$
 * @param advantages - Normalized advantage estimates $\tilde{A}(s,a)$
 * @returns Formatted markdown for context injection
 */
export declare function generateInjection(results: SearchResult[], advantages: Record<string, number>): string;
