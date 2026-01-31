"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.generateInjection = generateInjection;
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
function generateInjection(results, advantages) {
    if (results.length === 0) {
        return "";
    }
    const lines = ["## 💡 Past Experience Insights\n"];
    // Success patterns (positive score)
    const successes = results.filter((r) => r.experience.score > 0);
    if (successes.length > 0) {
        lines.push("### ✅ Success Patterns");
        for (const r of successes.slice(0, 3)) {
            const exp = r.experience;
            const tool = exp.action.tool_name || "action";
            const summary = (exp.action.summary || "").slice(0, 60);
            lines.push(`- **${tool}**: ${summary}`);
            lines.push(`  - Similarity: ${r.similarity.toFixed(2)}, Score: ${exp.score > 0 ? "+" : ""}${exp.score}`);
        }
    }
    // Failure patterns (negative score)
    const failures = results.filter((r) => r.experience.score < 0);
    if (failures.length > 0) {
        lines.push("\n### ⚠️ Failure Patterns (Avoid)");
        for (const r of failures.slice(0, 2)) {
            const exp = r.experience;
            const tool = exp.action.tool_name || "action";
            const error = exp.outcome.error_summary || exp.action.summary || "";
            lines.push(`- **${tool}**: ${error.slice(0, 60)}`);
        }
    }
    // Advantages (normalized + episode weighted)
    if (Object.keys(advantages).length > 0) {
        lines.push("\n### 📊 Recommended Approach (Advantage-Weighted)");
        const sorted = Object.entries(advantages).sort((a, b) => b[1] - a[1]);
        for (const [tool, adv] of sorted.slice(0, 4)) {
            let emoji;
            if (adv > 0.5) {
                emoji = "🟢";
            }
            else if (adv > 0) {
                emoji = "👍";
            }
            else if (adv > -0.5) {
                emoji = "👎";
            }
            else {
                emoji = "🔴";
            }
            lines.push(`- ${emoji} **${tool}**: Advantage ${adv > 0 ? "+" : ""}${adv.toFixed(2)}`);
        }
    }
    return lines.join("\n");
}
