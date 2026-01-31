/**
 * Context injection generator for JitRL
 *
 * Since Claude Code doesn't expose logprobs, we use context injection
 * as an approximation of the paper's logit adjustment.
 */

import { SearchResult } from "./store.js";

export function generateInjection(
  results: SearchResult[],
  advantages: Record<string, number>
): string {
  if (results.length === 0) {
    return "";
  }

  const lines: string[] = ["## 💡 Past Experience Insights\n"];

  // Success patterns (positive score)
  const successes = results.filter((r) => r.experience.score > 0);
  if (successes.length > 0) {
    lines.push("### ✅ Success Patterns");
    for (const r of successes.slice(0, 3)) {
      const exp = r.experience;
      const tool = exp.action.tool_name || "action";
      const summary = (exp.action.summary || "").slice(0, 60);
      lines.push(`- **${tool}**: ${summary}`);
      lines.push(
        `  - Similarity: ${r.similarity.toFixed(2)}, Score: ${exp.score > 0 ? "+" : ""}${exp.score}`
      );
    }
  }

  // Failure patterns (negative score)
  const failures = results.filter((r) => r.experience.score < 0);
  if (failures.length > 0) {
    lines.push("\n### ⚠️ Failure Patterns (Avoid)");
    for (const r of failures.slice(0, 2)) {
      const exp = r.experience;
      const tool = exp.action.tool_name || "action";
      const error =
        exp.outcome.error_summary || exp.action.summary || "";
      lines.push(`- **${tool}**: ${error.slice(0, 60)}`);
    }
  }

  // Advantages (normalized + episode weighted)
  if (Object.keys(advantages).length > 0) {
    lines.push("\n### 📊 Recommended Approach (Advantage-Weighted)");

    const sorted = Object.entries(advantages).sort((a, b) => b[1] - a[1]);

    for (const [tool, adv] of sorted.slice(0, 4)) {
      let emoji: string;
      if (adv > 0.5) {
        emoji = "🟢";
      } else if (adv > 0) {
        emoji = "👍";
      } else if (adv > -0.5) {
        emoji = "👎";
      } else {
        emoji = "🔴";
      }
      lines.push(`- ${emoji} **${tool}**: Advantage ${adv > 0 ? "+" : ""}${adv.toFixed(2)}`);
    }
  }

  return lines.join("\n");
}
