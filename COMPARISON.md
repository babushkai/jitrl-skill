# JitRL Implementation Comparison

## Original Paper vs Our Implementation

This document compares our Claude Code implementation with the original JitRL paper and codebase.

---

## Summary of Differences

| Feature | Original JitRL | Our Implementation | Status |
|---------|---------------|-------------------|--------|
| **Dual Vector Search** | History + State indexes | Single index | ❌ Missing |
| **Jaccard N-gram Similarity** | 0.3 × history + 0.7 × state | Not implemented | ❌ Missing |
| **LLM Step Scoring** | Structured JSON, -10 to +10 | Simple heuristics | ⚠️ Partial |
| **Discounted Returns** | γ^t weighted future rewards | Basic scoring | ⚠️ Partial |
| **Normalized Advantages** | Divide by max positive | Raw difference only | ❌ Missing |
| **Episode Weighting** | 1.0 → 1.5 over 50 episodes | Not implemented | ❌ Missing |
| **Dynamic Threshold** | Decreases with step count | Fixed threshold | ❌ Missing |
| **Trajectory Context** | LLM-generated summaries | Raw data storage | ❌ Missing |
| **Logit Adjustment** | Add normalized advantage | Context injection only | ❌ Different approach |

---

## Detailed Comparison

### 1. Similarity Search

**Original JitRL:**
```python
# Dual vector search with weighted combination
hist_scores, hist_indices = self.history_index.search(query_history_vec, k)
state_scores, state_indices = self.state_index.search(query_state_vec, k)

# Combine: 0.25 history + 0.75 state
combined_score = 0.25 * hist_scores[0][i] + 0.75 * state_scores[0][i]

# Also compute Jaccard N-gram similarity
sim1 = self._jaccard(query_history_tokens, history_tokens, ngram=1)
sim2 = self._jaccard(query_current_tokens, current_tokens, ngram=4)
similarity = sim1 * 0.3 + sim2 * 0.7
```

**Our Implementation:**
```python
# Single vector search
scores, indices = self.state_index.search(query_vector, k)
```

**Gap:** We're missing the dual-index approach and Jaccard similarity that provides more robust matching.

---

### 2. Advantage Calculation

**Original JitRL:**
```python
# 1. Aggregate discounted rewards per action
for action, rewards in action_rewards.items():
    action_avg_rewards[action] = sum(rewards) / len(rewards)

# 2. Calculate baseline
all_rewards = [r for rewards in action_rewards.values() for r in rewards]
overall_avg_reward = sum(all_rewards) / len(all_rewards)

# 3. Raw advantage
action_advantages[action] = avg_reward - overall_avg_reward

# 4. Normalize by max positive
positive_advs = [adv for adv in adv_values if adv > 0]
max_positive = max(positive_advs)
normalized_advantages = {action: adv / max_positive for action, adv in action_advantages.items()}

# 5. Episode weight (increases over time)
episode_weight = min(1.0 + (current_episode / 50.0) * 0.5, 1.5)
weighted_normalized_advantage = normalized_advantage * episode_weight

# 6. Apply to logit
corrected_logprob = normalized_prob + weighted_normalized_advantage
```

**Our Implementation:**
```python
# Basic advantage only
advantages = {k: v - baseline for k, v in action_avg.items()}
```

**Gap:** Missing normalization, episode weighting, and proper logit adjustment.

---

### 3. LLM Step Scoring

**Original JitRL:**
```python
# Structured JSON evaluation with reasoning
response_format = {
    "type": "json_schema",
    "json_schema": {
        "schema": {
            "properties": {
                "step_analysis": {
                    "items": {
                        "properties": {
                            "step": {"type": "integer"},
                            "action": {"type": "string"},
                            "detailed_reasoning": {"type": "string"},  # 80+ words
                            "score": {"type": "number", "minimum": -10, "maximum": 10}
                        }
                    }
                },
                "overall_assessment": {"type": "string"}
            }
        }
    }
}
```

**Our Implementation:**
```python
# Simple heuristic scoring
base_score = 5 if outcome.get('success', False) else -2
if 'perfect' in feedback.lower():
    base_score += 5
```

**Gap:** Our scoring lacks LLM-based trajectory evaluation with detailed reasoning.

---

### 4. Discounted Returns

**Original JitRL:**
```python
# Store future rewards for flexible discounting
future_rewards = []
for u in range(step_index, len(steps)):
    future_rewards.append(steps[u].get('llm_step_score', 0))

# At retrieval time, compute discounted return
discounted_return = 0.0
for u, step_reward in enumerate(future_rewards):
    discount = self.gamma ** u
    discounted_return += discount * step_reward
```

**Our Implementation:**
```python
# We store final score only, not step-by-step rewards
```

**Gap:** Missing per-step reward storage and proper discounting.

---

### 5. Dynamic Threshold

**Original JitRL:**
```python
# Threshold decreases as episode progresses
step_count = len(game_history)
max_steps = 20
threshold_decrease = 0.1 * (step_count / max_steps)
dynamic_threshold = r - threshold_decrease  # e.g., 0.7 → 0.6
```

**Our Implementation:**
```python
# Fixed threshold
threshold = 0.6
```

**Gap:** Static threshold doesn't adapt to exploration needs.

---

### 6. Core Algorithm: Logit Adjustment

The key innovation in JitRL is **direct logit adjustment** rather than context injection:

**Original JitRL (from paper):**
```
π*(a|s) = π₀(a|s) · exp(A(s,a) / β)

Where:
- π₀ = base policy (frozen LLM)
- A(s,a) = advantage estimate from retrieved experiences
- β = temperature parameter
```

In code:
```python
corrected_logprob = normalized_prob + weighted_normalized_advantage
```

**Our Implementation:**
We use **context injection** instead of logit adjustment because Claude Code doesn't expose logit access:
```python
# Inject context into prompt
context = """
## Past Experience Insights
✅ Edit: worked before (Score: +7)
⚠️ Write: failed before (Score: -3)
"""
```

**Gap:** This is fundamentally different. We can't do true logit adjustment without API access, so context injection is our approximation.

---

## What We're Doing Right

✅ **Policy Triplet Storage**: `<state, action, outcome>` - matches paper
✅ **Experience Replay**: Storing and retrieving past experiences
✅ **Advantage Concept**: Calculating relative performance vs baseline
✅ **Faiss Vector Database**: Same technology as original
✅ **Hook-based Capture**: Automatic experience collection

---

## Recommended Improvements

### Priority 1: Implement Dual Vector Search
```python
# Add history index alongside state index
self.history_index = faiss.IndexFlatIP(self.vector_dim)
self.state_index = faiss.IndexFlatIP(self.vector_dim)
```

### Priority 2: Add Jaccard N-gram Similarity
```python
def jaccard_ngrams(self, a: str, b: str, n: int = 4) -> float:
    a_ngrams = set(zip(*[a.split()[i:] for i in range(n)]))
    b_ngrams = set(zip(*[b.split()[i:] for i in range(n)]))
    intersection = len(a_ngrams & b_ngrams)
    union = len(a_ngrams | b_ngrams)
    return intersection / union if union > 0 else 0.0
```

### Priority 3: Implement Proper LLM Scoring
```python
def evaluate_with_llm(self, trajectory) -> List[float]:
    prompt = f"""Score each step from -10 to +10...
    Return JSON: {{"step_analysis": [...], "overall_assessment": "..."}}
    """
    # Use structured output
```

### Priority 4: Add Episode Weighting
```python
episode_weight = min(1.0 + (self.episode_count / 50.0) * 0.5, 1.5)
weighted_advantage = normalized_advantage * episode_weight
```

---

## Fundamental Limitation

**The original JitRL modifies output logits directly:**
```
corrected_logprob = original_logprob + advantage
```

**Claude Code doesn't expose logprobs**, so we can only approximate via context injection.

This is a fundamental architectural difference. Our approach works because:
1. Claude is good at following contextual hints
2. Showing past success/failure patterns influences behavior
3. We rank recommendations by advantage

But it's not mathematically equivalent to the paper's closed-form solution.

---

## Conclusion

Our implementation captures the **spirit** of JitRL (experience-based learning without fine-tuning) but lacks several technical details from the original:

| Aspect | Original | Ours |
|--------|----------|------|
| Method | Logit adjustment | Context injection |
| Search | Dual vector + Jaccard | Single vector |
| Scoring | LLM structured | Heuristic |
| Advantages | Normalized + weighted | Raw |

For production Claude Code use, our approach is practical and effective. For research replication, the gaps above should be addressed.
