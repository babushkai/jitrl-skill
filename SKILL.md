---
name: jitrl
description: Experience-based learning system that remembers successful and failed patterns across sessions. Use when you want Claude to learn from past mistakes, recall similar problems solved before, or avoid repeating failed approaches. Automatically activated when debugging errors, refactoring code, or tackling problems similar to past work.
allowed-tools: Bash(python3 *), Read, Write, Edit
---

# JitRL: Just-In-Time Reinforcement Learning for Claude Code

## Overview

JitRL gives Claude persistent memory of past experiences. When you encounter a problem, JitRL searches for similar situations from your history and provides:

- **Success patterns**: What worked before in similar situations
- **Failure patterns**: What to avoid (learned from past mistakes)
- **Action advantages**: Which tools/approaches have the best track record

## Quick Start

Initialize JitRL for your current project:

```bash
python3 ~/.claude/skills/jitrl/scripts/jitrl.py init
```

## How It Works

### 1. Experience Storage

Every interaction is stored as a **Policy Triplet**:
- **State**: Current context (files, errors, goals)
- **Action**: What Claude did (tool used, changes made)
- **Outcome**: Result (success/failure, user feedback)

### 2. Similarity Search

When you face a new problem, JitRL:
1. Encodes your current context as a vector
2. Searches past experiences using Faiss
3. Ranks results by similarity AND outcome score

### 3. Advantage Calculation

For each action type, JitRL calculates:
```
Advantage = Average_Score(action) - Baseline_Score
```

Positive advantage = this approach worked better than average.

## Commands

### Check Status
```bash
python3 ~/.claude/skills/jitrl/scripts/jitrl.py stats
```

### Search Similar Experiences
```bash
python3 ~/.claude/skills/jitrl/scripts/jitrl.py search "fix TypeScript type error"
```

### Clear Project Memory
```bash
python3 ~/.claude/skills/jitrl/scripts/jitrl.py clear --confirm
```

### Export Experiences
```bash
python3 ~/.claude/skills/jitrl/scripts/jitrl.py export experiences.json
```

## Context Injection Format

When similar experiences are found, Claude receives:

```markdown
## 💡 Past Experience Insights

### ✅ Success Patterns (Score > 0)
- **Edit**: Fixed interface by adding missing export
  - Similarity: 0.85, Score: +7

### ⚠️ Failure Patterns (Avoid)
- **Write**: Created new type file (conflicted with existing)
  - Similarity: 0.78, Score: -3

### 📊 Recommended Approach
- 👍 **Edit**: Advantage +3.2
- 👎 **Write**: Advantage -1.5
```

## Scoring System

| Outcome | Base Score |
|---------|------------|
| Success + Positive feedback | +10 |
| Success | +5 |
| Partial success | +2 |
| Failure with learning | -2 |
| Complete failure | -5 |

### Bonus/Penalty Modifiers
- "Perfect", "Great" feedback: +5
- "Good" feedback: +2
- No follow-up needed: +2
- "Wrong", "Bad" feedback: -3

## Integration with Claude Code Hooks

JitRL uses three hooks:

1. **UserPromptSubmit**: Injects similar experiences before Claude responds
2. **Stop**: Captures interaction details
3. **SessionEnd**: Evaluates and stores experiences

## Configuration

Edit `~/.claude-jitrl/config.yaml`:

```yaml
# Core settings
gamma: 0.95              # Discount factor for future rewards
similarity_threshold: 0.6 # Minimum similarity for retrieval

# Performance
cache_embeddings: true
max_experiences: 10000
experience_ttl_days: 90

# Evaluation
use_llm_evaluation: false  # Set true for more accurate scoring
evaluation_model: "gpt-4o-mini"
```

## Best Practices

1. **Let it learn**: The more you use Claude Code, the smarter JitRL becomes
2. **Provide feedback**: Say "that worked" or "wrong approach" to improve scoring
3. **Project-specific**: Each project has its own experience memory
4. **Clear stale data**: Run `jitrl.py clear` when starting fresh

## Troubleshooting

### No experiences found
- Check if JitRL is initialized: `jitrl.py stats`
- Lower the similarity threshold in config

### Slow performance
- Enable embedding cache
- Reduce `max_experiences`
- Clear old experiences

### Wrong recommendations
- Experiences may need more data
- Consider enabling LLM evaluation for better scoring

## Dependencies

```bash
pip install faiss-cpu numpy
# Optional: pip install openai  # For better embeddings
```

## File Locations

```
~/.claude-jitrl/
├── experiences/{project_hash}/
│   ├── episodes.jsonl      # Experience data
│   └── step_metadata.pkl   # Faiss metadata
├── indexes/{project_hash}/
│   └── state_vectors.index # Vector index
├── cache/embeddings/       # Embedding cache
└── config.yaml             # Configuration
```

## See Also

- [reference.md](reference.md) - Complete API reference
- [examples/](examples/) - Usage examples
- [JitRL Paper](https://arxiv.org/abs/2601.18510) - Original research
