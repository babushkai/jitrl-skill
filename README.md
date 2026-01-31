# JitRL for Claude Code

> **Experience-based learning that makes Claude Code smarter with every session**

[![Claude Code Skill](https://img.shields.io/badge/Claude%20Code-Skill-blue)](https://code.claude.com/docs/en/skills)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

JitRL gives Claude Code persistent memory of past successes and failures. When you encounter a problem, JitRL automatically surfaces relevant experiences from your history - what worked, what didn't, and which approaches have the best track record.

## The Problem

Every time you start a new Claude Code session, it's a blank slate. Claude doesn't remember:
- That the same TypeScript error was fixed yesterday by adding an export
- That running `npm test` before commits catches 80% of issues
- That creating new files usually conflicts with existing ones in this project

**JitRL solves this** by storing every interaction as a scored experience and retrieving relevant ones when you face similar situations.

## How It Works

```
┌─────────────────────────────────────────────────────┐
│  You: "Fix this TypeScript error"                   │
├─────────────────────────────────────────────────────┤
│  JitRL searches past experiences...                 │
│                                                     │
│  💡 Past Experience Insights                        │
│                                                     │
│  ✅ Success Pattern                                 │
│  - Edit: Added missing type export (Score: +7)     │
│                                                     │
│  ⚠️ Failure Pattern (Avoid)                        │
│  - Write: Created new .d.ts file (Score: -3)       │
│                                                     │
│  📊 Recommended: Edit (+3.2 advantage)             │
├─────────────────────────────────────────────────────┤
│  Claude uses this context to make better decisions  │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install the Skill

```bash
# Clone to your Claude Code skills directory
git clone https://github.com/anthropics/jitrl-skill ~/.claude/skills/jitrl

# Install dependencies
pip install faiss-cpu numpy

# Optional: For better embeddings
pip install openai
```

### 2. Initialize for Your Project

```bash
cd /your/project
python3 ~/.claude/skills/jitrl/scripts/jitrl.py init
```

### 3. Start Using Claude Code

JitRL works automatically in the background:
- **On each prompt**: Searches for similar past experiences
- **On each response**: Captures what Claude did
- **On session end**: Evaluates and stores the experience

## Commands

| Command | Description |
|---------|-------------|
| `jitrl init` | Initialize for current project |
| `jitrl stats` | Show experience statistics |
| `jitrl search "query"` | Search similar experiences |
| `jitrl clear --confirm` | Clear project memory |
| `jitrl export file.json` | Export all experiences |

## How Scoring Works

Every experience gets a score based on outcome:

| Outcome | Base Score |
|---------|------------|
| Success + "Perfect!" feedback | +10 |
| Success | +5 |
| Partial success | +2 |
| Failure with learning | -2 |
| Complete failure | -5 |

JitRL uses these scores to calculate **advantages** for each action type:

```
Advantage(Edit) = Average_Score(Edit) - Baseline
```

Positive advantage = this approach works better than average in similar situations.

## Configuration

Create `~/.claude-jitrl/config.yaml`:

```yaml
# Search settings
similarity_threshold: 0.6    # Min similarity to retrieve (0.0-1.0)
max_experiences_per_search: 5

# Storage limits
max_experiences: 10000
experience_ttl_days: 90

# Performance
cache_embeddings: true
```

## Project Structure

```
jitrl-skill/
├── SKILL.md           # Skill definition for Claude Code
├── reference.md       # Complete API documentation
├── scripts/
│   ├── jitrl.py       # Main CLI and library
│   ├── __init__.py    # Module exports
│   └── hooks/         # Claude Code hook scripts
│       ├── on_prompt.py
│       ├── on_stop.py
│       └── on_session_end.py
└── examples/
    └── basic_usage.py
```

## Based on Research

JitRL is inspired by the paper ["JitRL: Just-in-Time Reinforcement Learning for LLM Agent Continual Improvements"](https://arxiv.org/abs/2501.18510) which demonstrates how to improve LLM agents through experience replay without gradient updates.

Key innovations applied here:
- **Policy Triplets**: Store `<state, action, outcome>` for each interaction
- **Advantage Calculation**: Rank actions by their relative success rate
- **No Fine-tuning**: Use context injection instead of model updates

## Requirements

- Python 3.8+
- Claude Code
- faiss-cpu (or faiss-gpu)
- numpy
- (Optional) openai - for better embeddings

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

Contributions welcome! Please read our [Contributing Guide](CONTRIBUTING.md) first.

## Related

- [Claude Code Skills](https://code.claude.com/docs/en/skills) - Official documentation
- [Claude Code Hooks](https://code.claude.com/docs/en/hooks) - Hook system docs
- [JitRL Paper](https://arxiv.org/abs/2501.18510) - Original research
