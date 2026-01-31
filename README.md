# JitRL for Claude Code

> **Experience-based learning that makes Claude Code smarter with every session**

[![Claude Code Plugin](https://img.shields.io/badge/Claude%20Code-Plugin-blue)](https://code.claude.com/docs/en/plugins)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

JitRL gives Claude Code persistent memory of past successes and failures. When you encounter a problem, JitRL automatically surfaces relevant experiences from your history.

## Installation (One Command)

```bash
# Add the marketplace
/plugin marketplace add babushkai/jitrl-skill

# Install the plugin
/plugin install jitrl@babushkai-jitrl-skill
```

**That's it.** The plugin automatically:
- Registers hooks for experience capture
- Injects relevant context on each prompt
- Stores and evaluates sessions

### Prerequisites

```bash
pip install faiss-cpu numpy
```

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

## Commands

After installation, these commands are available:

| Command | Description |
|---------|-------------|
| `/jitrl:init` | Initialize for current project |
| `/jitrl:stats` | Show experience statistics |
| `/jitrl:search [query]` | Search similar experiences |

## What Gets Stored

Every interaction is stored as a **Policy Triplet**:

| Component | Description | Example |
|-----------|-------------|---------|
| **State** | Current context | Files read, error messages, goal |
| **Action** | What Claude did | Edit, Write, Bash, etc. |
| **Outcome** | Result | Success/failure, user feedback |

## Scoring System

| Outcome | Score |
|---------|-------|
| Success + "Perfect!" | +10 |
| Success | +5 |
| Partial success | +2 |
| Failure with learning | -2 |
| Complete failure | -5 |

## Configuration

Create `~/.claude-jitrl/config.yaml`:

```yaml
similarity_threshold: 0.6    # Min similarity (0.0-1.0)
max_experiences_per_search: 5
max_experiences: 10000
experience_ttl_days: 90
cache_embeddings: true
```

## Plugin Structure

```
jitrl-skill/
├── .claude-plugin/
│   ├── plugin.json      # Plugin manifest
│   └── marketplace.json # For discoverability
├── skills/
│   └── jitrl-memory/
│       └── SKILL.md     # Main skill definition
├── commands/
│   ├── init.md          # /jitrl:init
│   ├── stats.md         # /jitrl:stats
│   └── search.md        # /jitrl:search
├── hooks/
│   └── hooks.json       # Auto-registered hooks
└── scripts/
    ├── jitrl.py         # Core implementation
    └── hooks/           # Hook handlers
```

## Manual Installation (Alternative)

If you prefer not to use the marketplace:

```bash
# Clone directly to plugins directory
git clone https://github.com/babushkai/jitrl-skill ~/.claude/plugins/jitrl

# Or test locally
claude --plugin-dir ./jitrl-skill
```

## Based on Research

JitRL is inspired by ["JitRL: Just-in-Time Reinforcement Learning for LLM Agent"](https://arxiv.org/abs/2501.18510).

Key innovations:
- **No Fine-tuning**: Uses context injection instead of model updates
- **Experience Replay**: Stores and retrieves relevant past experiences
- **Advantage Calculation**: Ranks actions by relative success rate

## Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Repeated explanations | Every time | ~60% less |
| Error fix attempts | 3-5 tries | 1-2 tries |
| Failed pattern repetition | Frequent | Near zero |

## Contributing

Contributions welcome! Please:
- Open issues for bugs or feature requests
- Submit PRs for improvements
- Help improve documentation

## License

MIT License - see [LICENSE](LICENSE)

## Links

- **GitHub**: https://github.com/babushkai/jitrl-skill
- **JitRL Paper**: https://arxiv.org/abs/2501.18510
- **Claude Code Plugins**: https://code.claude.com/docs/en/plugins
