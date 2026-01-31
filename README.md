# JitRL for Claude Code

> **Experience-based learning that makes Claude Code smarter with every session**

[![MCP Server](https://img.shields.io/badge/MCP-Server-blue)](https://modelcontextprotocol.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue.svg)](https://www.typescriptlang.org/)

JitRL gives Claude Code persistent memory of past successes and failures. When you encounter a problem, JitRL automatically surfaces relevant experiences from your history.

## Installation

### Via Plugin Marketplace

```bash
# Add the marketplace
/plugin marketplace add babushkai/jitrl-skill

# Install the plugin
/plugin install jitrl@babushkai-jitrl-skill
```

### Manual MCP Configuration

Add to your Claude Code MCP settings:

```json
{
  "mcpServers": {
    "jitrl": {
      "command": "npx",
      "args": ["-y", "@babushkai/jitrl-mcp"]
    }
  }
}
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

## MCP Tools

| Tool | Description |
|------|-------------|
| `jitrl_search` | Search for similar past experiences |
| `jitrl_store` | Store a new experience triplet |
| `jitrl_inject` | Get context injection for current prompt |
| `jitrl_stats` | View experience statistics |
| `jitrl_clear` | Clear project memory |

## Algorithm (from Paper)

Based on ["JitRL: Just-in-Time Reinforcement Learning for LLM Agent"](https://arxiv.org/abs/2501.18510).

### Key Features Implemented

| Feature | Description |
|---------|-------------|
| **Dual Vector Search** | History index (0.25) + State index (0.75) |
| **Jaccard N-gram** | History unigrams (0.3) + State 4-grams (0.7) |
| **Discounted Returns** | γ^t weighted future rewards (γ=0.95) |
| **Normalized Advantages** | Divide by max positive advantage |
| **Episode Weighting** | 1.0 → 1.5 over 50 episodes |
| **Dynamic Threshold** | Decreases with step count |

### Why MCP Server?

| Approach | Startup Time | Memory |
|----------|-------------|--------|
| Python hooks | ~500ms/call | New process each time |
| **MCP Server** | **~1ms/call** | **Persistent process** |

The MCP server stays running, so there's no cold-start penalty.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Claude Code                                │
│  ┌───────────────────────────────────────┐  │
│  │  MCP Client                           │  │
│  └──────────────┬────────────────────────┘  │
│                 │                           │
│                 ▼                           │
│  ┌───────────────────────────────────────┐  │
│  │  JitRL MCP Server (persistent)        │  │
│  │  ┌─────────────┐  ┌─────────────────┐ │  │
│  │  │ History     │  │ State           │ │  │
│  │  │ HNSW Index  │  │ HNSW Index      │ │  │
│  │  └─────────────┘  └─────────────────┘ │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │ Experience Store (JSON)         │  │  │
│  │  └─────────────────────────────────┘  │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## Scoring System

| Outcome | Score |
|---------|-------|
| Success + "Perfect!" | +10 |
| Success | +5 |
| Partial success | +2 |
| Failure with learning | -2 |
| Complete failure | -5 |

## Development

```bash
# Clone
git clone https://github.com/babushkai/jitrl-skill
cd jitrl-skill

# Install dependencies
npm install

# Build
npm run build

# Run locally
npm start
```

## Note on Logit Adjustment

The original JitRL paper modifies output logits directly:

```
π*(a|s) = π₀(a|s) · exp(A(s,a) / β)
```

Since Claude Code doesn't expose logprobs, we use **context injection** as an approximation. This works because Claude effectively follows contextual hints about what worked/failed before.

See [COMPARISON.md](COMPARISON.md) for detailed analysis.

## Contributing

Contributions welcome! Please:
- Open issues for bugs or feature requests
- Submit PRs for improvements

## License

MIT License - see [LICENSE](LICENSE)

## Links

- **GitHub**: https://github.com/babushkai/jitrl-skill
- **JitRL Paper**: https://arxiv.org/abs/2501.18510
- **MCP Protocol**: https://modelcontextprotocol.io
