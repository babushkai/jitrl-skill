# JitRL Reference Guide

Complete API reference for JitRL experience-based learning system.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Claude Code Session                      │
├─────────────────────────────────────────────────────────────┤
│  UserPromptSubmit        Stop              SessionEnd        │
│        │                  │                    │             │
│        ▼                  ▼                    ▼             │
│  ┌──────────┐      ┌───────────┐       ┌────────────┐       │
│  │ Context  │      │ Experience │       │   Eval &   │       │
│  │ Inject   │      │  Capture   │       │   Store    │       │
│  └────┬─────┘      └─────┬─────┘       └──────┬─────┘       │
│       │                  │                    │             │
│       ▼                  ▼                    ▼             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Experience Memory (Faiss + JSONL)       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Core Concepts

### Policy Triplet

Every experience is stored as a **Policy Triplet** `<S, A, O>`:

| Component | Description | Example |
|-----------|-------------|---------|
| **State (S)** | Current context when action was taken | Files read, error messages, user goal |
| **Action (A)** | What Claude did | Tool used, code written, command run |
| **Outcome (O)** | Result of the action | Success/failure, user feedback |

### Advantage Calculation

JitRL calculates **advantages** for each action type:

```
Advantage(action) = Mean_Score(action) - Baseline_Score
```

Where:
- `Mean_Score(action)` = average score of all experiences using that action
- `Baseline_Score` = average score across all actions

**Interpretation:**
- Positive advantage → This action performs better than average
- Negative advantage → This action performs worse than average
- Zero → Average performance

## Python API

### ExperienceStore

```python
from jitrl import ExperienceStore, JitRLConfig

# Initialize
config = JitRLConfig(Path.home() / '.claude-jitrl')
store = ExperienceStore('/path/to/project', config)

# Add experience
experience = store.add(
    state={
        'files_read': ['src/main.py', 'src/utils.py'],
        'error_context': 'TypeError: cannot read property',
        'user_goal': 'fix the type error'
    },
    action={
        'tool_name': 'Edit',
        'summary': 'Added type annotation to function parameter',
        'file_changed': 'src/main.py'
    },
    outcome={
        'success': True,
        'user_feedback': 'perfect',
        'follow_up_needed': False
    }
)

# Search similar
results = store.search(
    query='fix type error in React component',
    k=5,
    threshold=0.6
)

# Get advantages
advantages = store.get_advantages(results)
# {'Edit': 2.3, 'Write': -1.5, 'Bash': 0.2}

# Statistics
stats = store.get_stats()
```

### Embedder

```python
from jitrl import Embedder

embedder = Embedder(cache_dir=Path.home() / '.claude-jitrl/cache')

# Embed text
vector = embedder.embed('fix TypeScript type error')
# Returns: np.ndarray of shape (1536,)

# Uses OpenAI API if available, falls back to hash-based embedding
```

### Configuration

```python
from jitrl import JitRLConfig

config = JitRLConfig(Path.home() / '.claude-jitrl')

# Access settings
gamma = config.get('gamma', 0.95)
threshold = config.get('similarity_threshold', 0.6)
```

## CLI Reference

### `jitrl init`

Initialize JitRL for the current project.

```bash
python3 jitrl.py init
```

**Output:**
```
✅ JitRL initialized for: /path/to/project
   Project hash: a1b2c3d4e5f6
   Experiences: 0
   Data dir: ~/.claude-jitrl/experiences/a1b2c3d4e5f6
```

### `jitrl stats`

Show experience statistics.

```bash
python3 jitrl.py stats
```

**Output:**
```
📊 JitRL Statistics
   Project: a1b2c3d4e5f6
   Experiences: 127
   Avg Score: 3.45
   Positive: 98
   Negative: 29
   Top Actions:
      - Edit: 67
      - Bash: 32
      - Write: 18
      - Read: 10
```

### `jitrl search`

Search for similar experiences.

```bash
python3 jitrl.py search "fix async await error" -k 5 -t 0.5
```

**Options:**
- `-k`: Number of results (default: 5)
- `-t, --threshold`: Minimum similarity (default: 0.6)

### `jitrl inject`

Get context injection for a prompt.

```bash
python3 jitrl.py inject "how do I fix this TypeScript error?"
```

**Output:** Markdown-formatted context to inject into Claude's prompt.

### `jitrl store`

Store a new experience manually.

```bash
python3 jitrl.py store '{"state": {"goal": "..."}, "action": {"tool_name": "Edit"}, "outcome": {"success": true}}'
```

### `jitrl clear`

Clear all experiences for current project.

```bash
python3 jitrl.py clear --confirm
```

### `jitrl export`

Export experiences to JSON.

```bash
python3 jitrl.py export experiences.json
```

## Hooks Integration

### hooks.json Configuration

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [{
          "type": "command",
          "command": "python3 ~/.claude/skills/jitrl/scripts/hooks/on_prompt.py"
        }]
      }
    ],
    "Stop": [
      {
        "matcher": "",
        "hooks": [{
          "type": "command",
          "command": "python3 ~/.claude/skills/jitrl/scripts/hooks/on_stop.py"
        }]
      }
    ],
    "SessionEnd": [
      {
        "matcher": "",
        "hooks": [{
          "type": "command",
          "command": "python3 ~/.claude/skills/jitrl/scripts/hooks/on_session_end.py"
        }]
      }
    ]
  }
}
```

### Hook Event Data

Each hook receives data via `CLAUDE_HOOK_DATA` environment variable:

**UserPromptSubmit:**
```json
{
  "cwd": "/path/to/project",
  "prompt": "user's prompt text",
  "session_id": "abc123"
}
```

**Stop:**
```json
{
  "cwd": "/path/to/project",
  "transcript_path": "/path/to/transcript.jsonl",
  "stop_hook_active": true
}
```

**SessionEnd:**
```json
{
  "cwd": "/path/to/project",
  "session_id": "abc123",
  "reason": "prompt_input_exit"
}
```

## Scoring System

### Base Scores

| Outcome | Score |
|---------|-------|
| Success + Positive feedback | +10 |
| Success | +5 |
| Partial success | +2 |
| Failure with learning | -2 |
| Complete failure | -5 |

### Modifiers

| Condition | Modifier |
|-----------|----------|
| "Perfect", "Great" in feedback | +5 |
| "Good" in feedback | +2 |
| No follow-up needed | +2 |
| "Wrong", "Bad" in feedback | -3 |

### Score Range

All scores are clamped to **[-10, +10]**.

## Vector Search

### Faiss Index

JitRL uses Faiss with Inner Product (IP) similarity for fast nearest-neighbor search:

```python
# Index creation
index = faiss.IndexFlatIP(1536)  # 1536-dim vectors

# Search
scores, indices = index.search(query_vector, k=5)
```

### Similarity Threshold

Default threshold: **0.6**

- Higher (0.8+): Only very similar experiences
- Lower (0.4-): More experiences, less relevant
- Recommended: 0.5-0.7

## File Locations

```
~/.claude-jitrl/
├── experiences/
│   └── {project_hash}/
│       ├── episodes.jsonl      # Raw experience data
│       └── step_metadata.pkl   # Pickled metadata for Faiss
├── indexes/
│   └── {project_hash}/
│       └── state_vectors.index # Faiss index file
├── cache/
│   └── embeddings/             # Cached embedding vectors
└── config.yaml                 # Global configuration
```

### Project Hash

Each project gets a unique hash based on its path:

```python
project_hash = hashlib.md5(project_path.encode()).hexdigest()[:12]
```

## Configuration Reference

### config.yaml

```yaml
# Core algorithm settings
gamma: 0.95                      # Discount factor (unused in current version)
similarity_threshold: 0.6       # Default search threshold
max_experiences_per_search: 5   # Max results per search

# Embedding settings
vector_dim: 1536                # Embedding dimension
cache_embeddings: true          # Cache embeddings to disk

# Storage limits
max_experiences: 10000          # Max experiences per project
experience_ttl_days: 90         # Auto-delete old experiences

# Evaluation settings
use_llm_evaluation: false       # Use LLM for scoring (more accurate, higher cost)
evaluation_model: "gpt-4o-mini" # Model for LLM evaluation
```

## Troubleshooting

### "No module named 'faiss'"

```bash
pip install faiss-cpu
# or for GPU:
pip install faiss-gpu
```

### "No similar experiences found"

1. Check if JitRL is initialized: `jitrl.py stats`
2. Lower similarity threshold: `-t 0.4`
3. Build more experiences first

### Slow embedding performance

1. Enable caching: `cache_embeddings: true`
2. Use fallback embedding (no OpenAI): works offline but less accurate

### Memory issues with large experience stores

1. Reduce `max_experiences`
2. Enable experience TTL cleanup
3. Use `jitrl.py clear --confirm` to reset

## Theory: JitRL Paper

JitRL is based on the paper "JitRL: Just-in-Time Reinforcement Learning for LLM Alignment" (arXiv:2601.18510).

Key innovations:
1. **No Gradient Updates**: Uses logit adjustment instead of fine-tuning
2. **Experience Replay**: Stores and retrieves relevant past experiences
3. **Advantage-Weighted Sampling**: Prioritizes successful action patterns

This implementation adapts JitRL for Claude Code by:
- Using Faiss for fast similarity search
- Integrating with Claude Code Hooks
- Providing context injection for in-context learning

## See Also

- [SKILL.md](SKILL.md) - Main skill documentation
- [JitRL Paper](https://arxiv.org/abs/2601.18510) - Original research
- [Claude Code Hooks](https://code.claude.com/docs/en/hooks) - Hook system docs
- [Faiss Documentation](https://github.com/facebookresearch/faiss/wiki)
