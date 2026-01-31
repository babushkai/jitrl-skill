---
description: Search for similar past experiences in JitRL memory
argument-hint: [query]
disable-model-invocation: true
---

Search for similar experiences using the provided query:

```bash
python3 "$CLAUDE_PLUGIN_DIR/scripts/jitrl.py" search "$ARGUMENTS" -k 5
```

Display the results showing:
- Similarity score
- Experience score (positive = success, negative = failure)
- Action type and summary
- What can be learned from each experience
