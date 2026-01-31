---
description: Initialize JitRL experience memory for the current project
disable-model-invocation: true
---

Initialize JitRL for the current project:

```bash
python3 "$CLAUDE_PLUGIN_DIR/scripts/jitrl.py" init
```

This creates a project-specific experience store that will:
- Automatically capture interactions during sessions
- Store success/failure outcomes
- Enable similarity search for future sessions

After initialization, JitRL works automatically in the background.
