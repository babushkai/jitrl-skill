#!/usr/bin/env python3
"""
JitRL UserPromptSubmit Hook

Injects relevant past experiences into Claude's context before it responds.
This enables experience-based learning without model fine-tuning.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from jitrl import ExperienceStore, JitRLConfig, generate_injection


def main():
    """Main hook handler."""
    try:
        # Parse hook data from environment
        hook_data = json.loads(os.environ.get('CLAUDE_HOOK_DATA', '{}'))

        cwd = hook_data.get('cwd', os.getcwd())
        prompt = hook_data.get('prompt', '')

        if not prompt:
            return

        # Initialize store
        base_dir = Path.home() / '.claude-jitrl'
        config = JitRLConfig(base_dir)
        store = ExperienceStore(cwd, config)

        # Search for similar experiences
        threshold = config.get('similarity_threshold', 0.6)
        max_results = config.get('max_experiences_per_search', 5)

        results = store.search(prompt, k=max_results, threshold=threshold)

        if not results:
            return

        # Calculate advantages
        advantages = store.get_advantages(results)

        # Generate and print injection
        injection = generate_injection(results, advantages)

        if injection:
            print(injection)

    except Exception as e:
        # Silently fail - don't interrupt Claude Code
        print(f"<!-- JitRL error: {e} -->", file=sys.stderr)


if __name__ == '__main__':
    main()
