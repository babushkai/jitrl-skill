#!/usr/bin/env python3
"""
JitRL SessionEnd Hook

Evaluates the session and stores experiences when Claude Code session ends.
This is where learning happens - experiences are scored and persisted.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from jitrl import ExperienceStore, JitRLConfig

SESSION_FILE = Path.home() / '.claude-jitrl' / 'cache' / 'current_session.json'


def evaluate_session(interactions: list, end_reason: str) -> dict:
    """
    Evaluate the session to determine success/failure.

    Uses heuristics based on:
    - How the session ended
    - Tools used
    - Error patterns
    """
    score = 0
    success = False
    feedback = ''

    # Evaluate based on end reason
    if end_reason == 'prompt_input_exit':
        # User exited normally - likely satisfied
        score += 3
        success = True
    elif end_reason == 'clear':
        # Context cleared - task probably complete
        score += 2
        success = True
    elif end_reason == 'logout':
        # Normal logout
        score += 1

    # Analyze interactions
    error_count = 0
    edit_count = 0

    for interaction in interactions:
        state = interaction.get('state', {})
        action = interaction.get('action', {})

        # Count errors
        if state.get('error_context'):
            error_count += 1

        # Count productive actions
        tool = action.get('tool_name', '')
        if tool in ('Edit', 'Write'):
            edit_count += 1
            score += 1
        elif tool == 'Bash':
            # Check if it's a test run
            summary = action.get('summary', '').lower()
            if 'test' in summary:
                score += 2

    # Adjust for error resolution
    if error_count > 0 and edit_count > 0:
        # Errors were addressed
        score += 2
        feedback = 'Resolved errors'
    elif error_count > 0 and edit_count == 0:
        # Errors not addressed
        score -= 1
        feedback = 'Errors not fully resolved'

    # Determine overall success
    if not success:
        success = score > 0

    return {
        'score': max(-5, min(10, score)),
        'success': success,
        'feedback': feedback,
        'follow_up': not success,
        'error_count': error_count,
        'edit_count': edit_count
    }


def main():
    """Main hook handler."""
    try:
        hook_data = json.loads(os.environ.get('CLAUDE_HOOK_DATA', '{}'))

        cwd = hook_data.get('cwd', os.getcwd())
        reason = hook_data.get('reason', 'unknown')

        # Check for session data
        if not SESSION_FILE.exists():
            return

        with open(SESSION_FILE, 'r') as f:
            session_data = json.load(f)

        interactions = session_data.get('interactions', [])

        if not interactions:
            SESSION_FILE.unlink(missing_ok=True)
            return

        # Initialize store
        base_dir = Path.home() / '.claude-jitrl'
        config = JitRLConfig(base_dir)
        store = ExperienceStore(cwd, config)

        # Evaluate session
        evaluation = evaluate_session(interactions, reason)

        # Store each interaction as an experience
        stored_count = 0

        for i, interaction in enumerate(interactions):
            state = interaction.get('state', {})
            action = interaction.get('action', {})

            # Skip if no meaningful action
            if action.get('tool_name') == 'unknown':
                continue

            # Position-weighted scoring
            # Later actions in session are more likely to be the resolution
            position_weight = (i + 1) / len(interactions)

            outcome = {
                'success': evaluation['success'],
                'user_feedback': evaluation['feedback'],
                'follow_up_needed': evaluation['follow_up'],
                'session_score': evaluation['score'] * position_weight
            }

            try:
                store.add(state, action, outcome)
                stored_count += 1
            except Exception as e:
                print(f"Failed to store experience: {e}", file=sys.stderr)

        # Clean up session file
        SESSION_FILE.unlink(missing_ok=True)

        if stored_count > 0:
            print(f"✅ JitRL: Stored {stored_count} experiences (session score: {evaluation['score']:+d})")

    except Exception as e:
        print(f"<!-- JitRL session-end error: {e} -->", file=sys.stderr)
        # Clean up on error too
        SESSION_FILE.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
