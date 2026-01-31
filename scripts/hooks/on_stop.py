#!/usr/bin/env python3
"""
JitRL Stop Hook

Captures interaction details when Claude finishes responding.
Stores intermediate data for session-end evaluation.
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime

# Session tracking file
SESSION_FILE = Path.home() / '.claude-jitrl' / 'cache' / 'current_session.json'


def extract_tool_usage(content: str) -> dict:
    """Extract tool usage information from transcript content."""
    tools_used = []

    # Common tool patterns
    tool_patterns = [
        (r'<Read>', 'Read'),
        (r'<Write>', 'Write'),
        (r'<Edit>', 'Edit'),
        (r'<Bash>', 'Bash'),
        (r'<Glob>', 'Glob'),
        (r'<Grep>', 'Grep'),
        (r'<Task>', 'Task'),
    ]

    for pattern, tool_name in tool_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            tools_used.append(tool_name)

    # Determine primary tool
    primary_tool = tools_used[0] if tools_used else 'unknown'

    return {
        'tool_name': primary_tool,
        'tools_used': list(set(tools_used)),
        'summary': extract_summary(content)
    }


def extract_summary(content: str) -> str:
    """Extract a brief summary of what was done."""
    # Look for common action indicators
    if 'error' in content.lower():
        return 'Addressed an error'
    if 'test' in content.lower():
        return 'Worked with tests'
    if 'fix' in content.lower():
        return 'Applied a fix'
    if 'create' in content.lower() or 'add' in content.lower():
        return 'Added new content'
    if 'update' in content.lower() or 'change' in content.lower():
        return 'Made updates'
    if 'delete' in content.lower() or 'remove' in content.lower():
        return 'Removed content'

    return 'Completed task'


def extract_context(content: str) -> dict:
    """Extract context information from content."""
    # Extract file paths mentioned
    file_pattern = r'[\'"`]([^\'"`]+\.(py|js|ts|tsx|jsx|md|json|yaml|yml|sh|go|rs|rb))[\'"`]'
    files = list(set(re.findall(file_pattern, content)))
    files = [f[0] for f in files][:10]  # Limit to 10 files

    # Look for error indicators
    error_context = None
    error_patterns = [
        r'(Error|Exception|TypeError|ValueError|SyntaxError)[:\s]([^\n]+)',
        r'(failed|error|crash)[:\s]([^\n]+)',
    ]

    for pattern in error_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            error_context = match.group(0)[:200]
            break

    return {
        'files_mentioned': files,
        'error_context': error_context,
        'content_length': len(content)
    }


def main():
    """Main hook handler."""
    try:
        hook_data = json.loads(os.environ.get('CLAUDE_HOOK_DATA', '{}'))

        cwd = hook_data.get('cwd', os.getcwd())
        transcript_path = hook_data.get('transcript_path', '')

        # Read recent transcript content
        content = ''
        if transcript_path and Path(transcript_path).exists():
            try:
                with open(transcript_path, 'r') as f:
                    # Read last portion of transcript
                    f.seek(0, 2)  # End of file
                    size = f.tell()
                    f.seek(max(0, size - 10000))  # Last 10KB
                    content = f.read()
            except Exception:
                pass

        # Extract information
        action = extract_tool_usage(content)
        state = extract_context(content)
        state['cwd'] = cwd

        # Create interaction record
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'state': state,
            'action': action,
            'raw_content_sample': content[-2000:] if content else ''
        }

        # Load or create session data
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)

        session_data = {'interactions': [], 'cwd': cwd}
        if SESSION_FILE.exists():
            try:
                with open(SESSION_FILE, 'r') as f:
                    session_data = json.load(f)
            except Exception:
                pass

        # Append interaction
        session_data['interactions'].append(interaction)
        session_data['last_updated'] = datetime.now().isoformat()

        # Save session data
        with open(SESSION_FILE, 'w') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"<!-- JitRL stop error: {e} -->", file=sys.stderr)


if __name__ == '__main__':
    main()
