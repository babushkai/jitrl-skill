#!/usr/bin/env python3
"""
Basic usage example for JitRL.

This demonstrates how to:
1. Initialize the experience store
2. Add experiences manually
3. Search for similar experiences
4. Calculate advantages
"""

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from jitrl import ExperienceStore, JitRLConfig, generate_injection


def main():
    # Initialize configuration
    base_dir = Path.home() / '.claude-jitrl'
    config = JitRLConfig(base_dir)

    # Create store for current directory
    store = ExperienceStore('.', config)

    print("📊 Initial Statistics:")
    print_stats(store)

    # Add some example experiences
    print("\n📝 Adding example experiences...")

    # Successful Edit experience
    store.add(
        state={
            'files_read': ['src/components/Button.tsx'],
            'error_context': 'Property onClick does not exist on type',
            'user_goal': 'fix TypeScript error'
        },
        action={
            'tool_name': 'Edit',
            'summary': 'Added onClick prop to Button interface',
            'file_changed': 'src/components/Button.tsx'
        },
        outcome={
            'success': True,
            'user_feedback': 'perfect',
            'follow_up_needed': False
        }
    )
    print("  ✅ Added successful Edit experience")

    # Failed Write experience
    store.add(
        state={
            'files_read': ['src/types/index.ts'],
            'error_context': 'Module not found',
            'user_goal': 'fix import error'
        },
        action={
            'tool_name': 'Write',
            'summary': 'Created new type definition file',
            'file_changed': 'src/types/button.d.ts'
        },
        outcome={
            'success': False,
            'user_feedback': 'wrong approach',
            'follow_up_needed': True
        }
    )
    print("  ❌ Added failed Write experience")

    # Another successful Edit
    store.add(
        state={
            'files_read': ['src/api/client.ts'],
            'error_context': 'async function without await',
            'user_goal': 'fix async/await issue'
        },
        action={
            'tool_name': 'Edit',
            'summary': 'Added await to async call',
            'file_changed': 'src/api/client.ts'
        },
        outcome={
            'success': True,
            'user_feedback': 'good',
            'follow_up_needed': False
        }
    )
    print("  ✅ Added successful Edit experience")

    print("\n📊 Updated Statistics:")
    print_stats(store)

    # Search for similar experiences
    print("\n🔍 Searching for 'TypeScript type error'...")
    results = store.search('TypeScript type error', k=3, threshold=0.3)

    print(f"   Found {len(results)} results:")
    for i, r in enumerate(results):
        exp = r['experience']
        print(f"   {i+1}. [{exp['score']:+d}] {exp['action']['tool_name']}: {exp['action']['summary']}")
        print(f"      Similarity: {r['similarity']:.2f}")

    # Calculate advantages
    print("\n📈 Action Advantages:")
    advantages = store.get_advantages(results)
    for action, adv in sorted(advantages.items(), key=lambda x: -x[1]):
        emoji = "👍" if adv > 0 else "👎" if adv < 0 else "➖"
        print(f"   {emoji} {action}: {adv:+.2f}")

    # Generate context injection
    print("\n💉 Context Injection Preview:")
    injection = generate_injection(results, advantages)
    print(injection)


def print_stats(store):
    """Print store statistics."""
    stats = store.get_stats()
    print(f"   Project hash: {stats['project_hash']}")
    print(f"   Total experiences: {stats['total_experiences']}")
    if stats['total_experiences'] > 0:
        print(f"   Positive: {stats.get('positive_count', 0)}")
        print(f"   Negative: {stats.get('negative_count', 0)}")


if __name__ == '__main__':
    main()
