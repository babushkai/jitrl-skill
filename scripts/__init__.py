"""
JitRL: Just-In-Time Reinforcement Learning for Claude Code

A persistent experience memory system that learns from your coding sessions.
"""

from .jitrl import (
    ExperienceStore,
    JitRLConfig,
    Embedder,
    generate_injection,
)

__version__ = '0.1.0'
__all__ = [
    'ExperienceStore',
    'JitRLConfig',
    'Embedder',
    'generate_injection',
]
