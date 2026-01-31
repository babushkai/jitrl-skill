#!/usr/bin/env python3
"""
JitRL: Just-In-Time Reinforcement Learning for Claude Code

A persistent experience memory system that learns from your coding sessions.
Based on the JitRL paper: https://arxiv.org/abs/2601.18510

Usage:
    python3 jitrl.py init              # Initialize for current project
    python3 jitrl.py stats             # Show experience statistics
    python3 jitrl.py search "query"    # Search similar experiences
    python3 jitrl.py inject "context"  # Get context injection for prompt
    python3 jitrl.py store <json>      # Store new experience
    python3 jitrl.py clear --confirm   # Clear project memory
    python3 jitrl.py export <file>     # Export experiences
"""

import os
import sys
import json
import hashlib
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not installed. Run: pip install faiss-cpu", file=sys.stderr)


class JitRLConfig:
    """Configuration management for JitRL."""

    DEFAULT_CONFIG = {
        'gamma': 0.95,
        'similarity_threshold': 0.6,
        'max_experiences_per_search': 5,
        'cache_embeddings': True,
        'max_experiences': 10000,
        'experience_ttl_days': 90,
        'use_llm_evaluation': False,
        'evaluation_model': 'gpt-4o-mini',
        'vector_dim': 1536,
    }

    def __init__(self, base_dir: Path):
        self.config_path = base_dir / 'config.yaml'
        self.config = self.DEFAULT_CONFIG.copy()
        self._load()

    def _load(self):
        if self.config_path.exists():
            try:
                import yaml
                with open(self.config_path) as f:
                    loaded = yaml.safe_load(f)
                    if loaded:
                        self.config.update(loaded)
            except ImportError:
                # Fallback: simple parsing
                pass

    def get(self, key: str, default=None):
        return self.config.get(key, default)


class Embedder:
    """Text embedding with caching."""

    def __init__(self, cache_dir: Path, vector_dim: int = 1536):
        self.cache_dir = cache_dir / 'embeddings'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dim = vector_dim
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except (ImportError, Exception):
                self._client = False
        return self._client

    def embed(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Get embedding for text."""
        if not text.strip():
            return np.zeros(self.vector_dim, dtype=np.float32)

        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        cache_path = self.cache_dir / f'{cache_key}.npy'

        if use_cache and cache_path.exists():
            return np.load(cache_path)

        # Try OpenAI
        if self.client:
            try:
                response = self.client.embeddings.create(
                    model='text-embedding-3-small',
                    input=text[:8000]
                )
                embedding = np.array(response.data[0].embedding, dtype=np.float32)
                np.save(cache_path, embedding)
                return embedding
            except Exception as e:
                print(f"Embedding API error: {e}", file=sys.stderr)

        # Fallback: hash-based embedding
        return self._fallback_embed(text)

    def _fallback_embed(self, text: str) -> np.ndarray:
        """Simple hash-based embedding fallback."""
        words = text.lower().split()
        embedding = np.zeros(self.vector_dim, dtype=np.float32)

        for i, word in enumerate(words[:500]):
            idx = hash(word) % self.vector_dim
            embedding[idx] += 1.0 / (i + 1)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding


class ExperienceStore:
    """
    JitRL Experience Storage using Faiss.

    Stores Policy Triplets: <state, action, outcome>
    """

    def __init__(self, project_path: str, config: JitRLConfig):
        self.project_hash = hashlib.md5(project_path.encode()).hexdigest()[:12]
        self.config = config
        self.vector_dim = config.get('vector_dim', 1536)

        # Paths
        self.base_dir = Path.home() / '.claude-jitrl'
        self.project_dir = self.base_dir / 'experiences' / self.project_hash
        self.index_dir = self.base_dir / 'indexes' / self.project_hash

        # Create directories
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.episodes_path = self.project_dir / 'episodes.jsonl'
        self.metadata_path = self.project_dir / 'step_metadata.pkl'
        self.index_path = self.index_dir / 'state_vectors.index'

        # Initialize
        self._init_index()
        self._load_metadata()

        # Embedder
        self.embedder = Embedder(self.base_dir / 'cache', self.vector_dim)

    def _init_index(self):
        """Initialize Faiss index."""
        if not FAISS_AVAILABLE:
            self.index = None
            return

        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatIP(self.vector_dim)

    def _load_metadata(self):
        """Load step metadata."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = []

    def _save(self):
        """Save index and metadata."""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def add(self, state: Dict, action: Dict, outcome: Dict) -> Dict:
        """
        Add an experience.

        Args:
            state: Context (files, errors, goals)
            action: What was done (tool, changes)
            outcome: Result (success, feedback)

        Returns:
            The stored experience
        """
        # Calculate score
        score = self._calculate_score(outcome)

        experience = {
            'timestamp': datetime.now().isoformat(),
            'state': state,
            'action': action,
            'outcome': outcome,
            'score': score
        }

        # Append to JSONL
        with open(self.episodes_path, 'a') as f:
            f.write(json.dumps(experience, ensure_ascii=False) + '\n')

        # Add to vector index
        if self.index is not None:
            state_text = json.dumps(state, ensure_ascii=False)
            embedding = self.embedder.embed(state_text)

            # Normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm

            self.index.add(embedding.reshape(1, -1).astype('float32'))
            self.metadata.append({
                'experience': experience,
                'idx': len(self.metadata)
            })

        self._save()
        return experience

    def _calculate_score(self, outcome: Dict) -> float:
        """Calculate experience score from outcome."""
        base = 5 if outcome.get('success', False) else -2

        feedback = outcome.get('user_feedback', '').lower()
        if 'perfect' in feedback or 'great' in feedback:
            base += 5
        elif 'good' in feedback:
            base += 2
        elif 'wrong' in feedback or 'bad' in feedback:
            base -= 3

        if not outcome.get('follow_up_needed', True):
            base += 2

        return max(-10, min(10, base))

    def search(self, query: str, k: int = 5, threshold: float = 0.6) -> List[Dict]:
        """
        Search for similar experiences.

        Args:
            query: Search query text
            k: Number of results
            threshold: Minimum similarity

        Returns:
            List of similar experiences with scores
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # Embed query
        embedding = self.embedder.embed(query)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        # Search
        scores, indices = self.index.search(
            embedding.reshape(1, -1).astype('float32'),
            min(k * 2, self.index.ntotal)  # Get extra for filtering
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity'] = float(score)
                results.append(result)

        # Sort by combined score (similarity + experience score)
        results.sort(key=lambda x: (
            x['similarity'] * 0.4 +
            (x['experience']['score'] + 10) / 20 * 0.6
        ), reverse=True)

        return results[:k]

    def get_advantages(self, experiences: List[Dict]) -> Dict[str, float]:
        """
        Calculate JitRL-style advantages for action types.

        Returns:
            Dict mapping action type to advantage value
        """
        if not experiences:
            return {}

        # Group scores by action type
        action_scores = {}
        for exp in experiences:
            action_type = exp['experience']['action'].get('tool_name', 'unknown')
            score = exp['experience']['score']

            if action_type not in action_scores:
                action_scores[action_type] = []
            action_scores[action_type].append(score)

        # Calculate averages
        action_avg = {k: sum(v) / len(v) for k, v in action_scores.items()}

        # Baseline (overall average)
        all_scores = [s for scores in action_scores.values() for s in scores]
        baseline = sum(all_scores) / len(all_scores) if all_scores else 0

        # Advantages
        return {k: v - baseline for k, v in action_avg.items()}

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the experience store."""
        stats = {
            'project_hash': self.project_hash,
            'total_experiences': self.index.ntotal if self.index else 0,
            'index_path': str(self.index_path),
            'episodes_path': str(self.episodes_path),
            'metadata_count': len(self.metadata)
        }

        # Score distribution
        if self.metadata:
            scores = [m['experience']['score'] for m in self.metadata]
            stats['avg_score'] = sum(scores) / len(scores)
            stats['positive_count'] = sum(1 for s in scores if s > 0)
            stats['negative_count'] = sum(1 for s in scores if s < 0)

            # Action type distribution
            action_types = Counter(
                m['experience']['action'].get('tool_name', 'unknown')
                for m in self.metadata
            )
            stats['action_types'] = dict(action_types.most_common(10))

        return stats

    def clear(self):
        """Clear all experiences for this project."""
        if self.index is not None:
            self.index.reset()
        self.metadata = []
        self._save()

        if self.episodes_path.exists():
            self.episodes_path.unlink()

    def export(self, filepath: str):
        """Export all experiences to a JSON file."""
        experiences = [m['experience'] for m in self.metadata]
        with open(filepath, 'w') as f:
            json.dump(experiences, f, indent=2, ensure_ascii=False)


def generate_injection(experiences: List[Dict], advantages: Dict[str, float]) -> str:
    """Generate context injection text for Claude."""
    if not experiences:
        return ""

    lines = ["## 💡 Past Experience Insights\n"]

    # Success patterns
    successes = [e for e in experiences if e['experience']['score'] > 0]
    if successes:
        lines.append("### ✅ Success Patterns")
        for exp in successes[:3]:
            action = exp['experience']['action']
            tool = action.get('tool_name', 'action')
            summary = action.get('summary', '')[:60]
            sim = exp['similarity']
            score = exp['experience']['score']
            lines.append(f"- **{tool}**: {summary}")
            lines.append(f"  - Similarity: {sim:.2f}, Score: {score:+d}")

    # Failure patterns
    failures = [e for e in experiences if e['experience']['score'] < 0]
    if failures:
        lines.append("\n### ⚠️ Failure Patterns (Avoid)")
        for exp in failures[:2]:
            action = exp['experience']['action']
            outcome = exp['experience']['outcome']
            tool = action.get('tool_name', 'action')
            error = outcome.get('error_summary', action.get('summary', ''))[:60]
            lines.append(f"- **{tool}**: {error}")

    # Advantages
    if advantages:
        lines.append("\n### 📊 Recommended Approach")
        sorted_adv = sorted(advantages.items(), key=lambda x: x[1], reverse=True)
        for tool, adv in sorted_adv[:4]:
            emoji = "👍" if adv > 0 else "👎" if adv < 0 else "➖"
            lines.append(f"- {emoji} **{tool}**: Advantage {adv:+.2f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='JitRL: Experience-based learning for Claude Code',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # init
    init_parser = subparsers.add_parser('init', help='Initialize JitRL for current project')

    # stats
    stats_parser = subparsers.add_parser('stats', help='Show experience statistics')

    # search
    search_parser = subparsers.add_parser('search', help='Search similar experiences')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('-k', type=int, default=5, help='Number of results')
    search_parser.add_argument('-t', '--threshold', type=float, default=0.6, help='Similarity threshold')

    # inject
    inject_parser = subparsers.add_parser('inject', help='Get context injection for prompt')
    inject_parser.add_argument('context', help='Current context/prompt')

    # store
    store_parser = subparsers.add_parser('store', help='Store new experience')
    store_parser.add_argument('experience', help='Experience JSON')

    # clear
    clear_parser = subparsers.add_parser('clear', help='Clear project memory')
    clear_parser.add_argument('--confirm', action='store_true', required=True, help='Confirm deletion')

    # export
    export_parser = subparsers.add_parser('export', help='Export experiences')
    export_parser.add_argument('filepath', help='Output file path')

    args = parser.parse_args()

    # Get current project path
    project_path = os.getcwd()
    base_dir = Path.home() / '.claude-jitrl'
    base_dir.mkdir(parents=True, exist_ok=True)

    config = JitRLConfig(base_dir)
    store = ExperienceStore(project_path, config)

    if args.command == 'init':
        print(f"✅ JitRL initialized for: {project_path}")
        print(f"   Project hash: {store.project_hash}")
        print(f"   Experiences: {store.index.ntotal if store.index else 0}")
        print(f"   Data dir: {store.project_dir}")

    elif args.command == 'stats':
        stats = store.get_stats()
        print("📊 JitRL Statistics")
        print(f"   Project: {stats['project_hash']}")
        print(f"   Experiences: {stats['total_experiences']}")
        if stats['total_experiences'] > 0:
            print(f"   Avg Score: {stats.get('avg_score', 0):.2f}")
            print(f"   Positive: {stats.get('positive_count', 0)}")
            print(f"   Negative: {stats.get('negative_count', 0)}")
            if stats.get('action_types'):
                print("   Top Actions:")
                for action, count in list(stats['action_types'].items())[:5]:
                    print(f"      - {action}: {count}")

    elif args.command == 'search':
        results = store.search(args.query, k=args.k, threshold=args.threshold)
        print(f"🔍 Found {len(results)} similar experiences:\n")

        for i, r in enumerate(results):
            exp = r['experience']
            print(f"{i+1}. [{exp['score']:+d}] {exp['action'].get('tool_name', 'action')}")
            print(f"   Similarity: {r['similarity']:.2f}")
            print(f"   {exp['action'].get('summary', '')[:80]}")
            print()

    elif args.command == 'inject':
        results = store.search(args.context, k=config.get('max_experiences_per_search', 5))
        advantages = store.get_advantages(results)
        injection = generate_injection(results, advantages)
        print(injection)

    elif args.command == 'store':
        try:
            data = json.loads(args.experience)
            exp = store.add(
                state=data.get('state', {}),
                action=data.get('action', {}),
                outcome=data.get('outcome', {})
            )
            print(f"✅ Stored experience (score: {exp['score']:+d})")
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == 'clear':
        store.clear()
        print(f"✅ Cleared all experiences for project {store.project_hash}")

    elif args.command == 'export':
        store.export(args.filepath)
        print(f"✅ Exported {len(store.metadata)} experiences to {args.filepath}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
