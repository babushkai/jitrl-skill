#!/usr/bin/env python3
"""
JitRL: Just-in-Time Reinforcement Learning for Claude Code

A persistent experience memory system that learns from your coding sessions.
Based on the JitRL paper: https://arxiv.org/abs/2501.18510

Key concepts from the paper:
- Policy Triplets: <state, action, outcome>
- Dual vector search (history + state)
- Jaccard N-gram similarity
- Advantage normalization with episode weighting
- LLM-based step scoring

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
from typing import List, Dict, Any, Optional, Tuple
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
        # Core algorithm (from paper)
        'gamma': 0.95,                    # Discount factor for future rewards
        'similarity_threshold': 0.7,      # Base similarity threshold
        'history_weight': 0.25,           # Weight for history similarity
        'state_weight': 0.75,             # Weight for state similarity
        'jaccard_history_weight': 0.3,    # Jaccard weight for history
        'jaccard_state_weight': 0.7,      # Jaccard weight for current state
        'ngram_size': 4,                  # N-gram size for Jaccard

        # Episode weighting (from paper)
        'max_episode_weight': 1.5,        # Max weight at 50 episodes
        'episode_weight_scale': 50,       # Episodes to reach max weight

        # Retrieval settings
        'max_experiences_per_search': 5,
        'dynamic_threshold_decay': 0.1,   # How much threshold decreases
        'max_steps_for_threshold': 20,    # Steps at which decay maxes out

        # Storage
        'cache_embeddings': True,
        'max_experiences': 10000,
        'experience_ttl_days': 90,

        # LLM evaluation
        'use_llm_evaluation': False,
        'evaluation_model': 'gpt-4o-mini',

        # Embeddings
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

        cache_key = hashlib.md5(text.encode()).hexdigest()
        cache_path = self.cache_dir / f'{cache_key}.npy'

        if use_cache and cache_path.exists():
            return np.load(cache_path)

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
    JitRL Experience Storage using dual Faiss indexes.

    Implements the paper's approach:
    - Dual vector search (history + state)
    - Jaccard N-gram similarity
    - Discounted future rewards
    - Normalized advantages with episode weighting
    """

    def __init__(self, project_path: str, config: JitRLConfig):
        self.project_hash = hashlib.md5(project_path.encode()).hexdigest()[:12]
        self.config = config
        self.vector_dim = config.get('vector_dim', 1536)
        self.gamma = config.get('gamma', 0.95)
        self.episode_count = 0

        # Paths
        self.base_dir = Path.home() / '.claude-jitrl'
        self.project_dir = self.base_dir / 'experiences' / self.project_hash
        self.index_dir = self.base_dir / 'indexes' / self.project_hash

        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.episodes_path = self.project_dir / 'episodes.jsonl'
        self.metadata_path = self.project_dir / 'step_metadata.pkl'
        self.history_index_path = self.index_dir / 'history_vectors.index'
        self.state_index_path = self.index_dir / 'state_vectors.index'

        # Initialize
        self._init_dual_indexes()
        self._load_metadata()
        self._count_episodes()

        # Embedder
        self.embedder = Embedder(self.base_dir / 'cache', self.vector_dim)

    def _init_dual_indexes(self):
        """Initialize dual Faiss indexes (history + state) per the paper."""
        if not FAISS_AVAILABLE:
            self.history_index = None
            self.state_index = None
            return

        # History index (trajectory context)
        if self.history_index_path.exists():
            self.history_index = faiss.read_index(str(self.history_index_path))
        else:
            self.history_index = faiss.IndexFlatIP(self.vector_dim)

        # State index (current state)
        if self.state_index_path.exists():
            self.state_index = faiss.read_index(str(self.state_index_path))
        else:
            self.state_index = faiss.IndexFlatIP(self.vector_dim)

    def _load_metadata(self):
        """Load step metadata."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = []

    def _count_episodes(self):
        """Count existing episodes."""
        if self.episodes_path.exists():
            with open(self.episodes_path, 'r') as f:
                self.episode_count = sum(1 for _ in f)

    def _save(self):
        """Save indexes and metadata."""
        if self.history_index is not None:
            faiss.write_index(self.history_index, str(self.history_index_path))
        if self.state_index is not None:
            faiss.write_index(self.state_index, str(self.state_index_path))
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for Jaccard similarity."""
        return [t for t in (text or '').lower().replace('\n', ' ').split()
                if t.isalnum()]

    def _get_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Convert tokens to n-grams."""
        if len(tokens) < n:
            return tokens
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def _jaccard_similarity(self, a_tokens: List[str], b_tokens: List[str],
                           ngram: int = 4) -> float:
        """
        Compute Jaccard similarity using n-grams (per paper).
        Uses Counter for multiset intersection/union.
        """
        a_ngrams = self._get_ngrams(a_tokens, ngram)
        b_ngrams = self._get_ngrams(b_tokens, ngram)

        counter_a = Counter(a_ngrams)
        counter_b = Counter(b_ngrams)

        if not counter_a or not counter_b:
            return 0.0

        inter = sum((counter_a & counter_b).values())
        union = sum((counter_a | counter_b).values())

        return inter / union if union > 0 else 0.0

    def _compute_discounted_return(self, future_rewards: List[float]) -> float:
        """
        Compute discounted return from future rewards (per paper).
        R = Σ γ^t * r_t
        """
        discounted = 0.0
        for t, reward in enumerate(future_rewards):
            discounted += (self.gamma ** t) * reward
        return discounted

    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity."""
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)

    def add(self, state: Dict, action: Dict, outcome: Dict,
            trajectory_context: str = "", current_env_info: str = "",
            future_rewards: List[float] = None) -> Dict:
        """
        Add an experience with proper metadata for the paper's algorithm.

        Args:
            state: Current state context
            action: Action taken
            outcome: Result of action
            trajectory_context: Summary of history (for history vector)
            current_env_info: Current environment info (for state vector)
            future_rewards: List of future rewards for discounting
        """
        score = self._calculate_score(outcome)

        # Compute discounted return if future rewards provided
        discounted_return = 0.0
        if future_rewards:
            discounted_return = self._compute_discounted_return(future_rewards)

        experience = {
            'timestamp': datetime.now().isoformat(),
            'episode_number': self.episode_count,
            'state': state,
            'action': action,
            'outcome': outcome,
            'score': score,
            'trajectory_context': trajectory_context,
            'current_env_info': current_env_info,
            'future_rewards': future_rewards or [score],
            'discounted_return': discounted_return if discounted_return else score
        }

        # Append to JSONL
        with open(self.episodes_path, 'a') as f:
            f.write(json.dumps(experience, ensure_ascii=False) + '\n')

        # Add to dual vector indexes
        if self.history_index is not None and self.state_index is not None:
            # Create text for embeddings
            history_text = trajectory_context or json.dumps(state, ensure_ascii=False)
            state_text = current_env_info or f"{action.get('tool_name', '')}: {action.get('summary', '')}"

            history_vec = self._normalize_vector(self.embedder.embed(history_text))
            state_vec = self._normalize_vector(self.embedder.embed(state_text))

            self.history_index.add(history_vec.reshape(1, -1).astype('float32'))
            self.state_index.add(state_vec.reshape(1, -1).astype('float32'))

            self.metadata.append({
                'experience': experience,
                'history_tokens': self._tokenize(history_text),
                'state_tokens': self._tokenize(state_text),
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

    def search(self, query: str, k: int = 5, step_count: int = 0) -> List[Dict]:
        """
        Search for similar experiences using dual vector + Jaccard similarity.

        Implements the paper's retrieval algorithm:
        1. Dual vector search (history 0.25 + state 0.75)
        2. Jaccard N-gram similarity (history 0.3 + state 0.7)
        3. Dynamic threshold based on step count
        """
        if self.history_index is None or self.state_index is None:
            return []
        if self.history_index.ntotal == 0:
            return []

        # Dynamic threshold (decreases with step count per paper)
        max_steps = self.config.get('max_steps_for_threshold', 20)
        threshold_decay = self.config.get('dynamic_threshold_decay', 0.1)
        base_threshold = self.config.get('similarity_threshold', 0.7)

        if step_count >= max_steps:
            dynamic_threshold = base_threshold - threshold_decay
        else:
            decay = threshold_decay * (step_count / max_steps)
            dynamic_threshold = base_threshold - decay

        # Embed query for both indexes
        query_vec = self._normalize_vector(self.embedder.embed(query))
        query_tokens = self._tokenize(query)

        # Search both indexes
        recall_size = min(max(k * 10, 100), self.history_index.ntotal)

        history_scores, history_indices = self.history_index.search(
            query_vec.reshape(1, -1).astype('float32'), recall_size
        )
        state_scores, state_indices = self.state_index.search(
            query_vec.reshape(1, -1).astype('float32'), recall_size
        )

        # Combine scores (per paper: 0.25 history + 0.75 state)
        history_weight = self.config.get('history_weight', 0.25)
        state_weight = self.config.get('state_weight', 0.75)
        jaccard_history_weight = self.config.get('jaccard_history_weight', 0.3)
        jaccard_state_weight = self.config.get('jaccard_state_weight', 0.7)
        ngram_size = self.config.get('ngram_size', 4)

        combined_scores = {}

        for score, idx in zip(history_scores[0], history_indices[0]):
            if idx < len(self.metadata):
                combined_scores[int(idx)] = {'history_vec': float(score), 'state_vec': 0.0}

        for score, idx in zip(state_scores[0], state_indices[0]):
            if idx < len(self.metadata):
                if int(idx) not in combined_scores:
                    combined_scores[int(idx)] = {'history_vec': 0.0, 'state_vec': float(score)}
                else:
                    combined_scores[int(idx)]['state_vec'] = float(score)

        results = []
        for idx, scores_dict in combined_scores.items():
            meta = self.metadata[idx]
            exp = meta['experience']

            # Vector similarity (weighted)
            vec_sim = (history_weight * scores_dict['history_vec'] +
                      state_weight * scores_dict['state_vec'])

            # Jaccard N-gram similarity (per paper)
            jaccard_history = self._jaccard_similarity(
                query_tokens, meta.get('history_tokens', []), ngram=1
            )
            jaccard_state = self._jaccard_similarity(
                query_tokens, meta.get('state_tokens', []), ngram=ngram_size
            )
            jaccard_sim = (jaccard_history_weight * jaccard_history +
                         jaccard_state_weight * jaccard_state)

            # Final similarity (use Jaccard as primary per paper implementation)
            similarity = jaccard_sim

            if similarity < dynamic_threshold:
                continue

            result = {
                'experience': exp,
                'similarity': similarity,
                'vector_similarity': vec_sim,
                'jaccard_similarity': jaccard_sim,
                'history_sim': scores_dict['history_vec'],
                'state_sim': scores_dict['state_vec'],
                'discounted_return': exp.get('discounted_return', exp['score'])
            }
            results.append(result)

        # Sort by similarity, then by discounted return (per paper)
        results.sort(key=lambda x: (x['similarity'], x['discounted_return']), reverse=True)

        return results[:k]

    def get_advantages(self, similar_experiences: List[Dict]) -> Dict[str, float]:
        """
        Calculate normalized advantages with episode weighting (per paper).

        Formula:
        1. avg_reward[action] = mean of discounted returns for action
        2. baseline = mean of all discounted returns
        3. advantage[action] = avg_reward[action] - baseline
        4. normalized = advantage / max(positive advantages)
        5. weighted = normalized * episode_weight

        Episode weight = min(1.0 + (episode / 50) * 0.5, 1.5)
        """
        if not similar_experiences:
            return {}

        # Aggregate discounted returns per action
        action_returns = {}
        for exp_data in similar_experiences:
            exp = exp_data['experience']
            action_type = exp['action'].get('tool_name', 'unknown')
            discounted_return = exp_data.get('discounted_return', exp['score'])

            if action_type not in action_returns:
                action_returns[action_type] = []
            action_returns[action_type].append(discounted_return)

        # Calculate average returns
        action_avg = {k: sum(v) / len(v) for k, v in action_returns.items()}

        # Calculate baseline (overall average)
        all_returns = [r for returns in action_returns.values() for r in returns]
        baseline = sum(all_returns) / len(all_returns) if all_returns else 0

        # Raw advantages
        raw_advantages = {k: v - baseline for k, v in action_avg.items()}

        # Normalize by max positive (per paper)
        positive_advs = [a for a in raw_advantages.values() if a > 0]
        if positive_advs:
            max_positive = max(positive_advs)
            normalized = {k: v / max_positive for k, v in raw_advantages.items()}
        else:
            # If no positive, normalize by max negative absolute
            max_neg = max(abs(min(raw_advantages.values())), 1e-8)
            normalized = {k: v / max_neg for k, v in raw_advantages.items()}

        # Episode weighting (per paper: 1.0 → 1.5 over 50 episodes)
        max_weight = self.config.get('max_episode_weight', 1.5)
        scale = self.config.get('episode_weight_scale', 50)
        episode_weight = min(1.0 + (self.episode_count / scale) * 0.5, max_weight)

        weighted_advantages = {k: v * episode_weight for k, v in normalized.items()}

        return weighted_advantages

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the experience store."""
        stats = {
            'project_hash': self.project_hash,
            'total_experiences': self.history_index.ntotal if self.history_index else 0,
            'episode_count': self.episode_count,
            'history_index_path': str(self.history_index_path),
            'state_index_path': str(self.state_index_path),
            'episodes_path': str(self.episodes_path),
            'metadata_count': len(self.metadata)
        }

        if self.metadata:
            scores = [m['experience']['score'] for m in self.metadata]
            stats['avg_score'] = sum(scores) / len(scores)
            stats['positive_count'] = sum(1 for s in scores if s > 0)
            stats['negative_count'] = sum(1 for s in scores if s < 0)

            action_types = Counter(
                m['experience']['action'].get('tool_name', 'unknown')
                for m in self.metadata
            )
            stats['action_types'] = dict(action_types.most_common(10))

        return stats

    def clear(self):
        """Clear all experiences for this project."""
        if self.history_index is not None:
            self.history_index.reset()
        if self.state_index is not None:
            self.state_index.reset()
        self.metadata = []
        self.episode_count = 0
        self._save()

        if self.episodes_path.exists():
            self.episodes_path.unlink()

    def export(self, filepath: str):
        """Export all experiences to JSON."""
        experiences = [m['experience'] for m in self.metadata]
        with open(filepath, 'w') as f:
            json.dump(experiences, f, indent=2, ensure_ascii=False)


def generate_injection(experiences: List[Dict], advantages: Dict[str, float]) -> str:
    """Generate context injection text for Claude (our approximation of logit adjustment)."""
    if not experiences:
        return ""

    lines = ["## 💡 Past Experience Insights\n"]

    # Success patterns (high similarity + positive score)
    successes = [e for e in experiences if e['experience']['score'] > 0]
    if successes:
        lines.append("### ✅ Success Patterns")
        for exp_data in successes[:3]:
            exp = exp_data['experience']
            action = exp['action']
            tool = action.get('tool_name', 'action')
            summary = action.get('summary', '')[:60]
            sim = exp_data['similarity']
            score = exp['score']
            lines.append(f"- **{tool}**: {summary}")
            lines.append(f"  - Similarity: {sim:.2f}, Score: {score:+d}")

    # Failure patterns
    failures = [e for e in experiences if e['experience']['score'] < 0]
    if failures:
        lines.append("\n### ⚠️ Failure Patterns (Avoid)")
        for exp_data in failures[:2]:
            exp = exp_data['experience']
            action = exp['action']
            outcome = exp['outcome']
            tool = action.get('tool_name', 'action')
            error = outcome.get('error_summary', action.get('summary', ''))[:60]
            lines.append(f"- **{tool}**: {error}")

    # Advantages (normalized + episode weighted)
    if advantages:
        lines.append("\n### 📊 Recommended Approach (Advantage-Weighted)")
        sorted_adv = sorted(advantages.items(), key=lambda x: x[1], reverse=True)
        for tool, adv in sorted_adv[:4]:
            if adv > 0.5:
                emoji = "🟢"
            elif adv > 0:
                emoji = "👍"
            elif adv > -0.5:
                emoji = "👎"
            else:
                emoji = "🔴"
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
    subparsers.add_parser('init', help='Initialize JitRL for current project')

    # stats
    subparsers.add_parser('stats', help='Show experience statistics')

    # search
    search_parser = subparsers.add_parser('search', help='Search similar experiences')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('-k', type=int, default=5, help='Number of results')
    search_parser.add_argument('--step', type=int, default=0, help='Current step (for dynamic threshold)')

    # inject
    inject_parser = subparsers.add_parser('inject', help='Get context injection for prompt')
    inject_parser.add_argument('context', help='Current context/prompt')
    inject_parser.add_argument('--step', type=int, default=0, help='Current step count')

    # store
    store_parser = subparsers.add_parser('store', help='Store new experience')
    store_parser.add_argument('experience', help='Experience JSON')

    # clear
    clear_parser = subparsers.add_parser('clear', help='Clear project memory')
    clear_parser.add_argument('--confirm', action='store_true', required=True)

    # export
    export_parser = subparsers.add_parser('export', help='Export experiences')
    export_parser.add_argument('filepath', help='Output file path')

    args = parser.parse_args()

    project_path = os.getcwd()
    base_dir = Path.home() / '.claude-jitrl'
    base_dir.mkdir(parents=True, exist_ok=True)

    config = JitRLConfig(base_dir)
    store = ExperienceStore(project_path, config)

    if args.command == 'init':
        print(f"✅ JitRL initialized for: {project_path}")
        print(f"   Project hash: {store.project_hash}")
        print(f"   Episodes: {store.episode_count}")
        print(f"   Experiences: {store.history_index.ntotal if store.history_index else 0}")
        print(f"   Using dual vector indexes (history + state)")
        print(f"   Data dir: {store.project_dir}")

    elif args.command == 'stats':
        stats = store.get_stats()
        print("📊 JitRL Statistics")
        print(f"   Project: {stats['project_hash']}")
        print(f"   Episodes: {stats['episode_count']}")
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
        results = store.search(args.query, k=args.k, step_count=args.step)
        print(f"🔍 Found {len(results)} similar experiences:\n")

        for i, r in enumerate(results):
            exp = r['experience']
            print(f"{i+1}. [{exp['score']:+d}] {exp['action'].get('tool_name', 'action')}")
            print(f"   Similarity: {r['similarity']:.2f} (vec: {r['vector_similarity']:.2f}, jac: {r['jaccard_similarity']:.2f})")
            print(f"   Discounted Return: {r['discounted_return']:.2f}")
            print(f"   {exp['action'].get('summary', '')[:80]}")
            print()

    elif args.command == 'inject':
        results = store.search(args.context, k=config.get('max_experiences_per_search', 5),
                              step_count=args.step)
        advantages = store.get_advantages(results)
        injection = generate_injection(results, advantages)
        print(injection)

    elif args.command == 'store':
        try:
            data = json.loads(args.experience)
            exp = store.add(
                state=data.get('state', {}),
                action=data.get('action', {}),
                outcome=data.get('outcome', {}),
                trajectory_context=data.get('trajectory_context', ''),
                current_env_info=data.get('current_env_info', ''),
                future_rewards=data.get('future_rewards', None)
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
