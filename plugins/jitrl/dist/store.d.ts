/**
 * JitRL Experience Store
 *
 * Implements the JitRL (Just-in-Time Reinforcement Learning) algorithm from:
 * "JitRL: Just-in-Time Reinforcement Learning for Real-World LLM Agents"
 * https://arxiv.org/abs/2601.18510
 *
 * ## Core Algorithm
 *
 * JitRL is a training-free framework for test-time policy optimization that
 * maintains a dynamic, non-parametric memory of experiences and retrieves
 * relevant trajectories to estimate action advantages on-the-fly.
 *
 * ### Mathematical Foundation
 *
 * **KL-Constrained Policy Optimization Objective:**
 * $$
 * \pi^* = \arg\max_{\pi'} \left( \mathbb{E}_{a \sim \pi'}[A(s,a)] - \frac{1}{\beta} D_{KL}(\pi' \| \pi_\theta) \right)
 * $$
 *
 * **Closed-Form Solution (Theorem from paper):**
 * $$
 * \pi^*(a|s) \propto \pi_\theta(a|s) \exp(\beta \cdot A(s,a))
 * $$
 *
 * **Logit Space Implementation:**
 * $$
 * z'(s,a) = z(s,a) + \beta \cdot \hat{A}(s,a)
 * $$
 *
 * ### Key Components Implemented
 *
 * 1. **Experience Memory**: Stores $(s_i, a_i, G_i)$ triplets
 * 2. **Similarity-based Retrieval**: Jaccard + vector similarity
 * 3. **Value Estimation**: $\hat{V}(s)$ and $\hat{Q}(s,a)$ from neighbors
 * 4. **Advantage Calculation**: $\hat{A}(s,a) = \hat{Q}(s,a) - \hat{V}(s)$
 * 5. **Context Injection**: Approximates logit adjustment for Claude Code
 *
 * @see https://arxiv.org/abs/2601.18510
 */
export interface Experience {
    timestamp: string;
    episodeNumber: number;
    state: Record<string, unknown>;
    action: {
        tool_name?: string;
        summary?: string;
        parameters?: Record<string, unknown>;
    };
    outcome: {
        success?: boolean;
        error_summary?: string;
        user_feedback?: string;
    };
    score: number;
    trajectoryContext: string;
    currentEnvInfo: string;
    futureRewards: number[];
    discountedReturn: number;
}
export interface SearchResult {
    experience: Experience;
    similarity: number;
    vectorSimilarity: number;
    jaccardSimilarity: number;
    discountedReturn: number;
}
export declare class ExperienceStore {
    private baseDir;
    private projectDir;
    private indexDir;
    private historyIndex;
    private stateIndex;
    private metadata;
    episodeCount: number;
    /**
     * Algorithm hyperparameters from the JitRL paper.
     *
     * ### Discounted Return
     * $$
     * G_t = \sum_{u=t}^{T} \gamma^{u-t} r_u
     * $$
     */
    /** Discount factor $\gamma \in [0,1]$ for future rewards */
    private readonly gamma;
    /** Vector embedding dimension for HNSW index */
    private readonly vectorDim;
    /**
     * Dual-index similarity weights.
     *
     * Combined vector similarity:
     * $$
     * \text{sim}_{vec} = w_h \cdot \text{sim}_{history} + w_s \cdot \text{sim}_{state}
     * $$
     */
    private readonly historyWeight;
    private readonly stateWeight;
    /**
     * Jaccard similarity weights for n-gram matching.
     *
     * Combined Jaccard:
     * $$
     * J_{combined} = w_h \cdot J(q, h, n=1) + w_s \cdot J(q, s, n=4)
     * $$
     */
    private readonly jaccardHistoryWeight;
    private readonly jaccardStateWeight;
    private readonly ngramSize;
    /**
     * Dynamic threshold for similarity filtering.
     *
     * Threshold decreases over steps to encourage exploration:
     * $$
     * \tau(t) = \tau_{base} - \delta \cdot \min\left(1, \frac{t}{t_{max}}\right)
     * $$
     */
    private readonly baseThreshold;
    private readonly thresholdDecay;
    private readonly maxStepsForThreshold;
    /** Episode weighting (extension beyond paper for confidence scaling) */
    private readonly maxEpisodeWeight;
    private readonly episodeWeightScale;
    /**
     * Numerical stability epsilon for advantage normalization.
     * From paper: $A_{max} = \max_{a' \in C} |A(s,a')| + \epsilon$
     */
    private readonly epsilon;
    /**
     * Exploration parameters for unseen actions (from paper).
     *
     * For unseen actions $|N(s,a)| = 0$:
     * $$
     * \hat{Q}(s,a) = \begin{cases}
     *   \hat{V}(s) + \frac{\alpha}{|N(s)|} & \text{with probability } \lambda \\
     *   0 & \text{with probability } 1-\lambda
     * \end{cases}
     * $$
     */
    private readonly explorationBonus;
    private readonly explorationProb;
    constructor(projectPath?: string);
    private loadState;
    private initIndexes;
    private saveState;
    /**
     * Tokenize text into lowercase alphanumeric tokens.
     *
     * Produces the token set $T(s)$ used for Jaccard similarity.
     *
     * @param text - Input text
     * @returns Array of cleaned tokens
     */
    private tokenize;
    /**
     * Extract n-grams from token sequence.
     *
     * For tokens $[t_1, t_2, ..., t_m]$ and n-gram size $n$:
     * $$
     * \text{ngrams} = \{(t_i, t_{i+1}, ..., t_{i+n-1}) : i \in [1, m-n+1]\}
     * $$
     *
     * @param tokens - Input token array
     * @param n - N-gram size
     * @returns Array of n-gram strings
     */
    private getNgrams;
    /**
     * Compute Jaccard similarity between token sequences.
     *
     * From the paper, Jaccard similarity with multiset (bag) semantics:
     * $$
     * J(s, s_i) = \frac{|T(s) \cap T(s_i)|}{|T(s) \cup T(s_i)|}
     * $$
     *
     * where $T(\cdot)$ denotes the tokenized n-gram representation.
     * Uses multiset intersection/union (counts overlap with multiplicity).
     *
     * @param aTokens - First token sequence
     * @param bTokens - Second token sequence
     * @param ngram - N-gram size for comparison
     * @returns Jaccard similarity score in $[0, 1]$
     */
    private jaccardSimilarity;
    /**
     * Compute discounted cumulative return.
     *
     * From the paper:
     * $$
     * G_t = \sum_{u=t}^{T} \gamma^{u-t} r_u
     * $$
     *
     * where $\gamma \in [0,1]$ is the discount factor balancing
     * immediate versus future rewards.
     *
     * @param futureRewards - Array of rewards $[r_t, r_{t+1}, ..., r_T]$
     * @returns Discounted return $G_t$
     */
    private computeDiscountedReturn;
    /**
     * Calculate immediate reward score from action outcome.
     *
     * Implements a reward function $r: \mathcal{O} \to [-10, 10]$:
     * - Base: $+5$ for success, $-2$ for failure
     * - Feedback modifiers: "perfect/great" $+5$, "good" $+2$, "wrong/bad" $-3$
     *
     * @param outcome - Action outcome with success flag and feedback
     * @returns Reward score clamped to $[-10, 10]$
     */
    private calculateScore;
    /**
     * Store a new experience in memory.
     *
     * Creates a compact triplet $(s_i, a_i, G_i)$ as defined in the paper:
     * - $s_i$: Abstracted state representation
     * - $a_i$: Action taken (tool + parameters)
     * - $G_i$: Cumulative discounted reward
     *
     * The experience is indexed in dual HNSW indexes for efficient retrieval.
     *
     * @param state - Environment state when action was taken
     * @param action - Action taken (tool_name, summary, parameters)
     * @param outcome - Result of the action (success, error, feedback)
     * @param trajectoryContext - Summary of trajectory history
     * @param futureRewards - Future rewards for computing $G_t$
     * @returns Stored experience with computed score and return
     */
    add(state: Record<string, unknown>, action: {
        tool_name?: string;
        summary?: string;
        parameters?: Record<string, unknown>;
    }, outcome: {
        success?: boolean;
        error_summary?: string;
        user_feedback?: string;
    }, trajectoryContext?: string, futureRewards?: number[]): Promise<Experience>;
    /**
     * Retrieve top-k similar experiences from memory.
     *
     * Implements the retrieval operation from the paper:
     * $$
     * N(s) \leftarrow \text{Top-}k \text{ neighbors from } \mathcal{M} \text{ by } J(s, s_i)
     * $$
     *
     * Uses a dual-index strategy combining:
     * 1. **Vector similarity** via HNSW (approximate nearest neighbor)
     * 2. **Jaccard n-gram similarity** for precise text matching
     *
     * Dynamic thresholding relaxes over time to encourage exploration:
     * $$
     * \tau(t) = \tau_{base} - \delta \cdot \min\left(1, \frac{t}{t_{max}}\right)
     * $$
     *
     * @param query - Current context/state to search for
     * @param k - Number of neighbors to retrieve
     * @param stepCount - Current step (affects threshold decay)
     * @returns Top-k similar experiences with similarity scores
     */
    search(query: string, k?: number, stepCount?: number): Promise<SearchResult[]>;
    /**
     * Compute advantage estimates for each action type.
     *
     * Implements the advantage estimation from the paper:
     *
     * **State Value (baseline):**
     * $$
     * \hat{V}(s) := \frac{1}{|N(s)|} \sum_{i \in N(s)} G_i
     * $$
     *
     * **Action Value:**
     * $$
     * \hat{Q}(s,a) := \frac{1}{|N(s,a)|} \sum_{j \in N(s,a)} G_j
     * $$
     *
     * **Advantage:**
     * $$
     * \hat{A}(s,a) = \hat{Q}(s,a) - \hat{V}(s)
     * $$
     *
     * **Normalization (from paper):**
     * $$
     * \tilde{A}(s,a) = \frac{A(s,a)}{A_{max} + \epsilon}
     * $$
     * where $A_{max} = \max_{a' \in C} |A(s,a')|$
     *
     * **Unseen Action Handling:**
     * For actions with $|N(s,a)| = 0$:
     * $$
     * \hat{Q}(s,a) = \begin{cases}
     *   \hat{V}(s) + \frac{\alpha}{|N(s)|} & \text{with probability } \lambda \\
     *   0 & \text{with probability } 1-\lambda
     * \end{cases}
     * $$
     *
     * @param results - Retrieved neighbor experiences $N(s)$
     * @param candidateActions - Optional set of candidate actions to include
     * @returns Normalized advantage estimates per action type
     */
    getAdvantages(results: SearchResult[], candidateActions?: string[]): Record<string, number>;
    /**
     * Get statistics about the experience store.
     *
     * Provides summary metrics for monitoring learning progress:
     * - Total experiences in memory $|\mathcal{M}|$
     * - Episode count for weighting
     * - Score distribution (positive/negative)
     * - Action type frequencies
     *
     * @returns Store statistics object
     */
    getStats(): Record<string, unknown>;
    /**
     * Clear all experiences and reset indexes.
     *
     * Resets the memory: $\mathcal{M} \leftarrow \emptyset$
     */
    clear(): void;
    /**
     * Increment episode counter.
     *
     * Called at session end to update the episode weight factor:
     * $$
     * w_{episode} = \min\left(1 + \frac{n_{episode}}{50} \cdot 0.5, 1.5\right)
     * $$
     */
    incrementEpisode(): void;
}
