"use strict";
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
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.ExperienceStore = void 0;
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
const crypto = __importStar(require("crypto"));
const hnswlib_node_1 = require("hnswlib-node");
/**
 * Simple hash-based text embedding (no external API needed).
 *
 * Creates a sparse vector representation using positional hashing:
 * $$
 * \vec{v}[h(w_i) \mod d] \mathrel{+}= \frac{1}{i+1}
 * $$
 *
 * where $h(w)$ is a polynomial rolling hash and $d$ is the vector dimension.
 * The $\frac{1}{i+1}$ weighting gives higher importance to earlier tokens.
 *
 * The vector is L2-normalized for cosine similarity:
 * $$
 * \hat{\vec{v}} = \frac{\vec{v}}{\|\vec{v}\|_2}
 * $$
 *
 * @param text - Input text to embed
 * @param dim - Vector dimension (default: 384)
 * @returns L2-normalized embedding vector
 */
function simpleEmbed(text, dim = 384) {
    const vec = new Array(dim).fill(0);
    const words = text.toLowerCase().split(/\s+/);
    for (let i = 0; i < words.length && i < 500; i++) {
        const word = words[i];
        let hash = 0;
        for (let j = 0; j < word.length; j++) {
            hash = (hash * 31 + word.charCodeAt(j)) >>> 0;
        }
        const idx = hash % dim;
        vec[idx] += 1.0 / (i + 1);
    }
    // L2 normalize
    let norm = 0;
    for (let i = 0; i < dim; i++) {
        norm += vec[i] * vec[i];
    }
    norm = Math.sqrt(norm);
    if (norm > 0) {
        for (let i = 0; i < dim; i++) {
            vec[i] /= norm;
        }
    }
    return vec;
}
class ExperienceStore {
    baseDir;
    projectDir;
    indexDir;
    historyIndex = null;
    stateIndex = null;
    metadata = [];
    episodeCount = 0;
    /**
     * Algorithm hyperparameters from the JitRL paper.
     *
     * ### Discounted Return
     * $$
     * G_t = \sum_{u=t}^{T} \gamma^{u-t} r_u
     * $$
     */
    /** Discount factor $\gamma \in [0,1]$ for future rewards */
    gamma = 0.95;
    /** Vector embedding dimension for HNSW index */
    vectorDim = 384;
    /**
     * Dual-index similarity weights.
     *
     * Combined vector similarity:
     * $$
     * \text{sim}_{vec} = w_h \cdot \text{sim}_{history} + w_s \cdot \text{sim}_{state}
     * $$
     */
    historyWeight = 0.25;
    stateWeight = 0.75;
    /**
     * Jaccard similarity weights for n-gram matching.
     *
     * Combined Jaccard:
     * $$
     * J_{combined} = w_h \cdot J(q, h, n=1) + w_s \cdot J(q, s, n=4)
     * $$
     */
    jaccardHistoryWeight = 0.3;
    jaccardStateWeight = 0.7;
    ngramSize = 4;
    /**
     * Dynamic threshold for similarity filtering.
     *
     * Threshold decreases over steps to encourage exploration:
     * $$
     * \tau(t) = \tau_{base} - \delta \cdot \min\left(1, \frac{t}{t_{max}}\right)
     * $$
     */
    baseThreshold = 0.7;
    thresholdDecay = 0.1;
    maxStepsForThreshold = 20;
    /** Episode weighting (extension beyond paper for confidence scaling) */
    maxEpisodeWeight = 1.5;
    episodeWeightScale = 50;
    /**
     * Numerical stability epsilon for advantage normalization.
     * From paper: $A_{max} = \max_{a' \in C} |A(s,a')| + \epsilon$
     */
    epsilon = 1e-8;
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
    explorationBonus = 0.1; // α
    explorationProb = 0.3; // λ
    constructor(projectPath) {
        const cwd = projectPath || process.cwd();
        const projectHash = crypto
            .createHash("md5")
            .update(cwd)
            .digest("hex")
            .slice(0, 12);
        this.baseDir = path.join(process.env.HOME || "~", ".claude-jitrl");
        this.projectDir = path.join(this.baseDir, "experiences", projectHash);
        this.indexDir = path.join(this.baseDir, "indexes", projectHash);
        fs.mkdirSync(this.projectDir, { recursive: true });
        fs.mkdirSync(this.indexDir, { recursive: true });
        this.loadState();
    }
    loadState() {
        const metaPath = path.join(this.projectDir, "metadata.json");
        const episodePath = path.join(this.projectDir, "episode_count.txt");
        if (fs.existsSync(metaPath)) {
            try {
                this.metadata = JSON.parse(fs.readFileSync(metaPath, "utf-8"));
            }
            catch {
                this.metadata = [];
            }
        }
        if (fs.existsSync(episodePath)) {
            this.episodeCount = parseInt(fs.readFileSync(episodePath, "utf-8"), 10) || 0;
        }
        this.initIndexes();
    }
    initIndexes() {
        const historyPath = path.join(this.indexDir, "history.hnsw");
        const statePath = path.join(this.indexDir, "state.hnsw");
        // Initialize HNSW indexes
        this.historyIndex = new hnswlib_node_1.HierarchicalNSW("cosine", this.vectorDim);
        this.stateIndex = new hnswlib_node_1.HierarchicalNSW("cosine", this.vectorDim);
        if (fs.existsSync(historyPath) && this.metadata.length > 0) {
            this.historyIndex.readIndexSync(historyPath);
        }
        else {
            this.historyIndex.initIndex(10000, 16, 200, 100);
        }
        if (fs.existsSync(statePath) && this.metadata.length > 0) {
            this.stateIndex.readIndexSync(statePath);
        }
        else {
            this.stateIndex.initIndex(10000, 16, 200, 100);
        }
    }
    saveState() {
        const metaPath = path.join(this.projectDir, "metadata.json");
        const episodePath = path.join(this.projectDir, "episode_count.txt");
        const historyPath = path.join(this.indexDir, "history.hnsw");
        const statePath = path.join(this.indexDir, "state.hnsw");
        fs.writeFileSync(metaPath, JSON.stringify(this.metadata, null, 2));
        fs.writeFileSync(episodePath, String(this.episodeCount));
        if (this.historyIndex && this.metadata.length > 0) {
            this.historyIndex.writeIndexSync(historyPath);
        }
        if (this.stateIndex && this.metadata.length > 0) {
            this.stateIndex.writeIndexSync(statePath);
        }
    }
    /**
     * Tokenize text into lowercase alphanumeric tokens.
     *
     * Produces the token set $T(s)$ used for Jaccard similarity.
     *
     * @param text - Input text
     * @returns Array of cleaned tokens
     */
    tokenize(text) {
        return (text || "")
            .toLowerCase()
            .replace(/\n/g, " ")
            .split(/\s+/)
            .filter((t) => /^[a-z0-9]+$/i.test(t));
    }
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
    getNgrams(tokens, n) {
        if (tokens.length < n)
            return tokens;
        const ngrams = [];
        for (let i = 0; i <= tokens.length - n; i++) {
            ngrams.push(tokens.slice(i, i + n).join(" "));
        }
        return ngrams;
    }
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
    jaccardSimilarity(aTokens, bTokens, ngram) {
        const aNgrams = this.getNgrams(aTokens, ngram);
        const bNgrams = this.getNgrams(bTokens, ngram);
        if (aNgrams.length === 0 || bNgrams.length === 0)
            return 0;
        const aCount = new Map();
        const bCount = new Map();
        for (const ng of aNgrams) {
            aCount.set(ng, (aCount.get(ng) || 0) + 1);
        }
        for (const ng of bNgrams) {
            bCount.set(ng, (bCount.get(ng) || 0) + 1);
        }
        let intersection = 0;
        let union = 0;
        const allKeys = new Set([...aCount.keys(), ...bCount.keys()]);
        for (const key of allKeys) {
            const aVal = aCount.get(key) || 0;
            const bVal = bCount.get(key) || 0;
            intersection += Math.min(aVal, bVal);
            union += Math.max(aVal, bVal);
        }
        return union > 0 ? intersection / union : 0;
    }
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
    computeDiscountedReturn(futureRewards) {
        let discounted = 0;
        for (let t = 0; t < futureRewards.length; t++) {
            discounted += Math.pow(this.gamma, t) * futureRewards[t];
        }
        return discounted;
    }
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
    calculateScore(outcome) {
        let base = outcome.success ? 5 : -2;
        const feedback = (outcome.user_feedback || "").toLowerCase();
        if (feedback.includes("perfect") || feedback.includes("great")) {
            base += 5;
        }
        else if (feedback.includes("good")) {
            base += 2;
        }
        else if (feedback.includes("wrong") || feedback.includes("bad")) {
            base -= 3;
        }
        return Math.max(-10, Math.min(10, base));
    }
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
    async add(state, action, outcome, trajectoryContext, futureRewards) {
        const score = this.calculateScore(outcome);
        const rewards = futureRewards || [score];
        const discountedReturn = this.computeDiscountedReturn(rewards);
        const historyText = trajectoryContext || JSON.stringify(state);
        const stateText = `${action.tool_name || ""}: ${action.summary || ""}`;
        const experience = {
            timestamp: new Date().toISOString(),
            episodeNumber: this.episodeCount,
            state,
            action,
            outcome,
            score,
            trajectoryContext: historyText,
            currentEnvInfo: stateText,
            futureRewards: rewards,
            discountedReturn,
        };
        // Add to indexes
        if (this.historyIndex && this.stateIndex) {
            const historyVec = simpleEmbed(historyText, this.vectorDim);
            const stateVec = simpleEmbed(stateText, this.vectorDim);
            const idx = this.metadata.length;
            this.historyIndex.addPoint(historyVec, idx);
            this.stateIndex.addPoint(stateVec, idx);
            this.metadata.push({
                experience,
                historyTokens: this.tokenize(historyText),
                stateTokens: this.tokenize(stateText),
                idx,
            });
        }
        this.saveState();
        return experience;
    }
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
    async search(query, k = 5, stepCount = 0) {
        if (!this.historyIndex || !this.stateIndex || this.metadata.length === 0) {
            return [];
        }
        // Dynamic threshold (per paper)
        let dynamicThreshold;
        if (stepCount >= this.maxStepsForThreshold) {
            dynamicThreshold = this.baseThreshold - this.thresholdDecay;
        }
        else {
            const decay = this.thresholdDecay * (stepCount / this.maxStepsForThreshold);
            dynamicThreshold = this.baseThreshold - decay;
        }
        const queryVec = simpleEmbed(query, this.vectorDim);
        const queryTokens = this.tokenize(query);
        // Search both indexes
        const recallSize = Math.min(Math.max(k * 10, 100), this.metadata.length);
        const historyResults = this.historyIndex.searchKnn(queryVec, recallSize);
        const stateResults = this.stateIndex.searchKnn(queryVec, recallSize);
        // Combine scores
        const combined = new Map();
        for (let i = 0; i < historyResults.neighbors.length; i++) {
            const idx = historyResults.neighbors[i];
            const score = 1 - historyResults.distances[i]; // Convert distance to similarity
            combined.set(idx, { historyVec: score, stateVec: 0 });
        }
        for (let i = 0; i < stateResults.neighbors.length; i++) {
            const idx = stateResults.neighbors[i];
            const score = 1 - stateResults.distances[i];
            const existing = combined.get(idx);
            if (existing) {
                existing.stateVec = score;
            }
            else {
                combined.set(idx, { historyVec: 0, stateVec: score });
            }
        }
        const results = [];
        for (const [idx, scores] of combined) {
            if (idx >= this.metadata.length)
                continue;
            const meta = this.metadata[idx];
            // Vector similarity (weighted)
            const vecSim = this.historyWeight * scores.historyVec +
                this.stateWeight * scores.stateVec;
            // Jaccard N-gram similarity (per paper)
            const jaccardHistory = this.jaccardSimilarity(queryTokens, meta.historyTokens, 1);
            const jaccardState = this.jaccardSimilarity(queryTokens, meta.stateTokens, this.ngramSize);
            const jaccardSim = this.jaccardHistoryWeight * jaccardHistory +
                this.jaccardStateWeight * jaccardState;
            // Use Jaccard as primary (per paper implementation)
            const similarity = jaccardSim;
            if (similarity < dynamicThreshold)
                continue;
            results.push({
                experience: meta.experience,
                similarity,
                vectorSimilarity: vecSim,
                jaccardSimilarity: jaccardSim,
                discountedReturn: meta.experience.discountedReturn,
            });
        }
        // Sort by similarity, then discounted return
        results.sort((a, b) => {
            if (b.similarity !== a.similarity) {
                return b.similarity - a.similarity;
            }
            return b.discountedReturn - a.discountedReturn;
        });
        return results.slice(0, k);
    }
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
    getAdvantages(results, candidateActions) {
        if (results.length === 0)
            return {};
        const neighborCount = results.length; // |N(s)|
        // Aggregate returns per action: N(s,a) → [G_j]
        const actionReturns = new Map();
        for (const r of results) {
            const actionType = r.experience.action.tool_name || "unknown";
            const returns = actionReturns.get(actionType) || [];
            returns.push(r.discountedReturn);
            actionReturns.set(actionType, returns);
        }
        // Compute Q̂(s,a) = mean of returns for each action
        const actionQ = new Map();
        for (const [action, returns] of actionReturns) {
            actionQ.set(action, returns.reduce((a, b) => a + b, 0) / returns.length);
        }
        // Compute V̂(s) = mean of all returns (baseline)
        const allReturns = [];
        for (const returns of actionReturns.values()) {
            allReturns.push(...returns);
        }
        const baseline = allReturns.length > 0
            ? allReturns.reduce((a, b) => a + b, 0) / allReturns.length
            : 0;
        // Handle unseen candidate actions with exploration bonus (per paper)
        if (candidateActions) {
            for (const action of candidateActions) {
                if (!actionQ.has(action)) {
                    // Unseen action: apply exploration bonus with probability λ
                    if (Math.random() < this.explorationProb) {
                        // Q̂(s,a) = V̂(s) + α/|N(s)|
                        actionQ.set(action, baseline + this.explorationBonus / neighborCount);
                    }
                    else {
                        // Q̂(s,a) = 0
                        actionQ.set(action, 0);
                    }
                }
            }
        }
        // Compute raw advantages: Â(s,a) = Q̂(s,a) - V̂(s)
        const rawAdvantages = new Map();
        for (const [action, q] of actionQ) {
            rawAdvantages.set(action, q - baseline);
        }
        // Normalize by max absolute value (per paper):
        // A_max = max_{a' ∈ C} |A(s,a')| + ε
        const absAdvantages = [...rawAdvantages.values()].map(Math.abs);
        const aMax = absAdvantages.length > 0 ? Math.max(...absAdvantages) : 0;
        const normalizer = aMax + this.epsilon;
        // Ã(s,a) = A(s,a) / (A_max + ε)
        const normalized = new Map();
        for (const [action, adv] of rawAdvantages) {
            normalized.set(action, adv / normalizer);
        }
        // Episode weighting (extension: confidence scaling over episodes)
        // Not in original paper, but helps stabilize early learning
        const episodeWeight = Math.min(1.0 + (this.episodeCount / this.episodeWeightScale) * 0.5, this.maxEpisodeWeight);
        const weighted = {};
        for (const [action, norm] of normalized) {
            weighted[action] = norm * episodeWeight;
        }
        return weighted;
    }
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
    getStats() {
        const scores = this.metadata.map((m) => m.experience.score);
        const actionCounts = new Map();
        for (const m of this.metadata) {
            const action = m.experience.action.tool_name || "unknown";
            actionCounts.set(action, (actionCounts.get(action) || 0) + 1);
        }
        return {
            totalExperiences: this.metadata.length,
            episodeCount: this.episodeCount,
            avgScore: scores.length > 0
                ? scores.reduce((a, b) => a + b, 0) / scores.length
                : 0,
            positiveCount: scores.filter((s) => s > 0).length,
            negativeCount: scores.filter((s) => s < 0).length,
            actionTypes: Object.fromEntries(actionCounts),
            projectDir: this.projectDir,
        };
    }
    /**
     * Clear all experiences and reset indexes.
     *
     * Resets the memory: $\mathcal{M} \leftarrow \emptyset$
     */
    clear() {
        this.metadata = [];
        this.episodeCount = 0;
        this.initIndexes();
        this.saveState();
    }
    /**
     * Increment episode counter.
     *
     * Called at session end to update the episode weight factor:
     * $$
     * w_{episode} = \min\left(1 + \frac{n_{episode}}{50} \cdot 0.5, 1.5\right)
     * $$
     */
    incrementEpisode() {
        this.episodeCount++;
        this.saveState();
    }
}
exports.ExperienceStore = ExperienceStore;
