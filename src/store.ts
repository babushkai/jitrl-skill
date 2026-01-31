/**
 * JitRL Experience Store
 *
 * Implements the paper's algorithm:
 * - Dual vector search (history + state)
 * - Jaccard N-gram similarity
 * - Discounted future rewards
 * - Normalized advantages with episode weighting
 */

import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";
import { HierarchicalNSW } from "hnswlib-node";

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

interface StoreMetadata {
  experience: Experience;
  historyTokens: string[];
  stateTokens: string[];
  idx: number;
}

// Simple embedding using hash-based approach (no external API needed)
function simpleEmbed(text: string, dim: number = 384): number[] {
  const vec: number[] = new Array(dim).fill(0);
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

  // Normalize
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

export class ExperienceStore {
  private baseDir: string;
  private projectDir: string;
  private indexDir: string;

  private historyIndex: HierarchicalNSW | null = null;
  private stateIndex: HierarchicalNSW | null = null;
  private metadata: StoreMetadata[] = [];

  public episodeCount = 0;

  // Configuration (from paper)
  private readonly gamma = 0.95;
  private readonly vectorDim = 384;
  private readonly historyWeight = 0.25;
  private readonly stateWeight = 0.75;
  private readonly jaccardHistoryWeight = 0.3;
  private readonly jaccardStateWeight = 0.7;
  private readonly ngramSize = 4;
  private readonly baseThreshold = 0.7;
  private readonly thresholdDecay = 0.1;
  private readonly maxStepsForThreshold = 20;
  private readonly maxEpisodeWeight = 1.5;
  private readonly episodeWeightScale = 50;

  constructor(projectPath?: string) {
    const cwd = projectPath || process.cwd();
    const projectHash = crypto
      .createHash("md5")
      .update(cwd)
      .digest("hex")
      .slice(0, 12);

    this.baseDir = path.join(
      process.env.HOME || "~",
      ".claude-jitrl"
    );
    this.projectDir = path.join(this.baseDir, "experiences", projectHash);
    this.indexDir = path.join(this.baseDir, "indexes", projectHash);

    fs.mkdirSync(this.projectDir, { recursive: true });
    fs.mkdirSync(this.indexDir, { recursive: true });

    this.loadState();
  }

  private loadState() {
    const metaPath = path.join(this.projectDir, "metadata.json");
    const episodePath = path.join(this.projectDir, "episode_count.txt");

    if (fs.existsSync(metaPath)) {
      try {
        this.metadata = JSON.parse(fs.readFileSync(metaPath, "utf-8"));
      } catch {
        this.metadata = [];
      }
    }

    if (fs.existsSync(episodePath)) {
      this.episodeCount = parseInt(fs.readFileSync(episodePath, "utf-8"), 10) || 0;
    }

    this.initIndexes();
  }

  private initIndexes() {
    const historyPath = path.join(this.indexDir, "history.hnsw");
    const statePath = path.join(this.indexDir, "state.hnsw");

    // Initialize HNSW indexes
    this.historyIndex = new HierarchicalNSW("cosine", this.vectorDim);
    this.stateIndex = new HierarchicalNSW("cosine", this.vectorDim);

    if (fs.existsSync(historyPath) && this.metadata.length > 0) {
      this.historyIndex.readIndexSync(historyPath);
    } else {
      this.historyIndex.initIndex(10000, 16, 200, 100);
    }

    if (fs.existsSync(statePath) && this.metadata.length > 0) {
      this.stateIndex.readIndexSync(statePath);
    } else {
      this.stateIndex.initIndex(10000, 16, 200, 100);
    }
  }

  private saveState() {
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

  private tokenize(text: string): string[] {
    return (text || "")
      .toLowerCase()
      .replace(/\n/g, " ")
      .split(/\s+/)
      .filter((t) => /^[a-z0-9]+$/i.test(t));
  }

  private getNgrams(tokens: string[], n: number): string[] {
    if (tokens.length < n) return tokens;
    const ngrams: string[] = [];
    for (let i = 0; i <= tokens.length - n; i++) {
      ngrams.push(tokens.slice(i, i + n).join(" "));
    }
    return ngrams;
  }

  private jaccardSimilarity(
    aTokens: string[],
    bTokens: string[],
    ngram: number
  ): number {
    const aNgrams = this.getNgrams(aTokens, ngram);
    const bNgrams = this.getNgrams(bTokens, ngram);

    if (aNgrams.length === 0 || bNgrams.length === 0) return 0;

    const aCount = new Map<string, number>();
    const bCount = new Map<string, number>();

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

  private computeDiscountedReturn(futureRewards: number[]): number {
    let discounted = 0;
    for (let t = 0; t < futureRewards.length; t++) {
      discounted += Math.pow(this.gamma, t) * futureRewards[t];
    }
    return discounted;
  }

  private calculateScore(outcome: {
    success?: boolean;
    error_summary?: string;
    user_feedback?: string;
  }): number {
    let base = outcome.success ? 5 : -2;

    const feedback = (outcome.user_feedback || "").toLowerCase();
    if (feedback.includes("perfect") || feedback.includes("great")) {
      base += 5;
    } else if (feedback.includes("good")) {
      base += 2;
    } else if (feedback.includes("wrong") || feedback.includes("bad")) {
      base -= 3;
    }

    return Math.max(-10, Math.min(10, base));
  }

  async add(
    state: Record<string, unknown>,
    action: { tool_name?: string; summary?: string; parameters?: Record<string, unknown> },
    outcome: { success?: boolean; error_summary?: string; user_feedback?: string },
    trajectoryContext?: string,
    futureRewards?: number[]
  ): Promise<Experience> {
    const score = this.calculateScore(outcome);
    const rewards = futureRewards || [score];
    const discountedReturn = this.computeDiscountedReturn(rewards);

    const historyText = trajectoryContext || JSON.stringify(state);
    const stateText = `${action.tool_name || ""}: ${action.summary || ""}`;

    const experience: Experience = {
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

  async search(
    query: string,
    k: number = 5,
    stepCount: number = 0
  ): Promise<SearchResult[]> {
    if (!this.historyIndex || !this.stateIndex || this.metadata.length === 0) {
      return [];
    }

    // Dynamic threshold (per paper)
    let dynamicThreshold: number;
    if (stepCount >= this.maxStepsForThreshold) {
      dynamicThreshold = this.baseThreshold - this.thresholdDecay;
    } else {
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
    const combined = new Map<
      number,
      { historyVec: number; stateVec: number }
    >();

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
      } else {
        combined.set(idx, { historyVec: 0, stateVec: score });
      }
    }

    const results: SearchResult[] = [];

    for (const [idx, scores] of combined) {
      if (idx >= this.metadata.length) continue;

      const meta = this.metadata[idx];

      // Vector similarity (weighted)
      const vecSim =
        this.historyWeight * scores.historyVec +
        this.stateWeight * scores.stateVec;

      // Jaccard N-gram similarity (per paper)
      const jaccardHistory = this.jaccardSimilarity(
        queryTokens,
        meta.historyTokens,
        1
      );
      const jaccardState = this.jaccardSimilarity(
        queryTokens,
        meta.stateTokens,
        this.ngramSize
      );
      const jaccardSim =
        this.jaccardHistoryWeight * jaccardHistory +
        this.jaccardStateWeight * jaccardState;

      // Use Jaccard as primary (per paper implementation)
      const similarity = jaccardSim;

      if (similarity < dynamicThreshold) continue;

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

  getAdvantages(results: SearchResult[]): Record<string, number> {
    if (results.length === 0) return {};

    // Aggregate returns per action
    const actionReturns = new Map<string, number[]>();

    for (const r of results) {
      const actionType = r.experience.action.tool_name || "unknown";
      const returns = actionReturns.get(actionType) || [];
      returns.push(r.discountedReturn);
      actionReturns.set(actionType, returns);
    }

    // Average returns
    const actionAvg = new Map<string, number>();
    for (const [action, returns] of actionReturns) {
      actionAvg.set(action, returns.reduce((a, b) => a + b, 0) / returns.length);
    }

    // Baseline
    const allReturns: number[] = [];
    for (const returns of actionReturns.values()) {
      allReturns.push(...returns);
    }
    const baseline =
      allReturns.length > 0
        ? allReturns.reduce((a, b) => a + b, 0) / allReturns.length
        : 0;

    // Raw advantages
    const rawAdvantages = new Map<string, number>();
    for (const [action, avg] of actionAvg) {
      rawAdvantages.set(action, avg - baseline);
    }

    // Normalize by max positive (per paper)
    const positiveAdvs = [...rawAdvantages.values()].filter((a) => a > 0);
    let normalizer: number;

    if (positiveAdvs.length > 0) {
      normalizer = Math.max(...positiveAdvs);
    } else {
      const minNeg = Math.min(...rawAdvantages.values());
      normalizer = Math.abs(minNeg) || 1;
    }

    const normalized = new Map<string, number>();
    for (const [action, adv] of rawAdvantages) {
      normalized.set(action, adv / normalizer);
    }

    // Episode weighting (per paper: 1.0 → 1.5 over 50 episodes)
    const episodeWeight = Math.min(
      1.0 + (this.episodeCount / this.episodeWeightScale) * 0.5,
      this.maxEpisodeWeight
    );

    const weighted: Record<string, number> = {};
    for (const [action, norm] of normalized) {
      weighted[action] = norm * episodeWeight;
    }

    return weighted;
  }

  getStats(): Record<string, unknown> {
    const scores = this.metadata.map((m) => m.experience.score);
    const actionCounts = new Map<string, number>();

    for (const m of this.metadata) {
      const action = m.experience.action.tool_name || "unknown";
      actionCounts.set(action, (actionCounts.get(action) || 0) + 1);
    }

    return {
      totalExperiences: this.metadata.length,
      episodeCount: this.episodeCount,
      avgScore:
        scores.length > 0
          ? scores.reduce((a, b) => a + b, 0) / scores.length
          : 0,
      positiveCount: scores.filter((s) => s > 0).length,
      negativeCount: scores.filter((s) => s < 0).length,
      actionTypes: Object.fromEntries(actionCounts),
      projectDir: this.projectDir,
    };
  }

  clear() {
    this.metadata = [];
    this.episodeCount = 0;
    this.initIndexes();
    this.saveState();
  }

  incrementEpisode() {
    this.episodeCount++;
    this.saveState();
  }
}
