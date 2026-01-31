/**
 * JitRL Experience Store
 *
 * Implements the paper's algorithm:
 * - Dual vector search (history + state)
 * - Jaccard N-gram similarity
 * - Discounted future rewards
 * - Normalized advantages with episode weighting
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
    private readonly gamma;
    private readonly vectorDim;
    private readonly historyWeight;
    private readonly stateWeight;
    private readonly jaccardHistoryWeight;
    private readonly jaccardStateWeight;
    private readonly ngramSize;
    private readonly baseThreshold;
    private readonly thresholdDecay;
    private readonly maxStepsForThreshold;
    private readonly maxEpisodeWeight;
    private readonly episodeWeightScale;
    constructor(projectPath?: string);
    private loadState;
    private initIndexes;
    private saveState;
    private tokenize;
    private getNgrams;
    private jaccardSimilarity;
    private computeDiscountedReturn;
    private calculateScore;
    add(state: Record<string, unknown>, action: {
        tool_name?: string;
        summary?: string;
        parameters?: Record<string, unknown>;
    }, outcome: {
        success?: boolean;
        error_summary?: string;
        user_feedback?: string;
    }, trajectoryContext?: string, futureRewards?: number[]): Promise<Experience>;
    search(query: string, k?: number, stepCount?: number): Promise<SearchResult[]>;
    getAdvantages(results: SearchResult[]): Record<string, number>;
    getStats(): Record<string, unknown>;
    clear(): void;
    incrementEpisode(): void;
}
