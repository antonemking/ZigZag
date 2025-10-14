const std = @import("std");
const math = std.math;

// BM25 tuning parameters
pub const BM25Config = struct {
    // k1 controls term frequency saturation (1.2-2.0 typical)
    // Higher k1 = more weight to repeated terms
    k1: f32 = 1.5,

    // b controls document length normalization (0-1)
    // b=1 means full normalization, b=0 means no normalization
    b: f32 = 0.75,
};

// Calculate Inverse Document Frequency (IDF) for a term
// Rare terms get higher scores, common terms get lower scores
//
// doc_count: total number of documents in corpus
// docs_with_term: number of documents containing this term
pub fn calculateIDF(doc_count: usize, docs_with_term: usize) f32 {
    const n = @as(f32, @floatFromInt(doc_count));
    const df = @as(f32, @floatFromInt(docs_with_term));

    // Robertson-Zaragoza IDF formula with smoothing
    // The +0.5 prevents negative IDF for common terms
    return math.log(f32, math.e, (n - df + 0.5) / (df + 0.5) + 1.0);
}

// Calculate BM25 score for a single term in a document
//
// term_freq: how many times the term appears in the document
// doc_length: total word count in the document
// avg_doc_length: average document length across the corpus
// idf: inverse document frequency (from calculateIDF)
// config: BM25 tuning parameters
pub fn calculateBM25Score(
    term_freq: f32,
    doc_length: f32,
    avg_doc_length: f32,
    idf: f32,
    config: BM25Config,
) f32 {
    // Normalize document length relative to corpus average
    const normalized_length = doc_length / avg_doc_length;

    // BM25 formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * norm_length))
    const numerator = term_freq * (config.k1 + 1.0);
    const denominator = term_freq + config.k1 * (1.0 - config.b + config.b * normalized_length);

    return idf * (numerator / denominator);
}

test "BM25 IDF Calculation" {
    // 100 total docs, 10 contain the term
    // Expected: ln((100-10+0.5)/(10+0.5)+1) = ln(9.619) ≈ 2.264
    const idf = calculateIDF(100, 10);
    try std.testing.expectApproxEqRel(2.264, idf, 0.001);
}

test "BM25 Score Calculation" {
    const config = BM25Config{};

    // tf=3, doc_len=100, avg_len=150, idf=1.2
    // Shorter than average doc, so should boost the score
    const score = calculateBM25Score(
        3.0, // term appears 3 times
        100.0, // doc is 100 words
        150.0, // average doc is 150 words
        1.2, // IDF weight
        config,
    );

    // Expected: 1.2 * (7.5 / 3.4375) ≈ 2.182
    try std.testing.expectApproxEqRel(2.182, score, 0.001);
}
