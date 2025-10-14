# ZigZag

**High-performance cross-encoder inference for retrieval systems.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is ZigZag?

ZigZag is a high-performance inference engine for cross-encoder reranking models. It wraps any cross-encoder (like BGE-reranker, MiniLM) and makes it fast enough to rerank 1000+ candidates in real-time, eliminating the need for vector databases.

**What ZigZag does:**
- Loads quantized cross-encoder models (ONNX, INT8/FP16)
- Performs optimized batch inference with SIMD and parallelism
- Scores 1000+ (query, document) pairs in 50-100ms on CPU
- Provides simple APIs for Zig, Python, and C

**Use ZigZag when:**
- You have candidates from BM25/Elasticsearch (100-1000+ docs)
- Python reranking is too slow for your latency budget
- You want cross-encoder accuracy without GPU costs
- You want to skip vector database infrastructure entirely

## The Problem

Cross-encoders are more accurate than bi-encoders (embeddings), but they're considered "too slow" for real-time reranking:

```python
# Typical Python reranking with sentence-transformers
from sentence_transformers import CrossEncoder

model = CrossEncoder('BAAI/bge-reranker-base')
scores = model.predict([(query, doc) for doc in candidates])

# For 1000 documents: 500ms-2s on CPU
# Most systems only rerank 100 docs due to latency constraints
```

**Why it's slow:**
- Python overhead and GIL contention
- Sequential or poorly batched processing
- FP32 precision (no quantization)
- Inefficient memory layout and cache misses

## The Solution

ZigZag wraps your cross-encoder model and optimizes every part of the inference pipeline:

```zig
// ZigZag reranking
var reranker = try zigzag.Reranker.init(.{
    .model_path = "bge-reranker-v2-m3.onnx",  // Any cross-encoder
    .quantization = .int8,                     // Q8 quantization
    .batch_size = 32,
});

const scores = try reranker.rerank(
    query,
    candidates,  // 1000 documents
    .{ .top_k = 10 }
);
// Total: 50-100ms (5-10x faster than Python)
```

**How it's faster:**
- **Zero Python overhead** - Pure Zig inference, no GIL
- **INT8/FP16 quantization** - 2-4x faster than FP32 with minimal accuracy loss
- **SIMD operations** - Vectorized matrix ops (AVX2/AVX-512)
- **Parallel batching** - Multi-threaded scoring across CPU cores
- **Smart pruning** - Early exit for low-scoring candidates (optional)
- **Cache-friendly** - Optimized memory layouts for L1/L2/L3 caches

## Architecture

ZigZag is an inference engine, not a search system. You integrate it into your retrieval pipeline:

```
Your Application
    │
    ├─ Stage 1: Fast candidate retrieval (you provide this)
    │  └─ BM25 (Elasticsearch, Tantivy, custom implementation)
    │  └─ OR Sparse embeddings (SPLADE)
    │  └─ OR Keyword search
    │     Returns: 1000 candidates in ~5-50ms
    │
    └─ Stage 2: Accurate reranking (ZigZag)
       └─ Loads your cross-encoder model (ONNX format)
       └─ Scores all (query, doc) pairs with optimized inference
          Returns: Top K results in ~50-100ms
```

**Key insight:** ZigZag doesn't do retrieval. It makes cross-encoder *reranking* fast enough to replace vector databases entirely.

## Performance Targets

| Operation | ZigZag Target | Python Baseline |
|-----------|---------------|-----------------|
| Score 1000 candidates | 50-100ms | 500ms-2s |
| Memory overhead | <100MB | 500MB+ (PyTorch) |
| Throughput | 10K+ docs/sec | 500-2K docs/sec |
| Typical rerank window | 1000+ docs | 100 docs (latency limited) |

## Use Cases

**What ZigZag enables:**

- **Skip vector databases entirely** - BM25 → ZigZag reranking → Results
- **Rerank 10x more candidates** - 1000 docs instead of 100 (better recall)
- **Real-time search** - Sub-100ms reranking at scale
- **Cost reduction** - CPU inference, no GPU/vector DB infrastructure
- **Use any cross-encoder** - Bring your own model (ONNX format)

**Who should use ZigZag:**

- Teams building RAG systems (10K-1M documents)
- Anyone frustrated with embedding quality
- Systems that need accuracy over approximate search
- Cost-conscious deployments

## Installation

**Note:** ZigZag is currently in development.

```bash
# Coming soon
git clone https://github.com/yourusername/zigzag.git
cd zigzag
zig build
```

## Quick Start

### Option 1: Zig API

```zig
const zigzag = @import("zigzag");

pub fn main() !void {
    // Load quantized cross-encoder
    var reranker = try zigzag.Reranker.init(.{
        .model_path = "models/ms-marco-minilm-l6-v2.onnx",
        .quantization = .int8,
        .batch_size = 32,
    });
    defer reranker.deinit();

    // Score candidates
    const query = "How do I implement authentication?";
    const candidates = try bm25_search(query, 1000);
    
    const scores = try reranker.score_batch(
        query,
        candidates,
        .{ .use_pruning = true }
    );
    
    // Get top K
    const top_k = try scores.top_k(10);
}
```

### Option 2: Python Bindings

```python
# Coming soon
import zigzag

# Initialize reranker
reranker = zigzag.Reranker(
    model_path="models/ms-marco-minilm-l6-v2.onnx",
    quantization="int8",
    batch_size=32
)

# Your existing BM25 search
candidates = bm25_index.search(query, top_k=1000)

# Rerank with ZigZag
results = reranker.rerank(
    query="How do I implement authentication?",
    documents=candidates,
    top_k=10
)
```

### Option 3: C API

```c
// Coming soon - For integration with any language
#include "zigzag.h"

zigzag_reranker_t* reranker = zigzag_init("model.onnx");
zigzag_scores_t* scores = zigzag_score_batch(
    reranker,
    query,
    candidates,
    num_candidates
);
zigzag_free(reranker);
```

## Roadmap

### Phase 1: Core Engine (Weeks 1-3)
- [ ] ONNX Runtime integration
- [ ] INT8 quantization support
- [ ] Batch inference with threading
- [ ] Early pruning optimization
- [ ] Benchmark on MS MARCO

### Phase 2: Production Ready (Weeks 4-8)
- [ ] Python bindings (PyO3 or C API)
- [ ] C API for language interop
- [ ] Documentation and examples
- [ ] Memory profiling and optimization
- [ ] CI/CD pipeline

### Phase 3: Advanced Features (Months 3+)
- [ ] GPU offload option
- [ ] Custom quantization schemes
- [ ] Model optimization tools
- [ ] Additional language bindings (Go, Rust, JS)

## Why Zig?

Zig makes this possible:

- **Manual memory control** - Zero allocations in hot path
- **SIMD intrinsics** - Direct vectorized operations
- **True parallelism** - Efficient threading without GIL
- **C interop** - Seamless ONNX Runtime integration
- **Comptime** - Generate optimized code per model
- **Cross-platform** - Single codebase, all targets

## Contributing

ZigZag is in early development. We need help with:

- **Core performance** - SIMD, quantization, batching
- **Language bindings** - Python, C, JavaScript, Go, Rust
- **Benchmarking** - Datasets, profiling, optimization
- **Documentation** - Examples, tutorials, best practices

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Benchmarks

**Coming soon** - Performance comparisons against:
- Sentence Transformers (Python)
- Torch-compiled inference
- ONNX Runtime (Python)
- Other cross-encoder implementations

## Requirements

- Zig 0.13.0 or later
- ONNX Runtime
- Cross-encoder model in ONNX format

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- **Sentence Transformers** - Cross-encoder training and Python inference
- **ONNX Runtime** - Underlying inference engine
- **Tantivy** - Rust-based BM25 search (pairs well with ZigZag)

## Citation

```bibtex
@software{zigzag2025,
  title={ZigZag: High-Performance Cross-Encoder Inference for Retrieval},
  author={Antone King},
  year={2025},
  url={https://github.com/antonemking/zigzag}
}
```

---

**ZigZag is a reranking inference engine, not a search system.**

You bring:
- Candidates (from BM25, Elasticsearch, or any retrieval system)
- A cross-encoder model (in ONNX format)

ZigZag provides:
- 5-10x faster inference than Python
- Ability to rerank 1000+ docs in real-time
- Simple APIs for Zig, Python, and C

This makes vector-DB-less retrieval practical for production systems.