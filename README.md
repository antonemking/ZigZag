# ZigZag

**High-performance cross-encoder inference for retrieval systems.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is ZigZag?

ZigZag is a Zig library that makes cross-encoder reranking fast enough to replace vector databases. It provides optimized batch inference, quantization, and smart scheduling to score hundreds or thousands of candidates in milliseconds instead of seconds.

**Use ZigZag when:**
- You want accurate retrieval without vector embeddings
- You have 100-1000 candidates from BM25/lexical search
- You need <100ms reranking latency
- You want to avoid vector database infrastructure

## The Problem

Cross-encoders are more accurate than bi-encoders (embeddings), but they're considered "too slow" for first-stage retrieval:

```python
# Typical Python reranking
for doc in candidates:  # 1000 documents
    score = model(query, doc)  # 0.2ms each
    # Total: 200ms + Python overhead = 300-400ms
```

**Why it's slow:**
- Python overhead (50%+ of time)
- Sequential processing
- No quantization
- Inefficient memory layout

## The Solution

ZigZag optimizes every part of the inference pipeline:

```zig
// ZigZag reranking
const scores = try reranker.score_batch(
    query,
    candidates,  // 1000 documents
    .{ .use_pruning = true }
);
// Total: 50-100ms
```

**How it's faster:**
- **Zero Python overhead** - Pure Zig inference
- **INT8 quantization** - 2-5x faster than FP32
- **SIMD operations** - Vectorized matrix math
- **Parallel batching** - Multi-threaded scoring
- **Smart pruning** - Early exit for low-scoring candidates
- **Cache-friendly** - Optimized memory layouts

## Architecture

ZigZag is infrastructure, not an application. You integrate it into your retrieval pipeline:

```
Your Application
    │
    ├─ Stage 1: Fast candidate retrieval
    │  └─ BM25, Elasticsearch, or any search system
    │     Returns: 1000 candidates in ~5ms
    │
    └─ Stage 2: Accurate reranking (ZigZag)
       └─ Cross-encoder scores all candidates
          Returns: Top K results in ~50-100ms
```

## Performance Targets

| Operation | Target | Comparison |
|-----------|--------|------------|
| Score 1000 candidates | 50-100ms | Python: 300-400ms |
| Memory overhead | <100MB | PyTorch: 500MB+ |
| Throughput | 10K+ docs/sec | Python: 2-3K docs/sec |

## Use Cases

**What ZigZag enables:**

- **Skip vector databases entirely** - BM25 → ZigZag → Results
- **Improve existing RAG** - Rerank more candidates faster
- **Real-time search** - Sub-100ms retrieval at scale
- **Cost reduction** - CPU inference instead of GPU/vector DB

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

**ZigZag is infrastructure code.** It's the fast reranking engine that makes vector-DB-less retrieval practical. You bring the candidates, we make them fast to score.