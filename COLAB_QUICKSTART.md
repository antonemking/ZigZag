# ZigZag Colab Quick Start

Copy these cells directly into Google Colab (with GPU enabled).

## Cell 1: Setup

```bash
%%bash
# Clone and setup
git clone https://github.com/antonemking/ZigZag.git
cd ZigZag
bash setup_colab.sh
```

## Cell 2: Download Model

```bash
%%bash
cd ZigZag
mkdir -p models
wget https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/model.onnx -O models/minilm.onnx
```

## Cell 3: Build and Test

```bash
%%bash
cd ZigZag
source /tmp/zigzag_env.sh
zig build test
```

## Expected Output

You should see:
```
Using CUDA GPU acceleration
✅ All 11 tests passed

=== Batch Inference Performance (batch_size=32) ===
Batched:    10-50ms total (0.3-1.5ms per inference)
Sequential: 200-500ms total (6-15ms per inference)
Speedup:    10-50x faster

Projected time for 1000 docs with batching: 300-1500ms
```

## Troubleshooting

### "No GPU found"
- Go to Runtime → Change runtime type → GPU
- Verify: `!nvidia-smi`

### "CUDA not available"
```python
import onnxruntime as ort
print(ort.get_available_providers())
# Should include 'CUDAExecutionProvider'
```

If missing:
```bash
!pip install onnxruntime-gpu --force-reinstall
```

### "libonnxruntime.so not found"
```bash
# Find the library
!find /usr -name "libonnxruntime.so*" 2>/dev/null

# Set LD_LIBRARY_PATH manually
export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH
```

### Manual Path Configuration

If automatic detection fails:

```bash
# Find ONNX Runtime paths
python3 -c "import onnxruntime as ort; import os; print(os.path.dirname(ort.__file__))"

# Set paths manually
export ONNX_INCLUDE=/usr/local/lib/python3.X/dist-packages/onnxruntime/capi/include
export ONNX_LIB=/usr/local/lib/python3.X/dist-packages/onnxruntime/capi/lib
export LD_LIBRARY_PATH=$ONNX_LIB:$LD_LIBRARY_PATH
export COLAB_GPU=1
export USE_GPU=1
```

## Benchmark Python Baseline First

If you're having trouble with Zig, verify GPU works with Python:

```python
!pip install sentence-transformers

from sentence_transformers import CrossEncoder
import time

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')
pairs = [["query", "document"]] * 1000

start = time.time()
scores = model.predict(pairs, batch_size=32)
elapsed = (time.time() - start) * 1000

print(f"1000 inferences: {elapsed:.0f}ms")
print(f"Per inference: {elapsed/1000:.2f}ms")
```

Expected: 50-200ms for 1000 docs on T4 GPU.
