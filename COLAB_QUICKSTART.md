# ZigZag Colab Quick Start

Copy these cells directly into Google Colab (with GPU enabled).

**IMPORTANT**: The setup script now extracts headers from the pip wheel itself to avoid ABI mismatch issues.

## Cell 1: Setup

```bash
%%bash
# Clone and setup
git clone https://github.com/antonemking/ZigZag.git
cd ZigZag
bash setup_colab.sh
```

**What this does**:
- Installs Zig 0.14.0
- Installs onnxruntime-gpu via pip
- Extracts C headers from the wheel (not GitHub) to match binary ABI
- Creates symlinks for library versioning
- Sets environment variables

## Cell 2: Build and Test

```bash
%%bash
cd ZigZag
source /tmp/zigzag_env.sh
zig build test
```

## Expected Output

### If GPU works:
```
✅ CUDA GPU acceleration enabled
✅ All 11 tests passed

=== Batch Inference Performance (batch_size=32) ===
Batched:    10-50ms total (0.3-1.5ms per inference)
Sequential: 200-500ms total (6-15ms per inference)
Speedup:    10-50x faster

Projected time for 1000 docs with batching: 50-100ms  ← TARGET
```

### If GPU fallback to CPU:
```
CUDA provider append failed: [error message]
Falling back to CPU execution
✅ All 11 tests passed

Projected time for 1000 docs: 46000ms  ← Too slow, need GPU
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

### ABI Mismatch / Persistent Segfaults

**New fix (2024-10)**: We now extract headers from the pip wheel itself instead of downloading from GitHub. This ensures compile-time headers match runtime binaries exactly.

If you still see segfaults:
1. Check that headers are from wheel: `ls $ONNX_INCLUDE`
2. The setup script should show "Found headers in wheel: ..."
3. If it shows "WARNING: downloading from GitHub", the wheel doesn't include headers
4. In that case, you may need to build ONNX Runtime from source

### Manual Path Configuration

If automatic detection fails:

```bash
# Find ONNX Runtime paths
ONNX_PATH=$(python3 -c "import onnxruntime as ort; import os; print(os.path.dirname(ort.__file__))")
echo "ONNX Runtime: $ONNX_PATH"

# Check if headers exist in wheel
ls "$ONNX_PATH/capi/include" || ls "$ONNX_PATH/include"

# Set paths manually (use wheel headers, not GitHub downloads!)
export ONNX_INCLUDE="$ONNX_PATH/capi/include"  # or "$ONNX_PATH/include"
export ONNX_LIB="$ONNX_PATH/capi"
export LD_LIBRARY_PATH=$ONNX_LIB:$LD_LIBRARY_PATH
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
