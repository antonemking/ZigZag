# Running ZigZag on Google Colab with GPU

ZigZag requires GPU acceleration for real-time performance. CPU inference is ~120x too slow for the target of 50-100ms per 1000 documents.

## Performance Comparison

| Platform | Per-inference | 1000 docs | Speedup Needed |
|----------|--------------|-----------|----------------|
| CPU (Mac M-series) | ~7.5ms | ~7.5 seconds | 1x baseline |
| **Target** | **~0.075ms** | **50-100ms** | **75-150x** |
| GPU (T4/V100) | ~0.1-0.5ms | ~100-500ms | **15-75x** ✅ |

## Google Colab Setup

### Quick Setup (Recommended)

```bash
# Clone ZigZag
!git clone https://github.com/antonemking/ZigZag.git
%cd ZigZag

# Run setup script (installs everything)
!bash setup_colab.sh

# Download model
!mkdir -p models
!wget https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/model.onnx -O models/minilm.onnx

# Build and test
!source /tmp/zigzag_env.sh && zig build test
```

### Manual Setup (if script fails)

### 1. Enable GPU Runtime

- Runtime → Change runtime type → Hardware accelerator → **GPU** (T4 or better)
- Verify GPU: `!nvidia-smi`

### 2. Install Dependencies

```bash
# Install Zig 0.14.0
!wget https://ziglang.org/download/0.14.0/zig-linux-x86_64-0.14.0.tar.xz
!tar -xf zig-linux-x86_64-0.14.0.tar.xz
!ln -s $(pwd)/zig-linux-x86_64-0.14.0/zig /usr/local/bin/zig
!zig version

# Install ONNX Runtime GPU
!pip install onnxruntime-gpu

# Verify CUDA is available
import onnxruntime as ort
print(f"ONNX Runtime version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")
# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 3. Clone and Build ZigZag

```bash
!git clone https://github.com/antonemking/ZigZag.git
%cd ZigZag

# Find ONNX Runtime paths (run in Python cell)
import onnxruntime as ort
import os
ort_path = os.path.dirname(ort.__file__)
print(f"ONNX Runtime path: {ort_path}")
print(f"Include: {ort_path}/capi/include")
print(f"Lib: {ort_path}/capi/lib")

# Set environment variables for build (run in bash cell)
export COLAB_GPU=1
export USE_GPU=1
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi/lib:$LD_LIBRARY_PATH
```

If the default Colab paths don't work, manually set paths:

```bash
# Find the actual paths first
!python3 -c "import onnxruntime as ort; import os; print(os.path.dirname(ort.__file__))"

# Then set environment variables with the correct paths
export ONNX_INCLUDE=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi/include
export ONNX_LIB=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi/lib
export LD_LIBRARY_PATH=$ONNX_LIB:$LD_LIBRARY_PATH
```

### 4. Download Model

```bash
!mkdir -p models
!wget https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/model.onnx -O models/minilm.onnx
```

### 5. Build and Run with GPU

```bash
# Build ZigZag
!COLAB_GPU=1 zig build

# Run tests with GPU enabled
!USE_GPU=1 COLAB_GPU=1 LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi/lib:$LD_LIBRARY_PATH zig build test
```

You should see output like:
```
Using CUDA GPU acceleration
=== Batch Inference Performance (batch_size=32) ===
Batched:    10-50ms total (0.3-1.5ms per inference)
Sequential: 200-500ms total (6-15ms per inference)
Speedup:    10-50x faster
```

## Expected GPU Performance

With CUDA enabled, you should see:
- **Single inference:** 0.1-0.5ms (vs 7.5ms on CPU)
- **Batch inference (32):** 2-10ms total (~0.06-0.3ms per inference)
- **1000 docs:** 60-300ms (hits target! ✅)

## Troubleshooting

### "CUDA GPU not available"

This means ONNX Runtime couldn't find CUDA. Check:
```python
import onnxruntime as ort
print(ort.get_available_providers())
# Should include: 'CUDAExecutionProvider'
```

If not available:
- Verify GPU is enabled in Colab runtime
- Try `!pip install onnxruntime-gpu --force-reinstall`
- Check CUDA version: `!nvcc --version`

### Link Errors

Update `build.zig` to point to the correct ONNX Runtime paths:
```bash
# Find onnxruntime location
!python -c "import onnxruntime as ort; import os; print(os.path.dirname(ort.__file__))"
```

Then update paths in `build.zig`:
```zig
exe.addIncludePath(.{ .cwd_relative = "/path/to/onnxruntime/include" });
exe.addLibraryPath(.{ .cwd_relative = "/path/to/onnxruntime/lib" });
```

## Next Steps: Optimization

Once GPU is working, further optimizations:
1. **Increase batch size** to 64-128 on GPU (better GPU utilization)
2. **Model quantization** (INT8) for 2-4x additional speedup
3. **TensorRT** execution provider for NVIDIA-optimized inference
4. **Smaller model** (3-layer instead of 6-layer) if accuracy permits

## Alternative: Use Python Baseline First

If Zig setup is difficult, first benchmark with Python to verify GPU works:

```python
from sentence_transformers import CrossEncoder
import time

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cuda')

# Test 1000 pairs
pairs = [["query", "document"]] * 1000

start = time.time()
scores = model.predict(pairs, batch_size=32)
elapsed = time.time() - start

print(f"1000 inferences: {elapsed*1000:.0f}ms")
print(f"Per inference: {elapsed*1000/1000:.2f}ms")
```

Expected: 50-200ms for 1000 docs on T4 GPU.
