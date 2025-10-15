#!/bin/bash
# ZigZag Google Colab Setup Script
# Run this in a Colab notebook with GPU enabled

set -e

echo "=== ZigZag Colab Setup ==="

# 1. Check GPU
echo "Checking for GPU..."
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: No GPU found. Enable GPU in Runtime → Change runtime type"
    exit 1
fi
nvidia-smi | grep "GPU Name" || nvidia-smi | head -5

# 2. Install Zig
echo "Installing Zig 0.14.0..."
if ! command -v zig &> /dev/null; then
    wget -q https://ziglang.org/download/0.14.0/zig-linux-x86_64-0.14.0.tar.xz
    tar -xf zig-linux-x86_64-0.14.0.tar.xz
    ln -sf $(pwd)/zig-linux-x86_64-0.14.0/zig /usr/local/bin/zig
fi
zig version

# 3. Install ONNX Runtime GPU
echo "Installing ONNX Runtime GPU..."
pip install -q onnxruntime-gpu

# 4. Find ONNX Runtime paths
echo "Locating ONNX Runtime..."
ONNX_PATH=$(python3 -c "import onnxruntime as ort; import os; print(os.path.dirname(ort.__file__))")
echo "ONNX Runtime path: $ONNX_PATH"

# 5. Verify CUDA provider
echo "Verifying CUDA provider..."
python3 -c "import onnxruntime as ort; providers = ort.get_available_providers(); print(f'Available: {providers}'); assert 'CUDAExecutionProvider' in providers, 'CUDA not available!'"

# 6. Set environment variables
export ONNX_INCLUDE="$ONNX_PATH/capi/include"
export ONNX_LIB="$ONNX_PATH/capi/lib"
export LD_LIBRARY_PATH="$ONNX_LIB:$LD_LIBRARY_PATH"
export COLAB_GPU=1
export USE_GPU=1

echo "Environment variables set:"
echo "  ONNX_INCLUDE=$ONNX_INCLUDE"
echo "  ONNX_LIB=$ONNX_LIB"
echo "  USE_GPU=1"
echo "  COLAB_GPU=1"

# 7. Save environment to file for future sessions
cat > /tmp/zigzag_env.sh << EOF
export ONNX_INCLUDE="$ONNX_INCLUDE"
export ONNX_LIB="$ONNX_LIB"
export LD_LIBRARY_PATH="$ONNX_LIB:\$LD_LIBRARY_PATH"
export COLAB_GPU=1
export USE_GPU=1
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "To use ZigZag, run:"
echo "  source /tmp/zigzag_env.sh"
echo "  zig build"
echo "  zig build test"
