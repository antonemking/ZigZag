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

# 3b. Download C API headers (not included in pip package)
echo "Downloading ONNX Runtime C headers..."
ONNX_VERSION=$(python3 -c "import onnxruntime as ort; print(ort.__version__)")
echo "ONNX Runtime version: $ONNX_VERSION"

mkdir -p /tmp/onnxruntime_headers
cd /tmp/onnxruntime_headers
wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
tar -xzf "onnxruntime-linux-x64-${ONNX_VERSION}.tgz"
HEADER_DIR="/tmp/onnxruntime_headers/onnxruntime-linux-x64-${ONNX_VERSION}/include"

if [ ! -d "$HEADER_DIR" ]; then
    echo "ERROR: Failed to download headers"
    exit 1
fi

cd -

# 4. Find ONNX Runtime paths
echo "Locating ONNX Runtime..."
ONNX_PATH=$(python3 -c "import onnxruntime as ort; import os; print(os.path.dirname(ort.__file__))")
echo "ONNX Runtime path: $ONNX_PATH"

# Set paths: use downloaded headers + pip library
ONNX_INCLUDE="$HEADER_DIR"
ONNX_LIB="$ONNX_PATH/capi"

# Verify library exists
LIBONNX="$ONNX_LIB/libonnxruntime.so.1.23.0"
if [ ! -f "$LIBONNX" ]; then
    # Try to find it
    LIBONNX=$(find "$ONNX_PATH" -name "libonnxruntime.so*" 2>/dev/null | head -1)
    if [ -z "$LIBONNX" ]; then
        echo "ERROR: libonnxruntime.so not found"
        exit 1
    fi
    ONNX_LIB=$(dirname "$LIBONNX")
fi
echo "Found library: $LIBONNX"

# 5. Verify CUDA provider
echo "Verifying CUDA provider..."
python3 -c "import onnxruntime as ort; providers = ort.get_available_providers(); print(f'Available: {providers}'); assert 'CUDAExecutionProvider' in providers, 'CUDA not available!'"

# 6. Verify include headers exist
echo "Verifying include files..."
if [ ! -f "$ONNX_INCLUDE/onnxruntime_c_api.h" ]; then
    echo "ERROR: onnxruntime_c_api.h not found in $ONNX_INCLUDE"
    echo "Contents of $ONNX_INCLUDE:"
    ls -la "$ONNX_INCLUDE" || true
    exit 1
fi
echo "Found headers in: $ONNX_INCLUDE"

# 7. Create symlinks for versioned library
if [ -f "$ONNX_LIB/libonnxruntime.so.1.23.0" ]; then
    echo "Creating symlinks for libonnxruntime..."
    ln -sf libonnxruntime.so.1.23.0 "$ONNX_LIB/libonnxruntime.so.1"
    ln -sf libonnxruntime.so.1.23.0 "$ONNX_LIB/libonnxruntime.so"
fi

# 8. Set environment variables
export ONNX_INCLUDE="$ONNX_INCLUDE"
export ONNX_LIB="$ONNX_LIB"
export LD_LIBRARY_PATH="$ONNX_LIB:$LD_LIBRARY_PATH"
export COLAB_GPU=1
export USE_GPU=1

echo ""
echo "✅ Environment variables set:"
echo "  ONNX_INCLUDE=$ONNX_INCLUDE"
echo "  ONNX_LIB=$ONNX_LIB"
echo "  USE_GPU=1"
echo "  COLAB_GPU=1"

# 9. Save environment to file for future sessions
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
