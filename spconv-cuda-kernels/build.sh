#!/bin/bash
# Build script for SPConv CUDA Kernels

set -e  # Exit on error

echo "Building SPConv CUDA Kernels..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON ..

# Build
echo "Building..."
make -j$(nproc)

echo "Build complete!"
echo ""
echo "To run tests:"
echo "  cd build/tests"
echo "  ./test_simple"
echo ""
echo "To use in your project, link against:"
echo "  $(pwd)/libspconv.a"