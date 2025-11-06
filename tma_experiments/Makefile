# Makefile for TMA experiments

NVCC = nvcc
ARCH = sm_90  # Use sm_90 for Hopper (H100/H200)
CXXFLAGS = -O3 -std=c++17
LDFLAGS = -lcuda

# Targets
all: tma_microbench test_tma_descriptor deform_attn_optimized deform_attn_tma

# Microbenchmark - tests 2x2x32 tile loading pattern
tma_microbench: tma_2x2x32_final.cu
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $< -o $@

# TMA descriptor test - verifies TMA API works on Hopper
test_tma_descriptor: test_tma_descriptor.cu
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $(LDFLAGS) $< -o $@

# Deformable attention with 2x2x32 tile loading (object file)
deform_attn_optimized: deform_attn_2x2x32_optimized.cu
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) -c $< -o deform_attn_2x2x32_optimized.o

# Deformable attention with real TMA (object file)
deform_attn_tma: deform_attn_tma.cu
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $(LDFLAGS) -c $< -o deform_attn_tma.o

# Baseline reference (object file)
deform_attn_baseline: deform_attn.cu
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) -c $< -o deform_attn_baseline.o

clean:
	rm -f *.o tma_microbench test_tma_descriptor

test: tma_microbench test_tma_descriptor
	@echo "=== Running tile loading microbenchmark ==="
	./tma_microbench
	@echo ""
	@echo "=== Testing TMA descriptor creation ==="
	./test_tma_descriptor

.PHONY: all clean test
