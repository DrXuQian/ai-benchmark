# Makefile for TMA experiments

NVCC = nvcc
ARCH = sm_90  # Change to sm_90 for Hopper
CXXFLAGS = -O3 -std=c++17
LDFLAGS = -lcuda

# Targets
all: tma_microbench deform_attn_optimized

# Microbenchmark
tma_microbench: tma_2x2x32_final.cu
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) $< -o $@

# Deformable attention optimized kernel (object file)
deform_attn_optimized: deform_attn_2x2x32_optimized.cu
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) -c $< -o deform_attn_2x2x32_optimized.o

# Baseline reference (object file)
deform_attn_baseline: deform_attn.cu
	$(NVCC) -arch=$(ARCH) $(CXXFLAGS) -c $< -o deform_attn_baseline.o

clean:
	rm -f *.o tma_microbench

test: tma_microbench
	./tma_microbench

.PHONY: all clean test
