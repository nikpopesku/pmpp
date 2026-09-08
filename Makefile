NVCC    ?= nvcc
ARCH    ?= sm_120          # RTX 5070 Mobile (Blackwell, GB206)
CCBIN   ?= g++-15          # CUDA 13.3 does not accept the system gcc 16
NVFLAGS := -O3 -arch=$(ARCH) -ccbin $(CCBIN) -diag-suppress 550

grayscale: grayscale.cu vendor/stb_image.h vendor/stb_image_write.h
	$(NVCC) $(NVFLAGS) -o $@ $<

# bit-exact match with the CPU reference (disables FMA contraction)
exact: NVFLAGS += -fmad=false
exact: clean grayscale

run: grayscale
	./grayscale $(IN) $(OUT)

clean:
	rm -f grayscale

.PHONY: run clean exact
