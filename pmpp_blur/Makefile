NVCC    ?= nvcc
ARCH    ?= sm_120          # RTX 5070 Mobile (Blackwell, GB206)
CCBIN   ?= g++-15          # CUDA 13.3 does not accept the system gcc 16
NVFLAGS := -O3 -arch=$(ARCH) -ccbin $(CCBIN) -diag-suppress 550

blur: blur.cu vendor/stb_image.h vendor/stb_image_write.h
	$(NVCC) $(NVFLAGS) -o $@ $<

run: blur
	./blur $(IN) $(OUT)

clean:
	rm -f blur

.PHONY: run clean
