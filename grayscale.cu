// PMPP (Kirk & Hwu) -- Ch. 2 host-side CUDA pattern (cudaMalloc / cudaMemcpy /
// kernel launch / cudaFree) + Ch. 3 multidimensional grid and the
// colorToGrayscaleConversion kernel.
// Real image in (JPG/PNG), real 8-bit grayscale image out.

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb_image_write.h"

#define CHANNELS 3   // book uses 3 (RGB); we force stbi to give us 3

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d -> %s\n",                  \
                    cudaGetErrorName(_e), __FILE__, __LINE__,                  \
                    cudaGetErrorString(_e));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---- The kernel, straight out of the book (Fig. 3.2) -----------------------
// One thread per output pixel. Threads are laid out in a 2D grid that covers
// the image; the `if` guards the ragged edge where the grid overshoots.
__global__ void colorToGrayscaleConversion(unsigned char *Pout,
                                           const unsigned char *Pin,
                                           int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // Linear index of the gray pixel, and of the first of its 3 RGB bytes.
        int grayOffset = row * width + col;
        int rgbOffset  = grayOffset * CHANNELS;

        unsigned char r = Pin[rgbOffset + 0];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        // Luminance weights (ITU-R BT.601) as printed in the book.
        Pout[grayOffset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

// ---- CPU reference, so we can prove the GPU result is right ----------------
static void grayscaleCPU(unsigned char *out, const unsigned char *in,
                         int width, int height)
{
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int grayOffset = row * width + col;
            int rgbOffset  = grayOffset * CHANNELS;
            unsigned char r = in[rgbOffset + 0];
            unsigned char g = in[rgbOffset + 1];
            unsigned char b = in[rgbOffset + 2];
            out[grayOffset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input image> <output.png>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // ---- Load a real image; force 3 channels so the book's math applies ----
    int width, height, channelsInFile;
    unsigned char *h_in = stbi_load(argv[1], &width, &height,
                                    &channelsInFile, CHANNELS);
    if (!h_in) {
        fprintf(stderr, "failed to load '%s': %s\n", argv[1], stbi_failure_reason());
        return EXIT_FAILURE;
    }
    size_t nPixels  = (size_t)width * height;
    size_t rgbBytes = nPixels * CHANNELS;
    size_t grayBytes = nPixels;

    printf("input : %s  %dx%d (%d channels in file, using %d)\n",
           argv[1], width, height, channelsInFile, CHANNELS);
    printf("pixels: %zu  (%.2f MB RGB in, %.2f MB gray out)\n",
           nPixels, rgbBytes / 1048576.0, grayBytes / 1048576.0);

    unsigned char *h_out     = (unsigned char *)malloc(grayBytes);
    unsigned char *h_ref     = (unsigned char *)malloc(grayBytes);

    // ---- Device allocation + H2D copy -------------------------------------
    unsigned char *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  rgbBytes));
    CUDA_CHECK(cudaMalloc(&d_out, grayBytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, rgbBytes, cudaMemcpyHostToDevice));

    // ---- Launch config: 16x16 threads per block, grid ceil-divided --------
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width  + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y, 1);
    printf("launch: grid(%u,%u) x block(%u,%u) = %llu threads for %zu pixels\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y,
           (unsigned long long)gridDim.x * gridDim.y * blockDim.x * blockDim.y,
           nPixels);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warm-up so the timed run isn't paying for context/JIT setup
    colorToGrayscaleConversion<<<gridDim, blockDim>>>(d_out, d_in, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    colorToGrayscaleConversion<<<gridDim, blockDim>>>(d_out, d_in, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernelMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, start, stop));

    CUDA_CHECK(cudaMemcpy(h_out, d_out, grayBytes, cudaMemcpyDeviceToHost));

    // ---- Verify against the CPU -------------------------------------------
    grayscaleCPU(h_ref, h_in, width, height);
    size_t mismatches = 0;
    int    maxDiff    = 0;
    for (size_t i = 0; i < grayBytes; ++i) {
        int d = abs((int)h_out[i] - (int)h_ref[i]);
        if (d != 0) { ++mismatches; if (d > maxDiff) maxDiff = d; }
    }

    double gbMoved = (double)(rgbBytes + grayBytes) / 1e9;
    printf("kernel: %.4f ms  (%.1f GB/s effective, %.2f Gpixel/s)\n",
           kernelMs, gbMoved / (kernelMs / 1e3), nPixels / (kernelMs * 1e6));
    printf("verify: %zu / %zu bytes differ from CPU reference (max diff %d)\n",
           mismatches, grayBytes, maxDiff);

    // ---- Write the real output image --------------------------------------
    if (!stbi_write_png(argv[2], width, height, 1, h_out, width)) {
        fprintf(stderr, "failed to write '%s'\n", argv[2]);
        return EXIT_FAILURE;
    }
    printf("output: %s written\n", argv[2]);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_out); free(h_ref);
    stbi_image_free(h_in);
    return EXIT_SUCCESS;
}
