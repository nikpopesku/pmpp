// PMPP (Kirk & Hwu) -- Ch. 3 image blur kernel (the book's follow-up example
// to colorToGrayscaleConversion, wired up to real image I/O).
// Real image in (JPG/PNG), real 8-bit blurred grayscale image out.

#include <cstdio>
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#include "../vendor/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../vendor/stb_image_write.h"

#define CHANNELS 1   // book's blur example works on a single-channel image

#ifndef BLUR_SIZE
#define BLUR_SIZE 8  // averages over a (2*BLUR_SIZE+1)^2 box; 8 -> 17x17
                     // (book's own example uses 1 -> 3x3, but that's too
                     // subtle to see at a glance on a multi-megapixel photo)
#endif

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

// ---- The kernel, straight out of the book -----------------------------
// One thread per output pixel. Each thread averages the BLUR_SIZE-radius box
// around its pixel, shrinking the divisor near the border instead of
// treating out-of-bounds neighbors as zero.
__global__ void blurKernel(const unsigned char *in, unsigned char *out,
                           int w, int h)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;

        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    pixVal += in[curRow * w + curCol];
                    ++pixels;
                }
            }
        }

        out[row * w + col] = (unsigned char)(pixVal / pixels);
    }
}

// ---- CPU reference, so we can prove the GPU result is right ----------------
static void blurCPU(unsigned char *out, const unsigned char *in, int w, int h)
{
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            int pixVal = 0;
            int pixels = 0;

            for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
                for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;

                    if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                        pixVal += in[curRow * w + curCol];
                        ++pixels;
                    }
                }
            }

            out[row * w + col] = (unsigned char)(pixVal / pixels);
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <input image> <output.png>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // ---- Load a real image, forced to 1 channel: the book's blurKernel -----
    // assumes a grayscale input with one byte per pixel.
    int width, height, channelsInFile;
    unsigned char *h_in = stbi_load(argv[1], &width, &height,
                                    &channelsInFile, CHANNELS);
    if (!h_in) {
        fprintf(stderr, "failed to load '%s': %s\n", argv[1], stbi_failure_reason());
        return EXIT_FAILURE;
    }
    size_t nPixels = (size_t)width * height;
    size_t nBytes  = nPixels * CHANNELS;

    printf("input : %s  %dx%d (%d channels in file, using %d)\n",
           argv[1], width, height, channelsInFile, CHANNELS);
    printf("pixels: %zu  (%.2f MB in/out, blur box %dx%d)\n",
           nPixels, nBytes / 1048576.0, 2 * BLUR_SIZE + 1, 2 * BLUR_SIZE + 1);

    unsigned char *h_out = (unsigned char *)malloc(nBytes);
    unsigned char *h_ref = (unsigned char *)malloc(nBytes);

    // ---- Device allocation + H2D copy -------------------------------------
    unsigned char *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  nBytes));
    CUDA_CHECK(cudaMalloc(&d_out, nBytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice));

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
    blurKernel<<<gridDim, blockDim>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    blurKernel<<<gridDim, blockDim>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernelMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, start, stop));

    CUDA_CHECK(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost));

    // ---- Verify against the CPU -------------------------------------------
    blurCPU(h_ref, h_in, width, height);
    size_t mismatches = 0;
    int    maxDiff    = 0;
    for (size_t i = 0; i < nBytes; ++i) {
        int d = abs((int)h_out[i] - (int)h_ref[i]);
        if (d != 0) { ++mismatches; if (d > maxDiff) maxDiff = d; }
    }

    double gbMoved = (double)(nBytes * 2) / 1e9;
    printf("kernel: %.4f ms  (%.1f GB/s effective, %.2f Gpixel/s)\n",
           kernelMs, gbMoved / (kernelMs / 1e3), nPixels / (kernelMs * 1e6));
    printf("verify: %zu / %zu bytes differ from CPU reference (max diff %d)\n",
           mismatches, nBytes, maxDiff);

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
