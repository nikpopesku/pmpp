# PMPP

Working through **Programming Massively Parallel Processors: A Hands-on
Approach** (Hwu, Kirk & El Hajj) — CUDA implementations run on real hardware
against real data, not synthetic buffers.

Each chapter's code lives in its own folder with its own `Makefile`, so nothing
depends on anything else. Clone it, `cd` into a folder, `make`.

## Contents

| Folder | Book chapters | What it does |
|---|---|---|
| [`pmpp_grayscale/`](pmpp_grayscale/) | Ch. 2–3 | `colorToGrayscaleConversion` — RGB photo in, 8-bit grayscale PNG out, verified against a CPU reference |

Ch. 2 supplies the host-side pattern (`cudaMalloc` → `cudaMemcpy` → launch →
`cudaFree`); Ch. 3 supplies the 2D grid and the grayscale kernel itself.

## Hardware and toolchain

Everything here is developed and benchmarked on:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Max-Q / Mobile (Blackwell, GB206, 8 GB) |
| Compute capability | `sm_120` |
| CUDA | 13.3 |
| Host compiler | `g++-15` |
| OS | Arch Linux |

## Building on different hardware

Two knobs, both overridable per invocation:

```sh
make ARCH=sm_86        # match your card's compute capability
make CCBIN=g++-14      # pick a host compiler your CUDA release accepts
```

`ARCH` defaults to `sm_120` and must match your GPU. `CCBIN` exists because
CUDA 13.3 rejects Arch's system gcc 16; if your distro ships a gcc that your
CUDA supports, you can drop the flag.

## Conventions

- **Images are gitignored.** Inputs and outputs are local test data, so a fresh
  clone ships no photos — point the programs at any image on your machine.
- **Every kernel is checked against a CPU reference** on each run, and prints
  its own timing and effective bandwidth. Results are stated, not assumed.
- **Third-party code lives in `vendor/`** and is host-side only. Nothing in
  those files runs on the GPU.

## Note on floating point

The GPU and CPU results are not always bit-identical, and that is expected
rather than a bug. `nvcc` contracts expressions like `a*x + b*y + c*z` into FMA
instructions, which carry more intermediate precision than the CPU's separate
multiply-then-add. In the grayscale conversion this moves ~0.1% of pixels across
a rounding boundary, always by exactly 1/255. Building with `-fmad=false`
restores bit-exact agreement at roughly a 3% cost.
