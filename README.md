# PMPP grayscale (CUDA)

`colorToGrayscaleConversion` from *Programming Massively Parallel Processors*
(Kirk & Hwu) — Ch. 2 host-side CUDA pattern, Ch. 3 multidimensional grid —
wired up to real image I/O via [stb](https://github.com/nothings/stb).

Real photo in (JPG/PNG/…), real 8-bit grayscale PNG out, checked against a CPU
reference on every run.

## Build

```sh
make                     # defaults: -arch=sm_120, -ccbin g++-15
make ARCH=sm_86          # override for your GPU
```

`ARCH` must match your card. `CCBIN` exists because CUDA 13.3 rejects gcc 16.

## Run

```sh
./grayscale input.jpg output.png
```

```
input : cheetah.jpg  4462x2512 (3 channels in file, using 3)
pixels: 11208544  (32.07 MB RGB in, 10.69 MB gray out)
launch: grid(279,157) x block(16,16) = 11213568 threads for 11208544 pixels
kernel: 0.1866 ms  (240.3 GB/s effective, 60.08 Gpixel/s)
verify: 10746 / 11208544 bytes differ from CPU reference (max diff 1)
output: output.png written
```

## Why the CPU reference differs by 1

The default build lets the compiler contract `0.21f*r + 0.71f*g + 0.07f*b` into
FMA instructions, which keep more intermediate precision than the CPU's
separate multiply-then-add. Roughly 0.1% of pixels land on the other side of a
rounding boundary, always by exactly 1/255.

```sh
make exact               # -fmad=false -> 0 / 11208544 bytes differ
```

That build is ~3% slower and bit-identical to the CPU.
