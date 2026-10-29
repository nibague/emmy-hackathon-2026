# 02 - The GPU execution model

A CPU runs a few threads very fast. A GPU runs **thousands of threads at once**, each doing a
small piece of the work. A **kernel** is the function every one of those threads runs. All
threads run the *same* code; each one uses its own ID to pick *which* data to work on.

## Threads, warps, blocks, grids

```
grid  (all the threads of one kernel launch)
+-------------------------------------------------------------+
| block 0                | block 1                | block 2 ...|
| +------+------+-----+  | +------+------+-----+  |            |
| |warp 0|warp 1| ... |  | |warp 0|warp 1| ... |  |            |
| |32 thr|32 thr|     |  | |32 thr|32 thr|     |  |            |
| +------+------+-----+  | +------+------+-----+  |            |
+-------------------------------------------------------------+
```

| level | what it is | size |
|---|---|---|
| **thread** | one instance of the kernel, with its own registers | 1 |
| **warp** | 32 threads that execute instructions together, in lockstep | always 32 |
| **block** (also **CTA**, "cooperative thread array") | a group of threads that can share fast memory and wait for each other | up to 1024 threads |
| **grid** | all blocks of one launch | as many blocks as needed |

Threads in the *same block* can cooperate through shared memory and barriers. Threads in
*different blocks* cannot synchronize with each other inside a kernel.

## How a thread knows who it is

PTX gives every thread read-only **special registers**:

| PTX | CUDA C++ name | meaning |
|---|---|---|
| `%tid.x` | `threadIdx.x` | my index inside my block (0 .. block size - 1) |
| `%ntid.x` | `blockDim.x` | how many threads per block |
| `%ctaid.x` | `blockIdx.x` | which block I am in (0 .. number of blocks - 1) |
| `%nctaid.x` | `gridDim.x` | how many blocks |

Each has `.y` and `.z` versions too, but every problem here uses only `.x`.

The usual **global thread ID** combines them:

```
gid = blockIdx.x * blockDim.x + threadIdx.x
```

```ptx
mov.u32    %r1, %ctaid.x;
mov.u32    %r2, %ntid.x;
mov.u32    %r3, %tid.x;
mad.lo.s32 %r4, %r1, %r2, %r3;    // %r4 = %r1 * %r2 + %r3  ("multiply-add")
```

Example: with blocks of 256 threads, thread 5 of block 2 has `gid = 2*256 + 5 = 517`.

Two more IDs that reductions use:

```
lane = threadIdx.x % 32    -> and.b32 lane, tid, 31   (my position inside my warp)
warp = threadIdx.x / 32    -> shr.u32 warp, tid, 5    (which warp of the block I am in)
```

## meta.json decides the launch

The compiler never chooses how many threads run. Each problem's `meta.json` fixes it, and the
runner launches exactly that:

```json
p00_scalar_add:  "grid": [1, 1, 1],  "block": [256, 1, 1]    -> 1 block x 256 threads = 256
p02_scale_shift: "grid": [4, 1, 1],  "block": [256, 1, 1]    -> 4 blocks x 256       = 1024
r01_array_sum:   "grid": [1, 1, 1],  "block": [128, 1, 1]    -> 1 block x 128        = 128
r04_row_sum:     "grid": [64, 1, 1], "block": [128, 1, 1]    -> 64 blocks x 128      = 8192
```

The IR's `Tile` says how many "cells" of work there are (`Tile[a0] (N=64)`). If the launch
has **more** threads than cells, the extra threads must do nothing:

- p00: 256 threads, N=64. Threads 64..255 must skip everything, so the compiler emits a
  **guard**: `if (gid >= 64) jump to end`.
- p02: 1024 threads, N=1024. No guard needed, none emitted.

The guard is not in the IR. The compiler derives it from N and the launch size.

## Three kinds of memory (plus parameters)

| memory | PTX state space | who sees it | speed | size | lifetime |
|---|---|---|---|---|---|
| **registers** | `.reg` | one thread | fastest | a few hundred bytes per thread | the thread |
| **shared** | `.shared` | all threads of one block | fast, on-chip | up to 99 KiB per block on the runner's GPU (docs/API.md) | the block |
| **global** | `.global` | every thread, and the CPU | slow (off-chip DRAM) | gigabytes | until freed |
| **parameters** | `.param` | the kernel's arguments | - | a few bytes each | the launch |

How they show up in our PTX:

```ptx
.reg .f32 %f<7>;                          // registers: declare 7 float registers
.shared .align 4 .b8 acc0_smem[16];       // shared: 16 bytes for the whole block
ld.param.u64       %rd1, [param_x];       // parameter: read the pointer argument
cvta.to.global.u64 %rd2, %rd1;            // turn it into a global-memory address
ld.global.f32      %f1, [%rd2];           // global: read a float from the input buffer
st.shared.f32      [%r20], %f1;           // shared: write a float into shared memory
```

`cvta.to.global` ("convert address") is needed because pointer arguments arrive as *generic*
addresses, and `ld.global`/`st.global` want *global* ones.

Global addresses are 64-bit (`%rd` registers). Shared addresses fit in 32 bits (`%r` registers).

## Why this matters for the compiler

- **Pointwise** kernels (category 1): each thread is independent. It reads a few elements,
  computes, writes. Only registers and global memory are needed.
- **Reductions** (category 2): many threads must combine their results into one number. That
  needs warps exchanging registers (`shfl.sync`), shared memory, and barriers (`bar.sync`) so
  threads wait for each other. See [06-reduction-walkthrough.md](06-reduction-walkthrough.md).
