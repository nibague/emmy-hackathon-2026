# 07 - Glossary

Alphabetical within each group. "See 0N" points to the entry that explains the term in context.

## Tools and languages

- **CUDA** - NVIDIA's platform for programming GPUs. "CUDA C++" is C++ with GPU extensions
  such as `__global__`, `threadIdx.x` and `__shared__`.
- **cubin** - the binary file `ptxas` produces. It holds the SASS for one GPU architecture.
- **Emmy** - the open-source compiler from PyTorch to GPU kernels that this hackathon is about.
  It produces the Kernel IR we read.
- **IR (intermediate representation)** - a language a compiler uses between its input and its
  output. See 01.
- **Kernel IR** - Emmy's last IR before code generation: one GPU kernel as a tree of statements.
  See 03.
- **nvcc** - NVIDIA's CUDA compiler driver. It turns `.cu` into PTX (front end), then calls
  `ptxas`. See 01.
- **PTX** - Parallel Thread eXecution: NVIDIA's text-based *virtual* assembly language.
  This compiler's output. See 01.
- **ptxas** - the PTX assembler: an optimizing compiler from PTX to SASS. It does the real
  register allocation.
- **runner** - the organizers' server (`runner.riftstack.ai`) with an RTX 5090 that assembles,
  runs and checks our PTX. `submit.py` talks to it.
- **SASS** - the real, architecture-specific GPU machine code. Not fully documented by NVIDIA.
- **virtual ISA** - an instruction set for an imaginary machine (PTX), translated later into the
  real one (SASS). ISA = "instruction set architecture".

## Architectures

- **compute capability / sm_XX** - a GPU generation's feature level. `sm_89` is Ada Lovelace
  (e.g. RTX 4090). `sm_120` is consumer Blackwell (RTX 5090, the runner).
- **`.target sm_89`** - PTX line that says which features the code may use. Our PTX targets
  sm_89 and the runner's `ptxas` compiles it for sm_120. See 01.
- **`.version 8.0`** - the PTX language version. sm_89 needs 7.8 or later.
- **`.address_size 64`** - pointers are 64-bit.

## Execution model (see 02)

- **block / CTA** - a group of up to 1024 threads that share shared memory and can use
  barriers. CTA = cooperative thread array.
- **divergence** - when threads of one warp take different branches. The warp runs both paths
  one after the other.
- **grid** - all blocks of one kernel launch.
- **kernel** - a function that runs on the GPU, once per thread.
- **lane** - a thread's position inside its warp, 0..31 (`tid & 31`).
- **launch geometry** - grid size x block size. Given by `meta.json`, never chosen by us.
- **thread** - one running instance of the kernel, with its own registers.
- **warp** - 32 threads that execute together. Shuffles work within a warp.

## Special registers

- **`%tid.x`** - thread index within the block (`threadIdx.x`).
- **`%ntid.x`** - threads per block (`blockDim.x`).
- **`%ctaid.x`** - block index within the grid (`blockIdx.x`).
- **`%nctaid.x`** - number of blocks (`gridDim.x`). Not used by this compiler.
- **gid** - global thread ID, `%ctaid.x * %ntid.x + %tid.x`.

## Memory and state spaces (see 02)

- **global memory** - the GPU's main memory (DRAM). Large, slow, visible to all threads.
  Accessed with `ld.global` / `st.global`.
- **`.param`** - the state space of kernel arguments. Read with `ld.param`.
- **`cvta.to.global`** - converts a generic address (how pointer arguments arrive) into a
  global-space address.
- **register** - per-thread storage, the fastest. In PTX: `%r`, `%rd`, `%f`, `%p` plus a number.
- **shared memory** - fast on-chip memory shared by one block. Declared with `.shared`, accessed
  with `ld.shared` / `st.shared`, 32-bit addresses.
- **state space** - PTX's word for "which kind of memory": `.reg`, `.param`, `.global`, `.shared`, ...
- **static vs dynamic shared memory** - static: size fixed in the PTX (`.shared ... [16]`, what we
  do). Dynamic: size given at launch (`smem_bytes` in meta.json, needed above 48 KiB).
- **vector load/store** - `ld.global.v4.f32` / `st.global.v2.f32`: move 2 or 4 consecutive values
  in one instruction. The address must be aligned (8 or 16 bytes).

## PTX instructions and syntax

- **`.reg .f32 %f<21>;`** - declares registers `%f0` .. `%f20`.
- **`.b32` / `.u32` / `.s32` / `.f32` / `.u64`** - type suffixes: raw bits, unsigned, signed,
  float, 64-bit unsigned.
- **immediate** - a constant written directly in an instruction: `add.s32 %r2, %r1, 4`.
- **`0f3F800000`** - how PTX writes a float immediate: the hex bits of the IEEE value (this is 1.0).
- **`mad.lo.s32 d, a, b, c`** - d = a*b + c, keeping the low 32 bits.
- **`mul.wide.s32 d, a, 4`** - multiply two 32-bit values into a 64-bit result (used for byte
  offsets).
- **predicate register** - a 1-bit true/false register (`%p1`), written by `setp`.
- **`setp.ge.s32 %p1, a, b`** - "set predicate": `%p1 = (a >= b)`. Others: `lt le gt eq ne`.
- **`@%p1 bra L` / `@!%p1 bra L`** - jump to label `L` if `%p1` is true / false. Any
  instruction can be guarded with `@%p`.
- **label** - `LOOP_1:`, a jump target.
- **`bar.sync 0`** - block-wide barrier: every thread waits until all threads of the block
  arrive. Also makes earlier shared-memory writes visible.
- **`shfl.sync.bfly.b32 d, a, k, 0x1f, mask`** - read register `a` from the lane
  `my_lane XOR k` in the same warp. `0x1f` is the lane clamp (whole warp). `mask` lists the
  lanes taking part (`0xffffffff` = all 32).
- **`ex2.approx.f32`, `lg2.approx.f32`, `rcp.rn.f32`** - fast 2^x, log2(x), 1/x. `.approx` is
  close but not exact. `.rn` is correctly rounded.
- **`red.global.add.f32`** - atomic add to global memory without returning the old value.
  Needed for m02. Not implemented yet.

## Compiler terms

- **accumulator** - a register that collects a running result (sum, max) across loop iterations.
- **barrier** - a point all threads must reach before any continues. See `bar.sync`.
- **butterfly** - the XOR pattern of `shfl.sync.bfly`. Five rounds combine 32 lanes.
- **deadlock** - threads waiting at a barrier for threads that will never arrive.
- **guard** - `if (gid >= N) skip`: stops extra threads when the launch exceeds the work.
- **identity** - the starting value of a reduction that doesn't change the result: 0 for sum,
  1 for product, -infinity (Emmy uses -1e30) for max.
- **pointwise / elementwise** - every output element depends only on the inputs at the same
  position. Category 1.
- **reduction** - combining many values into fewer (sum, mean, max). Category 2.
- **register allocation** - choosing which physical register holds each value. Done by `ptxas`.
  We only count virtual registers.
- **SSA (static single assignment)** - each value is defined once under its own name. The IR
  mostly works this way (`in0__u0`, `v0__u0`). Accumulators are the exception.
- **tolerance (rtol / atol)** - how far an output may be from the expected one and still pass:
  `|got - expected| <= atol + rtol * |expected|`.
- **unrolling** - writing a loop body out several times instead of looping (`__u0 .. __u3` in
  the IR).
