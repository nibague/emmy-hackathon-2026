# 06 - Reduction walkthrough: r01_array_sum

A **reduction** turns many values into one: a sum, a max, a mean. Unlike pointwise, the
threads **must cooperate**. Thread 5 alone cannot know the total.

## The problem

```
code:      torch.randn(1024).sum()
grid:      [1, 1, 1]     block: [128, 1, 1]     -> 1 block of 128 threads = 4 warps
arg_order: [x, sum_1]
tolerance: rtol = atol = 1e-4
```

## The strategy: three levels

```
 1024 floats in global memory
      |
      |  Level 1: each thread sums its own 8 elements (strided loop, in a register)
      v
 128 partial sums (one per thread, in registers)
      |
      |  Level 2: each warp combines its 32 values by exchanging registers (shfl.sync)
      v
 4 partial sums (one per warp) -> lane 0 of each warp writes it to shared memory
      |
      |  bar.sync: wait until all 4 are written
      |
      |  Level 3: tree in shared memory: 4 -> 2 -> 1, with a bar.sync after each step
      v
 1 total -> thread 0 writes it to global memory
```

Each level uses the fastest tool that can reach the threads involved: a register for one
thread, a shuffle within a warp, shared memory across warps of a block.

## The IR (`k_sum_1_reduce.ir.txt`)

```
Tile[a0_co] (N=128)
    for a0 in a0_co..1024:128
        in0 = load x[a0]
        f32 acc0 <- add(acc0, in0)
    WarpShuffle(acc0, length=32)
    Smem float acc0_smem[4]
    if ((lane == 0)):
        acc0_smem[warp] = acc0
    Sync
    TreeHalve(acc0, length=4, tid=warp)
    if ((a0_co == 0)):
        sum_1[] = acc0
```

## The generated PTX, annotated

The real output of `compile.py` for r01. Comments were added here by hand.

### Header: the shared array lives at module scope

```ptx
.version 8.0
.target sm_89
.address_size 64

.shared .align 4 .b8 acc0_smem[16];      // Smem float acc0_smem[4]: 4 floats = 16 bytes,
                                         // one copy per block, shared by its 128 threads
.visible .entry k_sum_1_reduce(
	.param .u64 param_x,
	.param .u64 param_sum_1
)
{
	.reg .pred %p<6>;                    // predicates: true/false values for branches
	.reg .b32 %r<28>;
	.reg .b64 %rd<7>;
	.reg .f32 %f<8>;
	... ld.param / cvta for x (%rd2) and sum_1 (%rd4), as in 05 ...
	mov.u32 %r1, %ctaid.x;
	mov.u32 %r2, %ntid.x;
	mov.u32 %r3, %tid.x;
	mad.lo.s32 %r4, %r1, %r2, %r3;       // a0_co = gid (0..127)
	and.b32 %r5, %r3, 31;                // lane = tid % 32
	shr.u32 %r6, %r3, 5;                 // warp = tid / 32   (0..3)
```

No guard: 128 threads launched, N = 128. A guard here would also be **dangerous**: see
"Why the guard matters" below.

### Level 1: `StridedLoop` + `Accum`

```ptx
	mov.f32 %f1, 0f00000000;             // acc0 = 0.0 (identity of add), BEFORE the loop
	mov.s32 %r7, %r4;                    // a0 = a0_co (start)
LOOP_1:
	setp.ge.s32 %p1, %r7, 1024;          // %p1 = (a0 >= 1024)
	@%p1 bra LOOP_END_2;                 // if so, leave the loop
	mul.wide.s32 %rd5, %r7, 4;
	add.s64 %rd6, %rd2, %rd5;
	ld.global.f32 %f2, [%rd6];           // in0 = x[a0]
	add.f32 %f1, %f1, %f2;               // acc0 += in0   (same register every iteration)
	add.s32 %r7, %r7, 128;               // a0 += 128 (step)
	bra LOOP_1;                          // jump back
LOOP_END_2:
```

PTX has no `for` keyword. A loop is a **label**, a **test** (`setp` writes a predicate), a
**conditional branch** (`@%p1 bra` jumps only if `%p1` is true), and an **unconditional
jump back**.

### Level 2: `WarpShuffle(acc0, length=32)`

```ptx
	mov.b32 %r8, %f1;                                      // float bits -> integer register
	shfl.sync.bfly.b32 %r9, %r8, 16, 0x1f, 0xffffffff;     // get acc0 from lane (my lane XOR 16)
	mov.b32 %f3, %r9;                                      // bits -> float
	add.f32 %f1, %f1, %f3;                                 // acc0 += partner's acc0
	... same with 8, 4, 2, 1 ...
```

`shfl.sync.bfly` ("butterfly") lets each thread read a register of *another thread in the
same warp*, the one whose lane is `my_lane XOR offset`. After 5 rounds (16, 8, 4, 2, 1),
every one of the 32 lanes holds the warp's total:

```
round offset 16: lane 0 + lane 16, lane 1 + lane 17, ...  -> 16 distinct partial sums
round offset  8: ...                                      -> 8
...
round offset  1:                                          -> 1 (in every lane)
```

`0xffffffff` says all 32 lanes take part. `shfl` moves 32-bit bit patterns, hence the
`mov.b32` round trips between float and integer registers.

### Write each warp's sum to shared memory

```ptx
	setp.eq.s32 %p2, %r5, 0;             // %p2 = (lane == 0)
	@!%p2 bra ENDIF_3;                   // skip if NOT lane 0
	mov.u32 %r18, acc0_smem;             // shared address of the array (32-bit)
	mul.lo.s32 %r19, %r6, 4;             // warp * 4 bytes
	add.s32 %r20, %r18, %r19;
	st.shared.f32 [%r20], %f1;           // acc0_smem[warp] = acc0
ENDIF_3:
	bar.sync 0;                          // Sync: nobody continues until all 128 threads arrive
```

**Why `bar.sync`?** Without it, warp 0 could start reading `acc0_smem[1]` before warp 1 has
written it. Threads in different warps run at different speeds. `bar.sync 0` is a barrier: each
thread stops there until *every* thread of the block has reached it, and their shared-memory
writes become visible to everyone.

### Level 3: `TreeHalve(acc0, length=4, tid=warp)`

```ptx
	setp.ge.s32 %p3, %r6, 2;             // step s=2: only warps 0 and 1 work
	@%p3 bra TREE_SKIP_4;
	mov.u32 %r21, acc0_smem;
	mul.lo.s32 %r22, %r6, 4;
	add.s32 %r23, %r21, %r22;            // &acc0_smem[warp]
	ld.shared.f32 %f4, [%r23];           //   acc0_smem[warp]
	ld.shared.f32 %f5, [%r23+8];         //   acc0_smem[warp + 2]   (+8 bytes)
	add.f32 %f4, %f4, %f5;
	st.shared.f32 [%r23], %f4;           // acc0_smem[warp] = sum of the pair
TREE_SKIP_4:
	bar.sync 0;                          // wait before the next step reads the results
	setp.ge.s32 %p4, %r6, 1;             // step s=1: only warp 0 works
	@%p4 bra TREE_SKIP_5;
	... acc0_smem[0] = acc0_smem[0] + acc0_smem[1] ...
TREE_SKIP_5:
	bar.sync 0;
	mov.u32 %r27, acc0_smem;
	ld.shared.f32 %f1, [%r27];           // broadcast: every thread reads the total into acc0
```

The tree as a picture:

```
slots:   [ w0 ][ w1 ][ w2 ][ w3 ]
s=2:     [w0+w2][w1+w3]                 bar.sync
s=1:     [w0+w1+w2+w3]                  bar.sync
all threads: acc0 = slot 0
```

All 32 threads of warp 0 do the step-1 work and write the same value to the same address.
That is harmless.

### The final write

```ptx
	setp.eq.s32 %p5, %r4, 0;             // if (a0_co == 0): only thread 0
	@!%p5 bra ENDIF_6;
	st.global.f32 [%rd4], %f1;           // sum_1[] = acc0  (empty index = element 0)
ENDIF_6:
	ret;
}
```

## Result

```
$ .venv/bin/python submit.py r01_array_sum --compiler compile.py
[k_sum_1_reduce] 18 registers
[k_sum_1_reduce] PASS  sum_1: max abs err 3.815e-06
```

The error is not 0 because floating-point addition is not associative: `(a+b)+c` can differ
from `a+(b+c)` in the last bits. Our order (strided, then butterfly, then tree) differs from
numpy's, which is why reductions get a tolerance (`1e-4` here).

## Why the guard matters

In pointwise code, extra threads are skipped with a guard: `if (gid >= N) jump to the end`.
Here that would break things. A thread that jumps past `bar.sync` never arrives at the barrier,
and the others may wait forever (a **deadlock**), or the results are undefined. Likewise
`shfl.sync` with mask `0xffffffff` expects all 32 lanes to be present. So the compiler:

- emits no guard when the launch has exactly N threads (true for every reduction here);
- **refuses to compile** (raises an error) if a guard would be needed around a barrier or shuffle.

## Why reductions are harder than pointwise

| | pointwise | reduction |
|---|---|---|
| threads talk to each other? | never | yes, through shuffles and shared memory |
| ordering between threads | none needed | `bar.sync` needed, or results are wrong |
| control flow | straight line (+ guard) | loops and branches |
| register lifetime | each value used once | the accumulator lives across loop iterations, so it must keep one register |
| failure modes | wrong index -> wrong value | also deadlocks, races, silently wrong totals |
| result | exact | depends on summation order -> tolerance |

r04 (row sum) uses the same machinery with a 2-D Tile: block `b` handles row `b`, and the index
`x[a0, a1]` becomes `a0*512 + a1`. See [13-log-reductions.md](13-log-reductions.md).
