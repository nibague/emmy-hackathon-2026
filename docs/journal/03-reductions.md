# 03 - Reductions (r01-r04): loops, shuffles, shared memory, barriers

New nodes: `StridedLoop`, `Accum`, `WarpShuffle`, `Smem`, `Cond`, `Sync`, `TreeHalve`.
New Tile features: several axes, the implicit `lane` / `warp` names, multi-dim indices.

## The IR

r01 / r02 / r03 have the same shape (r02 = 65536 elements, r03 divides by a constant at
the end); r04 adds a second Tile axis:

```
Tile[a0, a1_co] (N=8192)                 # 64 rows x 128 threads per row; grid 64, block 128
    for a1 in a1_co..512:128             # StridedLoop: each thread covers 4 columns
        in0 = load x[a0, a1]
        f32 acc0 <- add(acc0, in0)       # Accum
    WarpShuffle(acc0, length=32)         # per-warp sum in registers
    Smem float acc0_smem[4]              # one slot per warp
    if ((lane == 0)):
        acc0_smem[warp] = acc0
    Sync
    TreeHalve(acc0, length=4, tid=warp)  # combine the 4 warp partials, broadcast
    if ((a1_co == 0)):
        sum_1[a0] = acc0
```

r03 ends with `in1 = load mean_keepdim_count[0]; v0 = divide(acc0, in1); mean[] = v0`.
That is all existing machinery (ConstantOp buffer + `divide`).

## Translation, node by node

| IR | PTX | notes |
|---|---|---|
| `Tile[a0, a1_co]` | `gid = ctaid*ntid + tid`; `rem.u32 a1_co, gid, 128`; `div.u32 a0, gid, 128` | row-major decode, last axis fastest (same as ref.cu) |
| `lane`, `warp` | `and.b32 lane, tid, 31`; `shr.u32 warp, tid, 5` | defined in every Tile prologue: the IR uses them without defining them |
| guard | **omitted** when `grid*block == N` | see "the guard and barriers" below |
| `StridedLoop` | `mov var, start; LOOP: setp.ge.s32 p, var, end; @p bra END; body; add var, var, step; bra LOOP; END:` | `end=None` means the axis extent (512) |
| `Accum` | `mov.f32 acc, <identity>` **before** the loop, then `add.f32 acc, acc, in0` inside | identity comes from emmy's `op.identity` (0 for add, -1e30 for maximum, +1e30 for minimum) |
| `x[a0, a1]` | `mul.lo.s32 t, a0, 512; add.s32 i, t, a1` | `flat_index` builds `a0*512 + a1` as a `BinaryExpr` from the buffer's shape, then reuses `compile_index` |
| `WarpShuffle(length=32)` | for k in 16,8,4,2,1: `mov.b32 r, acc; shfl.sync.bfly.b32 r2, r, k, 0x1f, 0xffffffff; mov.b32 acc__o, r2;` then the node's `combine_states` (`add.f32 acc, acc, acc__o`) | the combine is not hardcoded to add: it is the IR's own Assign list |
| `Smem float acc0_smem[4]` | module scope `.shared .align 4 .b8 acc0_smem[16];` | Load/Write on a name in `self.smem` switch to `ld.shared` / `st.shared` with 32-bit addresses from `mov.u32 %r, acc0_smem` |
| `Cond (lane == 0)` | `setp.eq.s32 p, lane, 0; @!p bra ENDIF;` body `ENDIF:` | comparisons -> `setp.<cmp>.s32`, `&&`/`||` -> `and.pred`/`or.pred`; else_body gets its own label |
| `Sync` | `bar.sync 0;` (`bar.sync id, count` for named barriers, `bar.warp.sync` for warp scope) | |
| `TreeHalve(length=4, tid=warp)` | unrolled, for s = 2, 1: `setp.ge.s32 p, warp, s; @p bra SKIP; ld.shared a,[buf+warp*4]; ld.shared b,[..+4*s]; combine; st.shared; SKIP: bar.sync 0;` then every thread `ld.shared acc0, [buf]` | matches ref.cu's loop + broadcast. Not the worked example's "thread 0 adds all 4" shortcut |

### Registers that must survive control flow

New rule in `bind_value`: when an SSA name is defined again (Accum updates, shuffle
combines `acc0 = acc0 + acc0__o`), it **reuses its register**. A loop body is emitted once
and runs many times, so `acc0` must live in the same register on every iteration. The
same goes for values crossing a Cond. Fresh-register-per-definition (what pointwise used)
would be wrong here.

TreeHalve binds the state names to temporaries while it combines, then points them back at
the original register and loads `buf[0]` into it. Code after the tree sees `acc0` in
the same place as before.

### The guard and barriers

Pointwise launches more threads than cells (p00: 256 threads, 64 cells) and needs
`if (gid < N)`. In the reductions the launch is exactly N threads, and the body has
`bar.sync` and full-mask `shfl.sync`. Branching some threads around those would hang or
give undefined results. So the guard is emitted only when `grid*block > N` (from
meta.json). If a guard is needed *and* the body has a barrier or shuffle, the compiler
raises instead of emitting PTX that could deadlock.

## r04 PTX (shuffle ladder omitted)

```
	mad.lo.s32 %r4, %r1, %r2, %r3;      // gid
	and.b32 %r5, %r3, 31;               // lane
	shr.u32 %r6, %r3, 5;                // warp
	rem.u32 %r7, %r4, 128;              // a1_co
	div.u32 %r8, %r4, 128;              // a0
	mov.f32 %f1, 0f00000000;            // acc0 = identity(add)
	mov.s32 %r9, %r7;                   // a1 = a1_co
LOOP_1:
	setp.ge.s32 %p1, %r9, 512;
	@%p1 bra LOOP_END_2;
	mul.lo.s32 %r10, %r8, 512;          // a0*512 + a1
	add.s32 %r11, %r10, %r9;
	mul.wide.s32 %rd5, %r11, 4;
	add.s64 %rd6, %rd2, %rd5;
	ld.global.f32 %f2, [%rd6];
	add.f32 %f1, %f1, %f2;              // Accum
	add.s32 %r9, %r9, 128;
	bra LOOP_1;
LOOP_END_2:
	... 5 x shfl.sync.bfly + add.f32 %f1, %f1, %f3
	setp.eq.s32 %p2, %r5, 0;            // if lane == 0
	@!%p2 bra ENDIF_3;
	mov.u32 %r22, acc0_smem;
	mul.lo.s32 %r23, %r6, 4;
	add.s32 %r24, %r22, %r23;
	st.shared.f32 [%r24], %f1;          // acc0_smem[warp] = acc0
ENDIF_3:
	bar.sync 0;
	setp.ge.s32 %p3, %r6, 2;            // tree step s=2
	@%p3 bra TREE_SKIP_4;
	...  ld.shared [%r27], [%r27+8]; add.f32; st.shared
TREE_SKIP_4:
	bar.sync 0;
	...  tree step s=1
TREE_SKIP_5:
	bar.sync 0;
	mov.u32 %r31, acc0_smem;
	ld.shared.f32 %f1, [%r31];          // broadcast acc0 = buf[0]
	setp.eq.s32 %p5, %r7, 0;            // if a1_co == 0
	@!%p5 bra ENDIF_6;
	... st.global.f32 [sum_1 + a0*4], %f1;
ENDIF_6:
	ret;
```

## submit.py result

```
p00_scalar_add   PASS  max abs err 0.000e+00
p01_vector_add   PASS  max abs err 0.000e+00
p02_scale_shift  PASS  max abs err 0.000e+00
p03_relu_mul     PASS  max abs err 0.000e+00
r01_array_sum    PASS  max abs err 3.815e-06   (18 registers)
r02_big_sum      PASS  max abs err 1.526e-05
r03_mean         PASS  max abs err 9.313e-10
r04_row_sum      PASS  max abs err 5.722e-06   (64 blocks)
```

All eight MVP problems pass. Pointwise still passes after the refactor.

### Other reduce ops (hidden-set insurance)

A scratch harness replaced every `ElementwiseImpl('add')` in r04's IR (the Accum and the
shuffle/tree combines) with `maximum` / `minimum`. It ran with explicit numpy data through
the runner and compared against `x.max(-1)` / `x.min(-1)`:

```
add      max abs err 5.28e-06  ok
maximum  max abs err 0.00e+00  ok
minimum  max abs err 0.00e+00  ok
```

So the identity init (-1e30 / +1e30 from emmy) and the combine-by-IR-Assign path work for
ops other than add.

## What failed and why

Nothing failed under submit.py on the first run. The only failure was in my own test
harness: its first version had no request timeout or busy-retry, so it hung. Then an upload of
the 32k-float input timed out on the network. Adding timeouts and a retry loop (as submit.py
does) fixed it. The compiler was not involved.

Things I was careful about:

- Accumulator init placement: it is set before the loop label, not inside the loop
  (inside, it would reset every iteration).
- TreeHalve correctness with several warps writing: in step s, warps with `warp < s`
  update `buf[warp]`. All 32 lanes of such a warp write the *same* value to the *same*
  address, which is a benign race. The `bar.sync` after each step orders the steps.
- `[%r27+8]`: PTX allows `[reg+imm]` addressing, so `buf[tid + s]` needs no extra add.
