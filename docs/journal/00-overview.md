# 00 - Overview: how Kernel IR maps to PTX

No compiler code yet. This entry is the mental model I'll build on.

## The task in one paragraph

`compile.py kernel.ir.json > out.ptx`. The IR json is a small graph: `InputOp` nodes (the
buffers), maybe `ConstantOp` nodes (scalars that arrive as 1-element buffers), and one
`KernelOp` whose `body` is a tree of statements (`Tile`, `Load`, `Assign`, `Write`, ...).
`Graph.from_dict` turns the body string into typed Python objects, so I never parse it myself.
I walk that tree and print one PTX snippet per statement. The PTX must:

- use the IR's kernel `name` as the `.entry` name,
- take parameters in `meta.json`'s `arg_order` (each buffer is a `.param .u64` pointer),
- be correct under `meta.json`'s grid/block (which may have *more* threads than work),
- match `expected.npz` within `rtol`/`atol`.

MVP = categories 1 (pointwise: p00-p03) and 2 (reductions: r01-r04).

## The shape of every PTX file

```
.version 8.0            // sm_89 needs >= 7.8
.target sm_89
.address_size 64
[.shared arrays]        // one per IR Smem
.visible .entry <name>( .param .u64 param_<arg>, ... )   // arg_order from meta.json
{
    .reg declarations   // %p preds, %f f32, %r b32, %rd b64 - just count what I handed out
    prologue            // ld.param, cvta.to.global, %tid / %ctaid / %ntid, bounds guard
    body                // one translation per IR statement, in order
    ret;
}
```

Registers are "virtual": I hand out `%f1, %f2, ...` freely and declare `%f<count+1>` at the
top. `ptxas` does the real allocation, so no liveness analysis.

## Anchor problem: p00_scalar_add

`torch.randn(128) + torch.randn(128)`. meta.json: grid (1,1,1), block (256,1,1),
`arg_order = [x0, x1, add]`, rtol = atol = 0.

IR (`k_add_pointwise.ir.txt`):

```
Tile[a0] (N=64)
    in0__u0 = load x0[((a0 * 2) + 0)]
    in1__u0 = load x1[((a0 * 2) + 0)]
    f32 v0__u0 = add(in0__u0, in1__u0)
    in0__u1 = load x0[((a0 * 2) + 1)]
    in1__u1 = load x1[((a0 * 2) + 1)]
    f32 v0__u1 = add(in0__u1, in1__u1)
    add[((a0 * 2) + 0)] = (v0__u0, v0__u1)
```

After `Graph.from_dict`: `graph.nodes['add'].op` is a `KernelOp`; `op.body[0]` is a `Tile`
with `axes[0].name == 'a0'`, `axes[0].extent == Dim(64)`, and a body of
`Load, Load, Assign, Load, Load, Assign, Write`.

Things to notice:

- **N=64 cells but 256 threads.** 128 elements, unrolled by 2 (`__u0`, `__u1`), so only 64
  threads have work. The guard `if (gid < 64)` is *not* in the IR - I must derive it from the
  Tile's extent. Without it threads 64..255 read/write out of bounds.
- **Index expressions are trees**: `BinaryExpr('+', BinaryExpr('*', Var('a0'), Literal(2)), Literal(0))`.
  Compile recursively: `Var` -> the register holding it, `Literal` -> an immediate,
  `BinaryExpr` -> `mul.lo.s32` / `add.s32` into a fresh `%r`.
- **The Write has two values** -> one vector store `st.global.v2.f32 [addr], {%f3, %f6};`
  (address must be 8-byte aligned, which `a0*2` guarantees).

### Node-by-node mapping (what I expect the compiler to emit for p00)

| IR | PTX | why |
|---|---|---|
| signature (from meta `arg_order`) | `.param .u64 param_x0, param_x1, param_add` | buffers are 64-bit pointers in the `.param` space |
| (prologue) | `ld.param.u64 %rd1, [param_x0];` then `cvta.to.global.u64 %rd4, %rd1;` | params are generic addresses; convert to the global space so `ld.global` / `st.global` can use them |
| `Tile[a0] (N=64)` | `mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x; mad.lo.s32 %r4, %r1, %r2, %r3;` | `a0 = blockIdx.x * blockDim.x + threadIdx.x` (the `_gid` in ref.cu); `%r4` becomes the register for `Var('a0')` |
| (guard from N) | `setp.ge.s32 %p1, %r4, 64; @%p1 bra DONE;` | threads past N skip the body |
| index `(a0*2)+0` | `mul.lo.s32 %r5, %r4, 2; add.s32 %r6, %r5, 0;` | element index (could fold `+0`, but not required) |
| `Load in0__u0 = x0[idx]` | `mul.wide.s32 %rd7, %r6, 4; add.s64 %rd8, %rd4, %rd7; ld.global.f32 %f1, [%rd8];` | byte offset = idx * sizeof(f32), widened to 64 bit, added to the base pointer |
| `Assign v0__u0 = add(a, b)` f32 | `add.f32 %f3, %f1, %f2;` | elementwise op on registers; map remembers `v0__u0 -> %f3` |
| `Write add[idx] = (v0__u0, v0__u1)` | `st.global.v2.f32 [%rdK], {%f3, %f6};` | 2 values -> `.v2`, 4 -> `.v4`, 1 -> scalar `st.global.f32` |
| end of body | `DONE: ret;` | |

The central data structure is a dict **IR name -> PTX register** (`'a0' -> '%r4'`,
`'in0__u0' -> '%f1'`, `'x0' -> '%rd4'`). Each statement reads its args from the dict and
writes its result into a fresh register.

## The annotated example: r01_array_sum (`examples/k_sum_1_reduce.ptx`)

`sum` of 1024 floats, grid 1, block 128, `arg_order = [x, sum_1]`.

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

| IR | PTX in the example | notes |
|---|---|---|
| `Smem float acc0_smem[4]` | `.shared .align 4 .b8 acc0_smem[16];` (module scope) | 4 floats = 16 bytes; static smem needs no launch smem_bytes |
| `Tile[a0_co] (N=128)` | `mov.u32 %r2, %tid.x;` | grid is 1 block so tid is enough; in general use ctaid*ntid+tid |
| `for a0 in a0_co..1024:128` (`StridedLoop`) | init `%r4 = a0_co`; `LOOP: setp.ge.s32 %p1, %r4, 1024; @%p1 bra LOOP_END; ... add.s32 %r4, %r4, 128; bra LOOP; LOOP_END:` | start/end/step come from the node; each thread covers 8 elements |
| `Accum acc0 <- add(acc0, in0)` | `mov.f32 %f1, 0f00000000;` **before** the loop, `add.f32 %f1, %f1, %f2;` inside | the accumulator starts at the op's identity (0 for add, -inf for max) and lives in one register across iterations |
| `Load in0 = x[a0]` | `mul.wide.s32` + `add.s64` + `ld.global.f32` | same as p00 |
| `WarpShuffle(acc0, length=32)` | 5 x (`mov.b32` f32->b32, `shfl.sync.bfly.b32 %r6, %r5, k, 0x1f, 0xffffffff`, `mov.b32` back, `add.f32`) for k = 16,8,4,2,1 | XOR butterfly: after log2(32) steps every lane holds the warp sum. shfl works on `.b32`, hence the `mov.b32` round trips |
| `lane`, `warp` | `and.b32 %r7, %r2, 31;` `shr.u32 %r8, %r2, 5;` | implicit names the IR uses but never defines - I must provide them |
| `Cond (lane == 0)` + `acc0_smem[warp] = acc0` | `setp.ne.s32 %p2, %r7, 0; @%p2 bra AFTER;` then `mov.u32 %r9, acc0_smem; shl.b32 ...; st.shared.f32 [%r11], %f1;` | Cond = branch around the body on the negated predicate. Shared addresses are 32-bit (`%r`), not 64-bit |
| `Sync` | `bar.sync 0;` | |
| `TreeHalve(acc0, length=4, tid=warp)` | thread 0 loads all 4 slots and adds them | the example cheats a bit (only thread 0 reduces); ref.cu does a real halving loop with `bar.sync` each step and then broadcasts `acc0 = acc0_smem[0]` to all threads. A general compiler should follow ref.cu |
| `Cond (a0_co == 0)` + `sum_1[] = acc0` | `setp.ne.s32 %p3, %r2, 0; @%p3 bra DONE; st.global.f32 [%rd4], %f10;` | empty index `[]` = offset 0 |

## Address spaces / special registers cheat sheet

- `.param` - kernel arguments, read with `ld.param.u64`.
- `.global` - device memory; pointers go through `cvta.to.global.u64` first, then
  `ld.global` / `st.global` with 64-bit addresses (`%rd`).
- `.shared` - per-block scratch declared with `.shared`; addresses are 32-bit, taken with
  `mov.u32 %r, name`.
- `%tid.x` threadIdx.x, `%ntid.x` blockDim.x, `%ctaid.x` blockIdx.x.
- f32 literals are written as hex bits: `0f00000000` = 0.0, `0f3F800000` = 1.0.
- No non-ASCII anywhere in the PTX (comments included).

## Plan of attack (one step at a time)

1. Prologue + signature + `Tile` + guard + `Load` + `Assign(add)` + `Write` (scalar and v2/v4)
   -> p00. Then p01, p02 (`ConstantOp` args), p03 (more elementwise ops: relu, mul).
2. `StridedLoop` + `Accum` -> partial sums; then `WarpShuffle`, `Smem`, `Cond`, `Sync`,
   `TreeHalve` -> r01, then r02/r03/r04.

## submit.py result

Sanity-checked the toolchain with the hand-written example:

```
$ .venv/bin/python submit.py r01_array_sum examples/k_sum_1_reduce.ptx
[k_sum_1_reduce] 16 registers
[k_sum_1_reduce] PASS  sum_1: max abs err 3.815e-06  (24.864 us)
```

So GPU, ptxas and submit.py work. `compile.py` is still a stub (it prints node kinds to
stderr and no PTX), so nothing compiled has been tested yet.

## What failed and why

Nothing yet. One thing to remember: the example's TreeHalve is specialized (thread 0 does all
the work, size 1024 baked in). My compiler must generate the general pattern from the node's
fields, not copy that shortcut.
