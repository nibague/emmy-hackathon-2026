# 08 - Next steps: categories 3-5

Where things stand (verified 2026-10-29, see the main README): categories 1 and 2 pass, and so
do two fused problems, **f01_rmsnorm** and **f03_l2norm**. Those two are "a reduction, then a
pointwise pass over the same row", built only from node kinds the compiler already handles.

For the rest, the compiler stops at the first unsupported feature. These error messages are
verified (`.venv/bin/python compile.py <ir.json>`):

| problem | first error |
|---|---|
| f02_row_softmax | `NotImplementedError: Accum with base` |
| m01_matmul_512, m02_matmul_rect | `NotImplementedError: statement Init` |
| a01_attention, a02_attention_128 | `NotImplementedError: statement Loop` |

"First error" is only the first. Each problem probably needs more than that one feature. The
lists below come from reading the `.ir.txt` files, not from running anything, so treat them as
a plan (TO VERIFY as each one is implemented).

## Suggested order

1. **f02_row_softmax** (category 4). Closest to what already works.
2. **m02_matmul_rect**, then **m01_matmul_512** (category 3).
3. **a01_attention**, **a02_attention_128** (category 5). They combine everything.

## f02_row_softmax - "online softmax"

```
for a1 in a1_co..256:128
    f32 one = 1.0                                # a Let: bind a literal to a name
    in0 = load x[a0, a1]
    f32 in0__gn = maximum(acc0, in0)
    ...
    f32 acc1 <- add(in0__acc1_sa, in0__acc1_sb)   # an Accum whose value is not just "acc op x"
    f32 acc0 <- maximum(acc0, in0)
WarpShuffle(acc0, acc1, length=32)               # two coupled accumulators
TreeHalve(acc0, acc1, length=4, tid=warp)
```

Needed:
- **`Accum` with `base`**: understand what `base` means for this accumulator (TO VERIFY in
  emmy's source / `ref.cu`) and emit it.
- **`Let`**: bind `one = 1.0` (an f32 immediate) to a name.
- **Two-state WarpShuffle / TreeHalve**: the code loops over `state`/`state_b` pairs already,
  but it has never been tested with two coupled states whose combine uses `exp`.
- The tolerance must hold with our approximate `exp` (`ex2.approx`).

## m02_matmul_rect - matmul with atomics

```
Tile[_a2_ks, a0_b, a1_b, a0_u, a1_u] (N=131072)  # 5 axes
    Init(f32 acc0__c0_0 = 0.0)
    Smem float _a_smem[16, 8] align=16           # 2-D shared arrays
    for _ks in 0..8:8
        for _fa in ((a0_u * 32) + a1_u)..32:256
            cp.async[16B] _a_smem[...] <- x0[...]  # async copy global -> shared
        cp.async.commit_group
        cp.async.wait_group(0)
        Sync
        for _ki in 0..8                           # a plain Loop
            ...
    matmul[...] += acc0__c0_0  (atomic)
```

meta.json: `zero_outputs: ["matmul"]`, so the output is zeroed before each launch, and many
blocks add into it.

Needed:
- **`Init`**: declare a register with a starting value (like an accumulator, but set explicitly).
- **`Loop`**: a plain counted loop `for _ki in 0..8` (StridedLoop with step 1, possibly unrolled).
- **`CpAsyncCopy` / `CpAsyncCommit` / `CpAsyncWait`**: `cp.async.ca.shared.global [smem], [gmem], 16;`,
  `cp.async.commit_group;`, `cp.async.wait_group N;`. These copy global -> shared memory without
  going through registers.
- **Atomic `Write`**: `red.global.add.f32 [addr], value;`, because many blocks add to the same
  output element.
- **2-D shared arrays**: `flat_index` already handles this for any shape (TO VERIFY with smem).
- **5-axis Tile decode**: the current row-major decode of `gid` may or may not match how Emmy
  splits block axes (`_b`) and thread axes (`_u`). Compare with `ref.cu` (TO VERIFY).

## m01_matmul_512 - pipelined matmul

Everything in m02, plus:
- **Dynamic shared memory**: `meta.json` has `smem_bytes: 73728` (72 KiB). Static `.shared`
  arrays are limited to 48 KiB, so this kernel must declare `.extern .shared .align 16 .b8 pool[];`
  and place `_a_smem` / `_b_smem` at offsets inside it (see problems/PROVENANCE.md).
- **`TernaryExpr`** in indices: `((_ks + 128) < 512) ? (_ks + 128) : 448` -> `setp` + `selp`.
- **A 3-stage pipeline**: `cp.async.wait_group(2)` means "wait until at most 2 copy groups are
  still in flight", so copies for the next tiles overlap with math on the current one.

## a01 / a02 attention - softmax inside a matmul

```
for a1 in 0..64                                   # Loop
    for a2 in 0..64                               # nested Loop: a dot product
        ...
        f32 acc0__ar0 <- add(acc0__ar0, v3__ar0)
    ... online-softmax update of acc1, acc3, acc5 (as in f02) ...
    in7__bc0, in7__bc1 = load v[...]              # vector load (already supported)
```

Needed: `Loop` (nested), `Let`, many accumulators at once, and online softmax from f02. Once
f02 and the matmul `Loop` work, these should mostly be a combination of existing pieces
(TO VERIFY).

## Smaller gaps that the hidden set might hit

- Non-f32 data (`f16`, `i32`): `Load`/`Write`/`Assign` reject anything but f32 today.
- Elementwise ops not in the table: `tanh`, `gelu`, `erf`, `pow`, `where`.
- A Tile needing a guard *and* containing barriers: currently a compile error. A real fix would
  keep all threads alive and only predicate the loads/stores.
- `CastExpr`, `TernaryExpr` and `FuncCallExpr` in expressions.
