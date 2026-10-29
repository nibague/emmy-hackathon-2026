# 03 - The Kernel IR

The Kernel IR is the input to `compile.py`. Each problem has it twice:

- `*.ir.json` - the real input. Its `KernelOp.body` is a long Python-like string.
- `*.ir.txt` - the same body pretty-printed for humans. All snippets below come from these files.

`compile.py` never parses the body string itself. Emmy's library turns it into Python objects:

```python
import json
from emmy.compiler.graph import Graph
g = Graph.from_dict(json.load(open("problems/p00_scalar_add/k_add_pointwise.ir.json")))
kernel = g.nodes["add"].op          # a KernelOp
tile = kernel.body[0]               # a Tile object
tile.axes[0].name                   # 'a0'
[type(s).__name__ for s in tile.body]
# ['Load', 'Load', 'Assign', 'Load', 'Load', 'Assign', 'Write']
```

The IR uses **SSA names**: each computed value gets a name (`in0__u0`, `v0__u0`), and later
statements refer to it by name. The compiler keeps a dictionary from each name to the PTX
register holding it.

## Graph level (the top of the json)

| node | meaning | example |
|---|---|---|
| `InputOp` | an input tensor | `x` with shape `[1024]`, f32 |
| `ConstantOp` | a scalar constant from the PyTorch code. It arrives as a **1-element buffer** | `mul_c1` with value `2.0` in p02 |
| `KernelOp` | the kernel: `name` (the PTX entry name) and `body` (the statement tree) | `k_add_pointwise` |

The graph also gives every buffer's **shape**, which is needed to turn 2-D indices into
addresses. The kernel's argument order comes from `meta.json`'s `arg_order`, not from the IR.

## Statements handled by this compiler

### `Tile` - map work onto threads

```
Tile[a0] (N=64)                                  # p00: 64 cells, one per thread
Tile[a0, a1_co] (N=8192)                         # r04: 64 rows x 128 threads
```

The `Tile` defines **axis variables** (`a0`, `a1_co`) that each thread gets from its thread ID,
and N = the number of cells. Everything else is inside its body. For two axes the thread ID is
split row-major: `a0 = gid / 128`, `a1_co = gid % 128`.

### `Load` - read from a buffer

```
in0__u0 = load x0[((a0 * 2) + 0)]                # one value
in1__u0 = load mul_c1[0]                         # a constant (1-element buffer)
in0 = load x[a0, a1]                             # 2-D index (r04)
```

Several names on the left (`a, b = load v[i]`) means a vector load of consecutive values.

### `Assign` - one elementwise operation

```
f32 v0__u0 = add(in0__u0, in1__u0)
f32 v0__u0 = relu(in0__u0)
f32 v0 = divide(acc0, in1)
```

### `Write` - store to a buffer

```
add[((a0 * 2) + 0)] = (v0__u0, v0__u1)           # 2 consecutive values: a vector store
sum_1[] = acc0                                   # empty index = element 0
acc0_smem[warp] = acc0                           # into shared memory
```

### `StridedLoop` - each thread walks an axis with a stride

```
for a0 in a0_co..1024:128                        # r01: from a0_co to 1024, step 128
    in0 = load x[a0]
    f32 acc0 <- add(acc0, in0)
```

With 128 threads and 1024 elements, thread 5 handles elements 5, 133, 261, ... (8 of them).

### `Accum` - running accumulator

```
f32 acc0 <- add(acc0, in0)                       # note the arrow "<-", not "="
```

`acc0` starts at the operation's identity (0 for add, -1e30 for max) *before* the loop and
is updated on every iteration.

### `WarpShuffle` - combine values across the 32 threads of a warp

```
WarpShuffle(acc0, length=32)
```

After it runs, every thread in the warp holds the warp's total. It uses register-to-register
exchanges (`shfl.sync`), not memory. The node carries its combine step as an `Assign`
(`acc0 = add(acc0, acc0__o)`), so the same node works for sum, max, min.

### `Smem` - declare a shared-memory array

```
Smem float acc0_smem[4]                          # one float per warp (128 threads = 4 warps)
```

### `Cond` - run a body only if a condition holds

```
if ((lane == 0)):
    acc0_smem[warp] = acc0
```

### `Sync` - barrier: every thread in the block waits here

```
Sync
```

### `TreeHalve` - combine per-warp partials through shared memory

```
TreeHalve(acc0, length=4, tid=warp)
```

It folds the 4 slots of `acc0_smem` pairwise (4 -> 2 -> 1) with a barrier after each step.
Then every thread reads the final value back into `acc0`.

## Expressions

Indices and conditions are small trees:

| expression | meaning | example from the json |
|---|---|---|
| `Var` | a named value (axis, SSA name, `lane`, `warp`) | `Var(name='a0')` |
| `Literal` | a constant | `Literal(value=2, dtype='int')` |
| `BinaryExpr` | `left op right`: `+ - * / %`, comparisons `== < ...`, `&& ||` | `BinaryExpr(op='*', left=Var('a0'), right=Literal(2))` |

So `((a0 * 2) + 0)` is really:

```
BinaryExpr('+',
    BinaryExpr('*', Var('a0'), Literal(2)),
    Literal(0))
```

## Not handled yet

`Loop`, `Init`, `Let`, `CpAsyncCopy/Commit/Wait`, `Accum` with a `base`, atomic `Write`,
`TernaryExpr`, and non-f32 data. The compiler stops with `NotImplementedError` when it meets
one. See [08-next-steps.md](08-next-steps.md). The full list of IR kinds is in
[`TASK.md`](../../TASK.md).
