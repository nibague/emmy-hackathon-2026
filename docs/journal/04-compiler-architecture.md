# 04 - Compiler architecture

The whole compiler is one file, [`compile.py`](../../compile.py), about 550 lines. It does
four things: **load** the IR, **walk** its statement tree, hand out **registers**, and
**emit** PTX text.

## The big picture

```
 python compile.py kernel.ir.json
          |
          v
 main()                                         compile.py:533
   |- Graph.from_dict(json)        IR file -> Emmy Python objects
   |- find_kernel(graph)           the one KernelOp
   |- load_meta(...)               meta.json: arg_order, grid, block
   |- buffer_shapes(graph)         buffer name -> shape
   '- KernelCompiler(...).compile()
          |
          v
 compile()                                      compile.py:475
   |- prologue: ld.param + cvta for every argument
   |- compile_block(kernel.body)   <-- the tree walk
   |      |
   |      '- compile_stmt(s) --> stmt_Tile / stmt_Load / stmt_Assign / ...
   |              |                     |
   |              |                     |- compile_index(expr)  integer math -> %r
   |              |                     |- compile_pred(expr)   conditions   -> %p
   |              |                     |- element_address(..)  buffer[i]    -> address
   |              |                     '- emit("add.f32 ...")  append a PTX line
   |              '- nested bodies (Tile, StridedLoop, Cond) call compile_block again
   |
   '- assemble: header + .shared decls + entry signature + .reg decls + lines
          |
          v
   PTX text on stdout
```

## Key idea 1: dispatch by node kind

Every statement class has one method named after it:

```python
def compile_stmt(self, stmt):
    kind = type(stmt).__name__                # "Load", "Tile", ...
    handler = getattr(self, f"stmt_{kind}", None)
    if handler is None:
        raise NotImplementedError(f"statement {kind}")
    handler(stmt)
```

Adding a new node kind means writing one new `stmt_<Kind>` method. An unknown node stops the
compiler with a clear error on stderr, so it never prints half-correct PTX.

## Key idea 2: the name -> register map

`self.regs` maps every IR name to the PTX register holding it:

```
'x0'      -> '%rd2'     buffer base address (global, 64-bit)
'a0'      -> '%r4'      axis variable (thread's index)
'lane'    -> '%r5'
'in0__u0' -> '%f1'      a loaded value
'v0__u0'  -> '%f3'      a computed value
```

Statements look up their inputs here and record their outputs here.

## Key idea 3: "register allocation" is just counting

PTX has unlimited virtual registers, and `ptxas` does the real allocation (see 01). So:

```python
def new_reg(self, kind):            # kind is "r", "rd", "f" or "p"
    self.counts[kind] += 1
    return f"%{kind}{self.counts[kind]}"
```

At the end, the counts become declarations: `.reg .f32 %f<21>;` declares `%f0..%f20`.

| prefix | PTX type | used for |
|---|---|---|
| `%r` | `.b32` | 32-bit integers: thread IDs, indices, shared addresses |
| `%rd` | `.b64` | 64-bit: global addresses |
| `%f` | `.f32` | float values |
| `%p` | `.pred` | true/false predicates for branches |

**One exception** (`bind_value`): if a name is *defined again*, it reuses its register. That
happens for an accumulator (`acc0 <- add(acc0, in0)`) or a shuffle combine. A loop body is
emitted once but runs many times, so the running sum must live in the same register on every
iteration.

## Function by function

### Module-level helpers

| function / table | line | what it does |
|---|---|---|
| `REG_TYPES` | 16 | register prefix -> PTX type |
| `INT_BINOPS` | 20 | integer index ops (`+ - * / % ...`) -> PTX instruction, plus a Python function to fold constants |
| `CMP_OPS`, `LOGIC_OPS` | 36 | `==` -> `setp.eq`, `&&` -> `and.pred`, ... |
| `COLLECTIVE` | 40 | statements every thread must reach (`Sync`, `TreeHalve`, `WarpShuffle`) |
| `f32_imm(x)` | 46 | float -> PTX hex literal: `1.0` -> `0f3F800000` |
| `_exp`, `_log`, `_sigmoid`, `_silu` | 53-80 | multi-instruction elementwise ops |
| `_simple(instr)` | 82 | builds an emitter for a one-instruction op |
| `F32_OPS` | 87 | op name (`"add"`, `"relu"`, ...) -> emitter function |
| `static_int`, `is_imm`, `walk` | 114-128 | small utilities: Dim -> int, "is this an immediate?", iterate all nested statements |

### `KernelCompiler` - the per-kernel state

| method | line | what it does |
|---|---|---|
| `new_reg`, `new_label` | 145, 149 | fresh `%f7`, fresh `LOOP_3` |
| `emit`, `emit_label` | 153, 156 | append a PTX line |
| `to_reg` | 159 | put an immediate into a register (some instructions need a register) |
| `barrier` | 167 | `bar.sync id` or `bar.sync id, count` |
| `bind_value` | 173 | register for a name; reuses it if the name already has one |
| `compile_index` | 182 | `Var`/`Literal`/`BinaryExpr` -> register or immediate. Folds `2*3` to `6` |
| `compile_pred` | 208 | comparison/logic -> predicate register via `setp` |
| `flat_index` | 225 | `x[a0, a1]` with shape `[64, 512]` -> `a0*512 + a1` |
| `element_address` | 237 | `buf[index]` -> address. Global: 64-bit `base + i*4`. Shared: 32-bit |
| `compile_stmt`, `compile_block` | 264, 271 | the dispatcher and "compile a list of statements" |
| `stmt_Tile` | 275 | gid, lane, warp, axis decode, guard (only if launch > N) |
| `init_accumulators` | 315 | sets accumulators to identity before a loop |
| `stmt_StridedLoop` | 331 | loop label, bound check, body, step, jump back |
| `stmt_Accum` | 354 | `acc = op(acc, value)` in place |
| `stmt_Cond` | 363 | `setp` + branch around the body (+ else) |
| `stmt_Smem` | 377 | adds a `.shared` declaration at module scope |
| `stmt_Sync` | 385 | `bar.sync` |
| `stmt_WarpShuffle` | 391 | `shfl.sync.bfly` ladder + the node's combine Assigns |
| `stmt_TreeHalve` | 405 | unrolled shared-memory tree with barriers, then broadcast |
| `stmt_Load` | 440 | `ld.global`/`ld.shared`, scalar or `.v2`/`.v4` |
| `stmt_Assign` | 451 | look up the op in `F32_OPS` and emit it |
| `stmt_Write` | 461 | `st.global`/`st.shared`, scalar or `.v2`/`.v4` |
| `compile` | 475 | prologue, walk, then assemble the final text |

### Top-level functions

| function | line | what it does |
|---|---|---|
| `find_kernel` | 507 | the graph's single `KernelOp` |
| `buffer_shapes` | 516 | every buffer's shape from the graph |
| `load_meta` | 522 | this kernel's entry in `meta.json` (next to the IR file) |
| `main` | 533 | ties it together. Falls back to inputs+outputs and a guard if there is no meta.json |

## Rules the code follows

- **stdout is PTX only.** Warnings and errors go to stderr. Anything else on stdout breaks
  `python compile.py k.ir.json > out.ptx`.
- **Nothing is hardcoded per problem.** Sizes, names and argument order come from the IR and
  `meta.json`. The evaluation uses a hidden set of different kernels.
- **Fail loudly.** Unsupported features raise `NotImplementedError` instead of guessing.
- **Correct before fast.** Index math is recomputed for every load. `ptxas` removes the
  duplicates.
