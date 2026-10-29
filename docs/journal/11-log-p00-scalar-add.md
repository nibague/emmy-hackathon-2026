# 11 - Build log: p00_scalar_add, first working compiler (originally entry 01)

Nodes handled: `KernelOp`, `Tile` (1 axis), `Load`, `Assign` (`add`, f32), `Write`
(scalar / vector, non-atomic), and the expressions `Var`, `Literal`, `BinaryExpr` (+ - *).

## The IR

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

meta.json: grid (1,1,1), block (256,1,1), `arg_order = [x0, x1, add]`, rtol = atol = 0.

## How compile.py is organised

- `find_kernel` - the one graph node whose op is a `KernelOp`.
- `load_arg_order` - reads `meta.json` next to the IR file and picks the kernel entry by
  `entry` name (or `node`). Falls back to node inputs + outputs with a stderr warning.
- `KernelCompiler` - holds the name -> register map (`self.regs`), per-class register
  counters, and the emitted lines. Dispatch is by class name: `stmt_Tile`, `stmt_Load`, ...
  An unknown statement / op / dtype raises `NotImplementedError`, so failures are loud
  (non-zero exit, message on stderr, nothing half-baked on stdout).
- `compile_index` - recursive over `Var` / `Literal` / `BinaryExpr`. Returns either a
  register (`%r5`) or an immediate string (`"2"`). Two literals fold at compile time;
  a literal on the left of a commutative op is swapped to the right because PTX wants the
  first source to be a register.
- Register declarations are emitted last, once the counts are known: `%f<7>` means
  `%f0..%f6` exist (I start at 1, hence count + 1).

## The PTX it produced, and why

```
.version 8.0 / .target sm_89 / .address_size 64
.visible .entry k_add_pointwise(.param .u64 param_x0, .param .u64 param_x1, .param .u64 param_add)
{
	.reg .pred %p<2>;  .reg .b32 %r<15>;  .reg .b64 %rd<17>;  .reg .f32 %f<7>;

	ld.param.u64 %rd1, [param_x0];        // .param space: the kernel argument
	cvta.to.global.u64 %rd2, %rd1;         // generic -> global address
	... same for x1 (%rd4) and add (%rd6)
	mov.u32 %r1, %ctaid.x;                 // Tile: a0 = ctaid * ntid + tid
	mov.u32 %r2, %ntid.x;
	mov.u32 %r3, %tid.x;
	mad.lo.s32 %r4, %r1, %r2, %r3;
	setp.ge.s32 %p1, %r4, 64;              // guard from N=64 (not in the IR!)
	@%p1 bra TILE_END_1;
	mul.lo.s32 %r5, %r4, 2;                // index (a0*2)+0
	add.s32 %r6, %r5, 0;
	mul.wide.s32 %rd7, %r6, 4;             // * sizeof(f32), widened to 64 bit
	add.s64 %rd8, %rd2, %rd7;
	ld.global.f32 %f1, [%rd8];             // Load in0__u0
	...
	add.f32 %f3, %f1, %f2;                 // Assign v0__u0 = add(...)
	...
	st.global.v2.f32 [%rd16], {%f3, %f6};  // Write with 2 values -> vector store
TILE_END_1:
	ret;
}
```

- Registers: `%r` (b32) for thread ids and indices, `%rd` (b64) for addresses, `%f` for
  values, `%p` for the guard predicate.
- Address spaces: params via `ld.param`, data via `ld.global` / `st.global` after `cvta`.
- `%ctaid.x * %ntid.x + %tid.x` is used even though grid = 1, so multi-block launches work
  unchanged.
- The index expression is recomputed for every Load (no common-subexpression reuse).
  That is deliberate: correctness first, ptxas cleans it up.
- `Load` with several names becomes `ld.global.v2/v4.f32 {..}, [addr]` (not exercised by p00
  yet, same code path as the vector Write).

## submit.py result

```
$ .venv/bin/python submit.py p00_scalar_add --compiler compile.py
[k_add_pointwise] 16 registers
[k_add_pointwise] PASS  add: max abs err 0.000e+00  (19.584 us)

$ .venv/bin/python submit.py p01_vector_add --compiler compile.py
[k_add_pointwise] 16 registers
[k_add_pointwise] PASS  add: max abs err 0.000e+00  (33.632 us)
```

p01 is the same IR with N=512 and passes without changes, a first sign the compiler is
not tied to p00's shape.

Also checked: with `meta.json` removed, the fallback order (x0, x1, add) is used and a
warning goes to stderr; p02 stops with `NotImplementedError: elementwise op 'multiply'`.

## What failed and why

Nothing failed on the first run. Known limits, to address as problems need them:

- `F32_OPS` only has `add` (p02 needs `multiply`, p03 needs `relu`).
- Only 1-axis `Tile`, 1-D indices, f32 data, non-atomic writes.
- Indexes are signed 32-bit (`mul.wide.s32`), so buffers above 2^31 elements would
  overflow. No problem comes close.
