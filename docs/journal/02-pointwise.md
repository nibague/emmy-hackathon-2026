# 02 - All pointwise problems (p00-p03) + an elementwise op table

New node coverage: `Assign` with any common f32 elementwise op. Everything else (Tile, Load,
Write, index expressions) was already in place from entry 01.

## The IR

p02_scale_shift (`torch.randn(4096) * 2.0 + 1.0`), N=1024, unrolled x4:

```
Tile[a0] (N=1024)
    in1__u0 = load mul_c1[0]
    in2__u0 = load x[((a0 * 4) + 0)]
    f32 v0__u0 = multiply(in1__u0, in2__u0)
    in0__u0 = load add_c1[0]
    f32 v1__u0 = add(in0__u0, v0__u0)
    ... (u1, u2, u3)
    add[((a0 * 4) + 0)] = (v1__u0, v1__u1, v1__u2, v1__u3)
```

`arg_order = [mul_c1, add_c1, x, add]`. The constants `2.0` and `1.0` are `ConstantOp`
nodes, so they arrive as 1-element buffers and are read with an ordinary `Load` at
index `[0]`. No special case was needed: the `Literal` index folds to the immediate `0` and
`element_address` turns it into `mov.s32 %r, 0` + `mul.wide.s32` + `add.s64`.

p03_relu_mul (`torch.relu(x) * y`), N=512, unrolled x4:

```
in0__u0 = load x[((a0 * 4) + 0)]
f32 v0__u0 = relu(in0__u0)
in1__u0 = load y[((a0 * 4) + 0)]
f32 v1__u0 = multiply(in1__u0, v0__u0)
...
mul[((a0 * 4) + 0)] = (v1__u0, v1__u1, v1__u2, v1__u3)   -> st.global.v4.f32
```

## The PTX it became

```
ld.global.f32 %f1, [%rd..];           // mul_c1[0]
ld.global.f32 %f2, [%rd..];           // x[a0*4+0]
mul.f32 %f3, %f1, %f2;                // multiply
ld.global.f32 %f4, [%rd..];           // add_c1[0]
add.f32 %f5, %f4, %f3;                // add
...
st.global.v4.f32 [%rd..], {%f5, %f10, %f15, %f20};

max.f32 %f2, %f1, 0f00000000;         // relu(x) = max(x, 0.0)
mul.f32 %f4, %f3, %f2;
```

The constant is reloaded 4 times (once per unrolled copy), just as the IR says. ptxas can
CSE that; it does not affect correctness.

## The op table

`F32_OPS` maps the emmy op name (`assign.op.name`) to an emitter `(compiler, dst, *srcs)`.
Most are one instruction (`_simple("mul.f32")`). Composite ones use temporaries:

| op | PTX | precision note |
|---|---|---|
| add, subtract/sub, multiply, maximum, minimum, negative, abs, copy, relu, square | `add/sub/mul/max/min/neg/abs/mov.f32` | exact (IEEE round-to-nearest) |
| divide/true_divide, sqrt, reciprocal | `div.rn` / `sqrt.rn` / `rcp.rn` | correctly rounded |
| rsqrt | `rsqrt.approx.f32` | ~1e-7 relative |
| exp | `mul x, log2(e)` then `ex2.approx.f32` | ~2e-7 relative |
| log | `lg2.approx.f32` then `mul ln(2)` | ~1e-7 *absolute* (relative is large near x = 1, where log(x) is about 0) |
| sigmoid | `neg`, exp, `add 1`, `rcp.rn` | ~2e-7 |
| silu | `x * sigmoid(x)` | ~2e-7 |

f32 immediates are written as their hex bit pattern (`f32_imm(1.0)` = `0f3F800000`), so
there is no decimal rounding ambiguity.

Left out on purpose until a problem needs them: tanh, gelu, erf, pow, where (needs a
predicate input), integer and bitwise ops.

## submit.py result

```
p00_scalar_add   PASS  add: max abs err 0.000e+00
p01_vector_add   PASS  add: max abs err 0.000e+00
p02_scale_shift  PASS  add: max abs err 0.000e+00   (22 registers)
p03_relu_mul     PASS  mul: max abs err 0.000e+00   (26 registers)
```

## Testing ops no open problem uses

The runner accepts explicit `data` for each arg, so a scratch harness took p03's IR,
swapped `relu` (unary) or `multiply` (binary) for each op in the table, compiled it with
compile.py, ran it with fixed numpy inputs, and compared with a float64 numpy reference
(allclose rtol=atol=1e-5). All 19 ops pass. Max relative errors are 0-2e-7, except log,
whose 5e-4 *relative* error is near x = 1 (absolute error is still ~1e-7).

## What failed and why

- The first `divide` test "failed" with NaN. That was a bug in the test, not the compiler:
  it computed `y / relu(x)`, which divides by zero for half the inputs. Retesting as
  `y / (|x| + 0.1)` gave 5.9e-08, which passes.
- Nothing in the compiler failed. The `Assign` handler changed from "one instruction
  string" to "call an emitter" so that multi-instruction ops fit.
