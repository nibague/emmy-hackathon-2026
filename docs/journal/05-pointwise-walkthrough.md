# 05 - Pointwise walkthrough: p02_scale_shift

**Pointwise** (or **elementwise**) means every output element depends only on the input
elements at the same position. Threads never need each other. This is the simplest kind of GPU
kernel.

## The problem

From `problems/p02_scale_shift/meta.json`:

```
code:      torch.randn(4096) * 2.0 + 1.0
grid:      [4, 1, 1]     block: [256, 1, 1]      -> 4 x 256 = 1024 threads
arg_order: [mul_c1, add_c1, x, add]
args:      mul_c1 = [2.0], add_c1 = [1.0]   (constants arrive as 1-element buffers)
           x = 4096 random floats, add = 4096-float output
tolerance: rtol = atol = 1e-6
```

4096 elements over 1024 threads means **each thread handles 4 consecutive elements**.

## The IR (`k_add_pointwise.ir.txt`)

```
Tile[a0] (N=1024)
    in1__u0 = load mul_c1[0]
    in2__u0 = load x[((a0 * 4) + 0)]
    f32 v0__u0 = multiply(in1__u0, in2__u0)
    in0__u0 = load add_c1[0]
    f32 v1__u0 = add(in0__u0, v0__u0)
    ... the same 5 lines for __u1 (+1), __u2 (+2), __u3 (+3)
    add[((a0 * 4) + 0)] = (v1__u0, v1__u1, v1__u2, v1__u3)
```

- `__u0 .. __u3` are the 4 **unrolled** copies: Emmy already wrote the loop out 4 times.
- The final `Write` stores all 4 results at once, as a vector store.

## The generated PTX, annotated

This is the real output of
`.venv/bin/python compile.py problems/p02_scale_shift/k_add_pointwise.ir.json`. Comments were
added here by hand; the real file has none.

### Header and signature

```ptx
.version 8.0              // PTX language version (sm_89 needs 7.8+)
.target sm_89             // minimum GPU architecture
.address_size 64          // pointers are 64-bit

.visible .entry k_add_pointwise(       // entry name = KernelOp.name from the IR
	.param .u64 param_mul_c1,          // one 64-bit pointer per arg,
	.param .u64 param_add_c1,          // in meta.json's arg_order
	.param .u64 param_x,
	.param .u64 param_add
)
{
	.reg .b32 %r<25>;                  // registers, counted after the walk
	.reg .b64 %rd<35>;
	.reg .f32 %f<21>;
```

There is no `.reg .pred` line because this kernel never branches (see "Why no guard?" below).

### Prologue: arguments become global addresses

```ptx
	ld.param.u64 %rd1, [param_mul_c1];     // read argument 1 from parameter space
	cvta.to.global.u64 %rd2, %rd1;         // generic -> global address.  regs['mul_c1'] = %rd2
	ld.param.u64 %rd3, [param_add_c1];
	cvta.to.global.u64 %rd4, %rd3;         // regs['add_c1'] = %rd4
	ld.param.u64 %rd5, [param_x];
	cvta.to.global.u64 %rd6, %rd5;         // regs['x'] = %rd6
	ld.param.u64 %rd7, [param_add];
	cvta.to.global.u64 %rd8, %rd7;         // regs['add'] = %rd8
```

### `Tile[a0] (N=1024)`: who am I?

```ptx
	mov.u32 %r1, %ctaid.x;                 // block index
	mov.u32 %r2, %ntid.x;                  // threads per block (256)
	mov.u32 %r3, %tid.x;                   // thread index in block
	mad.lo.s32 %r4, %r1, %r2, %r3;         // a0 = ctaid*ntid + tid.  regs['a0'] = %r4
	and.b32 %r5, %r3, 31;                  // lane (unused here, always computed)
	shr.u32 %r6, %r3, 5;                   // warp (unused here)
```

**Why no guard?** The launch has 4 x 256 = 1024 threads and the Tile has N = 1024 cells, so
every thread has work. The compiler only emits `setp.ge.s32 %p, a0, N; @%p bra END;` when the
launch is bigger than N. In p00 (256 threads, N = 64) it does, and threads 64..255 jump to the end.

### Copy `__u0`: `load mul_c1[0]`

```ptx
	mov.s32 %r7, 0;                        // index 0 (an immediate, put in a register)
	mul.wide.s32 %rd9, %r7, 4;             // byte offset = index * 4 bytes, widened to 64-bit
	add.s64 %rd10, %rd2, %rd9;             // address = mul_c1 + offset
	ld.global.f32 %f1, [%rd10];            // %f1 = 2.0.  regs['in1__u0'] = %f1
```

Every load follows this recipe: **index -> x4 bytes -> add base -> ld.global**.

### `load x[((a0 * 4) + 0)]`

```ptx
	mul.lo.s32 %r8, %r4, 4;                // a0 * 4         (BinaryExpr '*')
	add.s32 %r9, %r8, 0;                   // ... + 0        (BinaryExpr '+')
	mul.wide.s32 %rd11, %r9, 4;            // * 4 bytes
	add.s64 %rd12, %rd6, %rd11;            // x + offset
	ld.global.f32 %f2, [%rd12];            // regs['in2__u0'] = %f2
```

The index expression tree `((a0 * 4) + 0)` is compiled bottom-up by `compile_index`:
`a0` is `%r4`, `4` is an immediate, the `*` node emits `mul.lo.s32`, and the `+` node emits
`add.s32`. (`mul.lo` keeps the low 32 bits of the product.)

### `multiply`, then `load add_c1[0]`, then `add`

```ptx
	mul.f32 %f3, %f1, %f2;                 // v0__u0 = 2.0 * x[i]
	mov.s32 %r10, 0;
	mul.wide.s32 %rd13, %r10, 4;
	add.s64 %rd14, %rd4, %rd13;
	ld.global.f32 %f4, [%rd14];            // in0__u0 = 1.0
	add.f32 %f5, %f4, %f3;                 // v1__u0 = 1.0 + 2.0*x[i]
```

### Copies `__u1`, `__u2`, `__u3`

The same 15 lines three more times, with `+ 1`, `+ 2`, `+ 3` in the index. They produce
`%f10`, `%f15`, `%f20`. (The constants are reloaded each time because the IR says so; `ptxas`
removes the repeats.)

### The `Write`: one vector store

```ptx
	mul.lo.s32 %r23, %r4, 4;               // a0 * 4
	add.s32 %r24, %r23, 0;
	mul.wide.s32 %rd33, %r24, 4;
	add.s64 %rd34, %rd8, %rd33;            // &add[a0*4]
	st.global.v4.f32 [%rd34], {%f5, %f10, %f15, %f20};   // 4 floats, 16 bytes, one instruction
	ret;
}
```

`.v4` writes 4 consecutive floats at once. It requires the address to be a multiple of 16
bytes, which `a0 * 4` elements x 4 bytes guarantees.

## Result

```
$ .venv/bin/python submit.py p02_scale_shift --compiler compile.py
[k_add_pointwise] 22 registers
[k_add_pointwise] PASS  add: max abs err 0.000e+00
```

Our PTX names 20 float + 24 int + 34 address registers. `ptxas` fit the real program into 22.

## What to take away

1. The thread ID picks the data: `a0 = ctaid*ntid + tid`.
2. Every buffer access is `base + index * 4`.
3. Each IR line becomes 1 to 5 PTX lines. No line depends on any other thread.
4. Unrolling and vector stores are already decided in the IR. The compiler just follows it.
