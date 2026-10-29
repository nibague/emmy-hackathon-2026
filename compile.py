#!/usr/bin/env python3
"""Kernel IR -> PTX compiler. Usage: python compile.py kernel.ir.json > out.ptx

PTX goes to stdout only; anything diagnostic goes to stderr.
"""
import sys
import json
import math
import struct
from pathlib import Path
from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var


# Register classes: prefix -> PTX declaration type.
REG_TYPES = {"p": ".pred", "r": ".b32", "rd": ".b64", "f": ".f32"}

# Integer BinaryExpr ops in index expressions: PTX instruction and compile-time fold.
# Indices are never negative, so truncating '/' and '%' agree with floor semantics.
INT_BINOPS = {
    "+": ("add.s32", lambda a, b: a + b),
    "-": ("sub.s32", lambda a, b: a - b),
    "*": ("mul.lo.s32", lambda a, b: a * b),
    "/": ("div.s32", lambda a, b: int(a / b)),
    "//": ("div.s32", lambda a, b: int(a / b)),
    "%": ("rem.s32", lambda a, b: int(math.fmod(a, b))),
    "&": ("and.b32", lambda a, b: a & b),
    "|": ("or.b32", lambda a, b: a | b),
    "^": ("xor.b32", lambda a, b: a ^ b),
    "<<": ("shl.b32", lambda a, b: a << b),
    ">>": ("shr.s32", lambda a, b: a >> b),
}
COMMUTATIVE = {"+", "*", "&", "|", "^"}

# Comparison BinaryExpr ops -> setp condition; logical ops -> predicate instruction.
CMP_OPS = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
LOGIC_OPS = {"&&": "and.pred", "||": "or.pred"}

# Statements that need every thread of the CTA (or warp) to arrive.
COLLECTIVE = {"Sync", "TreeHalve", "WarpShuffle"}

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


def f32_imm(x):
    """A float as a PTX f32 immediate (hex bit pattern, e.g. 0f3F800000 for 1.0)."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", x))[0]


# Elementwise Assign ops (f32). Each emitter gets (compiler, dst, *srcs) and emits PTX.
# dst may be one of the srcs (in-place update), so emitters write dst last.
def _exp(c, d, x):
    # e^x = 2^(x * log2 e)
    t = c.new_reg("f")
    c.emit(f"mul.f32 {t}, {x}, {f32_imm(LOG2E)};")
    c.emit(f"ex2.approx.f32 {d}, {t};")


def _log(c, d, x):
    # ln x = log2(x) * ln 2
    t = c.new_reg("f")
    c.emit(f"lg2.approx.f32 {t}, {x};")
    c.emit(f"mul.f32 {d}, {t}, {f32_imm(LN2)};")


def _sigmoid(c, d, x):
    # 1 / (1 + e^-x)
    n, e, s = c.new_reg("f"), c.new_reg("f"), c.new_reg("f")
    c.emit(f"neg.f32 {n}, {x};")
    _exp(c, e, n)
    c.emit(f"add.f32 {s}, {e}, {f32_imm(1.0)};")
    c.emit(f"rcp.rn.f32 {d}, {s};")


def _silu(c, d, x):
    s = c.new_reg("f")
    _sigmoid(c, s, x)
    c.emit(f"mul.f32 {d}, {x}, {s};")


def _simple(instr):
    """An op that is one PTX instruction taking every source in order."""
    return lambda c, d, *xs: c.emit(f"{instr} {d}, {', '.join(xs)};")


F32_OPS = {
    "add": _simple("add.f32"),
    "subtract": _simple("sub.f32"),
    "sub": _simple("sub.f32"),
    "multiply": _simple("mul.f32"),
    "divide": _simple("div.rn.f32"),
    "true_divide": _simple("div.rn.f32"),
    "maximum": _simple("max.f32"),
    "minimum": _simple("min.f32"),
    "negative": _simple("neg.f32"),
    "abs": _simple("abs.f32"),
    "copy": _simple("mov.f32"),
    "sqrt": _simple("sqrt.rn.f32"),
    "rsqrt": _simple("rsqrt.approx.f32"),
    "reciprocal": _simple("rcp.rn.f32"),
    "relu": lambda c, d, x: c.emit(f"max.f32 {d}, {x}, {f32_imm(0.0)};"),
    "square": lambda c, d, x: c.emit(f"mul.f32 {d}, {x}, {x};"),
    "exp": _exp,
    "log": _log,
    "sigmoid": _sigmoid,
    "silu": _silu,
}

# Smem dtype spellings (C names) -> element size in bytes; only f32 is supported.
SMEM_F32 = {"float", "f32"}


def static_int(dim):
    """An extent from the IR (Dim or int) as a Python int."""
    return int(dim.as_static()) if hasattr(dim, "as_static") else int(dim)


def is_imm(operand):
    return not operand.startswith("%")


def walk(stmts):
    """Every statement in stmts, recursively through bodies."""
    for s in stmts:
        yield s
        for field in ("body", "else_body"):
            yield from walk(getattr(s, field, None) or ())


class KernelCompiler:
    def __init__(self, kernel, arg_order, shapes, launch_threads):
        self.kernel = kernel
        self.arg_order = arg_order
        self.shapes = shapes                  # global buffer name -> shape (tuple of ints)
        self.launch_threads = launch_threads  # grid * block from meta.json, or None
        self.smem = {}                        # shared buffer name -> shape
        self.counts = {k: 0 for k in REG_TYPES}
        self.regs = {}       # IR name (axis, SSA value, buffer) -> PTX register
        self.lines = []      # body instructions
        self.decls = []      # module-scope declarations (.shared arrays)
        self.labels = 0

    # ---- helpers ---------------------------------------------------------
    def new_reg(self, kind):
        self.counts[kind] += 1
        return f"%{kind}{self.counts[kind]}"

    def new_label(self, stem):
        self.labels += 1
        return f"{stem}_{self.labels}"

    def emit(self, line):
        self.lines.append("\t" + line)

    def emit_label(self, label):
        self.lines.append(f"{label}:")

    def to_reg(self, operand):
        """Materialize an immediate into a %r register (registers pass through)."""
        if not is_imm(operand):
            return operand
        reg = self.new_reg("r")
        self.emit(f"mov.s32 {reg}, {operand};")
        return reg

    def barrier(self, barrier_id, count):
        if count is None:
            self.emit(f"bar.sync {barrier_id};")
        else:
            self.emit(f"bar.sync {barrier_id}, {count};")

    def bind_value(self, name, kind="f"):
        """Register for an SSA name being (re)defined. A name defined again (accumulator
        combines, shuffle ladders) reuses its register, so the value lives in one place
        across loop iterations and branches."""
        if name not in self.regs:
            self.regs[name] = self.new_reg(kind)
        return self.regs[name]

    # ---- expressions -----------------------------------------------------
    def compile_index(self, expr):
        """Compile an integer expression; returns a register or an immediate string."""
        kind = type(expr).__name__
        if kind == "Var":
            return self.regs[expr.name]
        if kind == "Literal":
            return str(int(expr.value))
        if kind == "BinaryExpr":
            if expr.op not in INT_BINOPS:
                raise NotImplementedError(f"index op {expr.op!r}")
            instr, fold = INT_BINOPS[expr.op]
            left = self.compile_index(expr.left)
            right = self.compile_index(expr.right)
            if is_imm(left) and is_imm(right):
                return str(fold(int(left), int(right)))
            if is_imm(left):
                # PTX wants the register first.
                if expr.op in COMMUTATIVE:
                    left, right = right, left
                else:
                    left = self.to_reg(left)
            dst = self.new_reg("r")
            self.emit(f"{instr} {dst}, {left}, {right};")
            return dst
        raise NotImplementedError(f"expression {kind}")

    def compile_pred(self, expr):
        """Compile a boolean expression into a predicate register."""
        if type(expr).__name__ != "BinaryExpr":
            raise NotImplementedError(f"predicate {type(expr).__name__}")
        p = self.new_reg("p")
        if expr.op in CMP_OPS:
            left = self.to_reg(self.compile_index(expr.left))
            right = self.compile_index(expr.right)
            self.emit(f"setp.{CMP_OPS[expr.op]}.s32 {p}, {left}, {right};")
        elif expr.op in LOGIC_OPS:
            a = self.compile_pred(expr.left)
            b = self.compile_pred(expr.right)
            self.emit(f"{LOGIC_OPS[expr.op]} {p}, {a}, {b};")
        else:
            raise NotImplementedError(f"predicate op {expr.op!r}")
        return p

    def flat_index(self, buf, index, shape):
        """Row-major flatten of a multi-dimensional index into one expression."""
        if len(index) != len(shape):
            # keepdim-style shapes: an index may skip the size-1 dims
            shape = tuple(d for d in shape if d != 1)
        if len(index) != len(shape):
            raise NotImplementedError(f"index rank {len(index)} for {buf} of shape {shape}")
        flat = index[0]
        for idx, extent in zip(index[1:], shape[1:]):
            flat = BinaryExpr("+", BinaryExpr("*", flat, Literal(extent, "int")), idx)
        return flat

    def element_address(self, buf, index, nbytes):
        """Address of buf[index] -> (state space, address operand).

        Global buffers use 64-bit addresses off the cvta'd param; shared arrays use
        32-bit addresses off the array symbol. An empty index means element 0.
        """
        shared = buf in self.smem
        shape = self.smem[buf] if shared else self.shapes.get(buf, ())
        if shared:
            base = self.new_reg("r")
            self.emit(f"mov.u32 {base}, {buf};")
        else:
            base = self.regs[buf]
        if not index:
            return ("shared" if shared else "global"), base
        idx = self.to_reg(self.compile_index(self.flat_index(buf, index, shape)))
        if shared:
            off, addr = self.new_reg("r"), self.new_reg("r")
            self.emit(f"mul.lo.s32 {off}, {idx}, {nbytes};")
            self.emit(f"add.s32 {addr}, {base}, {off};")
            return "shared", addr
        off, addr = self.new_reg("rd"), self.new_reg("rd")
        self.emit(f"mul.wide.s32 {off}, {idx}, {nbytes};")
        self.emit(f"add.s64 {addr}, {base}, {off};")
        return "global", addr

    # ---- statements ------------------------------------------------------
    def compile_stmt(self, stmt):
        kind = type(stmt).__name__
        handler = getattr(self, f"stmt_{kind}", None)
        if handler is None:
            raise NotImplementedError(f"statement {kind}")
        handler(stmt)

    def compile_block(self, stmts):
        for s in stmts:
            self.compile_stmt(s)

    def stmt_Tile(self, tile):
        sizes = [static_int(a.extent) for a in tile.axes]
        n = math.prod(sizes)
        # gid = blockIdx.x * blockDim.x + threadIdx.x
        ctaid, ntid, tid, gid = (self.new_reg("r") for _ in range(4))
        self.emit(f"mov.u32 {ctaid}, %ctaid.x;")
        self.emit(f"mov.u32 {ntid}, %ntid.x;")
        self.emit(f"mov.u32 {tid}, %tid.x;")
        self.emit(f"mad.lo.s32 {gid}, {ctaid}, {ntid}, {tid};")
        # Implicit names the IR uses without defining: lane and warp within the CTA.
        lane, warp = self.new_reg("r"), self.new_reg("r")
        self.emit(f"and.b32 {lane}, {tid}, 31;")
        self.emit(f"shr.u32 {warp}, {tid}, 5;")
        self.regs["lane"], self.regs["warp"] = lane, warp
        # Decode gid row-major over the axes: last axis varies fastest.
        rest = gid
        for axis, size in reversed(list(zip(tile.axes[1:], sizes[1:]))):
            a, q = self.new_reg("r"), self.new_reg("r")
            self.emit(f"rem.u32 {a}, {rest}, {size};")
            self.emit(f"div.u32 {q}, {rest}, {size};")
            self.regs[axis.name] = a
            rest = q
        self.regs[tile.axes[0].name] = rest

        # Guard: the launch may have more threads than the Tile has cells. Skip it when
        # the launch is exactly N threads - it must never split a CTA around a barrier.
        end = None
        if self.launch_threads is None or self.launch_threads > n:
            if any(type(s).__name__ in COLLECTIVE for s in walk(tile.body)):
                raise NotImplementedError(
                    f"Tile of {n} cells launched with {self.launch_threads} threads "
                    "needs a guard, but its body has barriers/shuffles")
            p = self.new_reg("p")
            end = self.new_label("TILE_END")
            self.emit(f"setp.ge.s32 {p}, {gid}, {n};")
            self.emit(f"@{p} bra {end};")
        self.compile_block(tile.body)
        if end:
            self.emit_label(end)

    def init_accumulators(self, body, reseed):
        """Set each Accum of a loop body to its op's identity before the loop starts.
        Accums of nested loops are left to those loops."""
        stack = list(body)
        while stack:
            s = stack.pop()
            kind = type(s).__name__
            if kind == "Accum" and (reseed or s.name not in self.regs):
                if s.base is not None:
                    raise NotImplementedError("Accum with base")
                acc = self.bind_value(s.name)
                self.emit(f"mov.f32 {acc}, {f32_imm(float(s.op.identity))};")
            elif kind == "Cond":
                stack.extend(s.body)
                stack.extend(s.else_body or ())

    def stmt_StridedLoop(self, loop):
        # for (axis = start; axis < end; axis += step)
        self.init_accumulators(loop.body, loop.seed)
        # end is an expression, or (when None) the axis extent
        end = loop.end if loop.end is not None else loop.axis.extent
        if isinstance(end, int) or hasattr(end, "as_static"):
            end = str(static_int(end))
        else:
            end = self.compile_index(end)
        step = self.compile_index(loop.step)
        var = self.new_reg("r")
        self.emit(f"mov.s32 {var}, {self.compile_index(loop.start)};")
        self.regs[loop.axis.name] = var
        top, done = self.new_label("LOOP"), self.new_label("LOOP_END")
        p = self.new_reg("p")
        self.emit_label(top)
        self.emit(f"setp.ge.s32 {p}, {var}, {end};")
        self.emit(f"@{p} bra {done};")
        self.compile_block(loop.body)
        self.emit(f"add.s32 {var}, {var}, {step};")
        self.emit(f"bra {top};")
        self.emit_label(done)

    def stmt_Accum(self, accum):
        if accum.dtype.name != "f32":
            raise NotImplementedError(f"Accum dtype {accum.dtype.name}")
        op = accum.op.name
        if op not in F32_OPS:
            raise NotImplementedError(f"accumulate op {op!r}")
        acc = self.bind_value(accum.name)
        F32_OPS[op](self, acc, acc, self.regs[accum.value])

    def stmt_Cond(self, cond):
        p = self.compile_pred(cond.cond)
        skip = self.new_label("ELSE" if cond.else_body else "ENDIF")
        self.emit(f"@!{p} bra {skip};")
        self.compile_block(cond.body)
        if cond.else_body:
            end = self.new_label("ENDIF")
            self.emit(f"bra {end};")
            self.emit_label(skip)
            self.compile_block(cond.else_body)
            self.emit_label(end)
        else:
            self.emit_label(skip)

    def stmt_Smem(self, smem):
        if smem.dtype not in SMEM_F32:
            raise NotImplementedError(f"Smem dtype {smem.dtype}")
        shape = tuple(static_int(e) for e in smem.extents)
        self.smem[smem.name] = shape
        align = max(4, smem.align or 0)
        self.decls.append(f".shared .align {align} .b8 {smem.name}[{4 * math.prod(shape)}];")

    def stmt_Sync(self, sync):
        if sync.warp:
            self.emit("bar.warp.sync 0xffffffff;")
        else:
            self.barrier(sync.barrier_id, sync.count)

    def stmt_WarpShuffle(self, ws):
        # XOR butterfly: at each offset every lane combines with lane ^ offset.
        if ws.dtype.name != "f32":
            raise NotImplementedError(f"WarpShuffle dtype {ws.dtype.name}")
        offset = ws.length // 2
        while offset >= 1:
            for state, other in zip(ws.state, ws.state_b):
                bits, got = self.new_reg("r"), self.new_reg("r")
                self.emit(f"mov.b32 {bits}, {self.regs[state]};")
                self.emit(f"shfl.sync.bfly.b32 {got}, {bits}, {offset}, 0x1f, 0xffffffff;")
                self.emit(f"mov.b32 {self.bind_value(other)}, {got};")
            self.compile_block(ws.combine_states)
            offset //= 2

    def stmt_TreeHalve(self, th):
        # for s = length/2 .. 1: if (tid < s) buf[tid] = combine(buf[tid], buf[tid + s]); sync
        # then every thread reads the result from buf[0].
        if th.inner is not None:
            raise NotImplementedError("TreeHalve with inner")
        if th.dtype.name != "f32":
            raise NotImplementedError(f"TreeHalve dtype {th.dtype.name}")
        tid = self.regs[th.tid_var]
        finals = {state: self.regs[state] for state in th.state}
        s = th.length // 2
        while s >= 1:
            skip = self.new_label("TREE_SKIP")
            p = self.new_reg("p")
            self.emit(f"setp.ge.s32 {p}, {tid}, {s};")
            self.emit(f"@{p} bra {skip};")
            mine = {}
            for buf, state, other in zip(th.bufs, th.state, th.state_b):
                _, here = self.element_address(buf, (Var(th.tid_var),), 4)
                a, b = self.new_reg("f"), self.new_reg("f")
                self.emit(f"ld.shared.f32 {a}, [{here}];")
                self.emit(f"ld.shared.f32 {b}, [{here}+{4 * s}];")
                self.regs[state], self.regs[other] = a, b
                mine[state] = here
            self.compile_block(th.combine_states)
            for state, here in mine.items():
                self.emit(f"st.shared.f32 [{here}], {self.regs[state]};")
            self.emit_label(skip)
            self.barrier(th.barrier_id, th.barrier_count)
            s //= 2
        # Broadcast: the state names go back to their own registers, loaded from buf[0].
        for buf, state in zip(th.bufs, th.state):
            self.regs[state] = finals[state]
            _, base = self.element_address(buf, (), 4)
            self.emit(f"ld.shared.f32 {finals[state]}, [{base}];")

    def stmt_Load(self, load):
        dt = load.dtype
        if dt.name != "f32":
            raise NotImplementedError(f"Load dtype {dt.name}")
        space, addr = self.element_address(load.input, load.index, dt.nbytes)
        dsts = [self.bind_value(name) for name in load.names]
        if len(dsts) == 1:
            self.emit(f"ld.{space}.f32 {dsts[0]}, [{addr}];")
        else:
            self.emit(f"ld.{space}.v{len(dsts)}.f32 {{{', '.join(dsts)}}}, [{addr}];")

    def stmt_Assign(self, assign):
        # Combine steps of WarpShuffle/TreeHalve carry dtype=None; they are f32 here.
        if assign.dtype is not None and assign.dtype.name != "f32":
            raise NotImplementedError(f"Assign dtype {assign.dtype.name}")
        op = assign.op.name
        if op not in F32_OPS:
            raise NotImplementedError(f"elementwise op {op!r}")
        srcs = [self.regs[a] for a in assign.args]
        F32_OPS[op](self, self.bind_value(assign.name), *srcs)

    def stmt_Write(self, write):
        dt = write.value_dtype
        if dt.name != "f32":
            raise NotImplementedError(f"Write dtype {dt.name}")
        if write.atomic:
            raise NotImplementedError("atomic Write")
        space, addr = self.element_address(write.output, write.index, dt.nbytes)
        srcs = [self.regs[v] for v in write.values]
        if len(srcs) == 1:
            self.emit(f"st.{space}.f32 [{addr}], {srcs[0]};")
        else:
            self.emit(f"st.{space}.v{len(srcs)}.f32 [{addr}], {{{', '.join(srcs)}}};")

    # ---- whole kernel ----------------------------------------------------
    def compile(self):
        # Prologue: load each pointer param and convert it to a global address.
        for arg in self.arg_order:
            raw = self.new_reg("rd")
            glob = self.new_reg("rd")
            self.emit(f"ld.param.u64 {raw}, [param_{arg}];")
            self.emit(f"cvta.to.global.u64 {glob}, {raw};")
            self.regs[arg] = glob
        self.compile_block(self.kernel.body)
        self.emit("ret;")

        params = ",\n".join(f"\t.param .u64 param_{a}" for a in self.arg_order)
        decls = [f"\t.reg {REG_TYPES[k]} %{k}<{c + 1}>;" for k, c in self.counts.items() if c]
        return "\n".join([
            ".version 8.0",
            ".target sm_89",
            ".address_size 64",
            "",
            *self.decls,
            "",
            f".visible .entry {self.kernel.name}(",
            params,
            ")",
            "{",
            *decls,
            "",
            *self.lines,
            "}",
            "",
        ])


def find_kernel(graph):
    """Return (node_name, KernelOp) for the single kernel in the graph."""
    kernels = [(name, node.op) for name, node in graph.nodes.items()
               if type(getattr(node, "op", None)).__name__ == "KernelOp"]
    if len(kernels) != 1:
        raise ValueError(f"expected one KernelOp, found {len(kernels)}")
    return kernels[0]


def buffer_shapes(graph):
    """Every buffer the graph defines (inputs, constants, kernel outputs) -> static shape."""
    return {out.name: tuple(static_int(d) for d in out.shape)
            for node in graph.nodes.values() for out in node.outputs}


def load_meta(ir_path, node_name, kernel_name):
    """This kernel's entry in meta.json beside the IR, or None."""
    meta_path = Path(ir_path).parent / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        for k in meta.get("kernels", []):
            if k.get("entry") == kernel_name or k.get("node") == node_name:
                return k
    return None


def main():
    if len(sys.argv) != 2:
        print("usage: python compile.py kernel.ir.json > out.ptx", file=sys.stderr)
        sys.exit(1)
    ir_path = sys.argv[1]
    with open(ir_path) as f:
        graph = Graph.from_dict(json.load(f))   # IR file -> Emmy's Python objects

    node_name, kernel = find_kernel(graph)
    meta = load_meta(ir_path, node_name, kernel.name)
    if meta is not None:
        arg_order = list(meta["arg_order"])
        launch_threads = math.prod(meta["grid"]) * math.prod(meta["block"])
    else:
        print("warning: no meta.json entry, using node inputs + outputs and guarding",
              file=sys.stderr)
        node = graph.nodes[node_name]
        arg_order = list(node.inputs) + [o.name for o in node.outputs]
        launch_threads = None
    compiler = KernelCompiler(kernel, arg_order, buffer_shapes(graph), launch_threads)
    sys.stdout.write(compiler.compile())


if __name__ == "__main__":
    main()
