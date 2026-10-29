#!/usr/bin/env python3
"""Kernel IR -> PTX compiler. Usage: python compile.py kernel.ir.json > out.ptx

PTX goes to stdout only; anything diagnostic goes to stderr.
"""
import sys
import json
import struct
from pathlib import Path
from emmy.compiler.graph import Graph


# Register classes: prefix -> PTX declaration type.
REG_TYPES = {"p": ".pred", "r": ".b32", "rd": ".b64", "f": ".f32"}

# Integer BinaryExpr ops used in index expressions -> PTX s32 instruction.
INT_BINOPS = {"+": "add.s32", "-": "sub.s32", "*": "mul.lo.s32"}

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


def f32_imm(x):
    """A float as a PTX f32 immediate (hex bit pattern, e.g. 0f3F800000 for 1.0)."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", x))[0]


# Elementwise Assign ops (f32). Each emitter gets (compiler, dst, *srcs) and emits PTX.
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


def static_int(dim):
    """An extent from the IR (Dim or int) as a Python int."""
    return int(dim.as_static()) if hasattr(dim, "as_static") else int(dim)


class KernelCompiler:
    def __init__(self, kernel, arg_order):
        self.kernel = kernel
        self.arg_order = arg_order
        self.counts = {k: 0 for k in REG_TYPES}
        self.regs = {}       # IR name (axis, SSA value, buffer) -> PTX register
        self.lines = []      # body instructions
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
            left = self.compile_index(expr.left)
            right = self.compile_index(expr.right)
            if not left.startswith("%") and not right.startswith("%"):
                # Both constant: fold at compile time.
                return str(eval(f"{left} {expr.op} {right}"))
            if not left.startswith("%"):
                # PTX wants the register first; all foldable ops here but '-' commute.
                if expr.op == "-":
                    tmp = self.new_reg("r")
                    self.emit(f"mov.s32 {tmp}, {left};")
                    left = tmp
                else:
                    left, right = right, left
            dst = self.new_reg("r")
            self.emit(f"{INT_BINOPS[expr.op]} {dst}, {left}, {right};")
            return dst
        raise NotImplementedError(f"expression {kind}")

    def element_address(self, buf, index, nbytes):
        """64-bit global address of buf[index] (index is a 1-tuple, or empty for buf[0])."""
        if len(index) > 1:
            raise NotImplementedError("multi-dimensional index")
        base = self.regs[buf]
        if not index:
            return base
        idx = self.compile_index(index[0])
        if not idx.startswith("%"):
            tmp = self.new_reg("r")
            self.emit(f"mov.s32 {tmp}, {idx};")
            idx = tmp
        off = self.new_reg("rd")
        addr = self.new_reg("rd")
        self.emit(f"mul.wide.s32 {off}, {idx}, {nbytes};")
        self.emit(f"add.s64 {addr}, {base}, {off};")
        return addr

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
        if len(tile.axes) != 1:
            raise NotImplementedError("Tile with more than one axis")
        axis = tile.axes[0]
        n = static_int(axis.extent)
        # axis = blockIdx.x * blockDim.x + threadIdx.x
        ctaid, ntid, tid, gid = (self.new_reg("r") for _ in range(4))
        self.emit(f"mov.u32 {ctaid}, %ctaid.x;")
        self.emit(f"mov.u32 {ntid}, %ntid.x;")
        self.emit(f"mov.u32 {tid}, %tid.x;")
        self.emit(f"mad.lo.s32 {gid}, {ctaid}, {ntid}, {tid};")
        self.regs[axis.name] = gid
        # Guard: the launch may have more threads than the Tile has cells.
        p = self.new_reg("p")
        end = self.new_label("TILE_END")
        self.emit(f"setp.ge.s32 {p}, {gid}, {n};")
        self.emit(f"@{p} bra {end};")
        self.compile_block(tile.body)
        self.emit_label(end)

    def stmt_Load(self, load):
        dt = load.dtype
        if dt.name != "f32":
            raise NotImplementedError(f"Load dtype {dt.name}")
        addr = self.element_address(load.input, load.index, dt.nbytes)
        dsts = [self.new_reg("f") for _ in load.names]
        for name, reg in zip(load.names, dsts):
            self.regs[name] = reg
        if len(dsts) == 1:
            self.emit(f"ld.global.f32 {dsts[0]}, [{addr}];")
        else:
            self.emit(f"ld.global.v{len(dsts)}.f32 {{{', '.join(dsts)}}}, [{addr}];")

    def stmt_Assign(self, assign):
        if assign.dtype.name != "f32":
            raise NotImplementedError(f"Assign dtype {assign.dtype.name}")
        op = assign.op.name
        if op not in F32_OPS:
            raise NotImplementedError(f"elementwise op {op!r}")
        srcs = [self.regs[a] for a in assign.args]
        dst = self.new_reg("f")
        F32_OPS[op](self, dst, *srcs)
        self.regs[assign.name] = dst

    def stmt_Write(self, write):
        dt = write.value_dtype
        if dt.name != "f32":
            raise NotImplementedError(f"Write dtype {dt.name}")
        if write.atomic:
            raise NotImplementedError("atomic Write")
        addr = self.element_address(write.output, write.index, dt.nbytes)
        srcs = [self.regs[v] for v in write.values]
        if len(srcs) == 1:
            self.emit(f"st.global.f32 [{addr}], {srcs[0]};")
        else:
            self.emit(f"st.global.v{len(srcs)}.f32 [{addr}], {{{', '.join(srcs)}}};")

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


def load_arg_order(ir_path, node_name, kernel_name, graph):
    """arg_order from meta.json beside the IR; fall back to graph inputs + outputs."""
    meta_path = Path(ir_path).parent / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        for k in meta.get("kernels", []):
            if k.get("entry") == kernel_name or k.get("node") == node_name:
                return list(k["arg_order"])
    print("warning: no meta.json arg_order, using node inputs + outputs", file=sys.stderr)
    node = graph.nodes[node_name]
    return list(node.inputs) + [o.name for o in node.outputs]


def main():
    if len(sys.argv) != 2:
        print("usage: python compile.py kernel.ir.json > out.ptx", file=sys.stderr)
        sys.exit(1)
    ir_path = sys.argv[1]
    with open(ir_path) as f:
        graph = Graph.from_dict(json.load(f))   # IR file -> Emmy's Python objects

    node_name, kernel = find_kernel(graph)
    arg_order = load_arg_order(ir_path, node_name, kernel.name, graph)
    sys.stdout.write(KernelCompiler(kernel, arg_order).compile())


if __name__ == "__main__":
    main()
