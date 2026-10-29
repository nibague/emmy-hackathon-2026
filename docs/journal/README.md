# Journal: index and reading order

There are two kinds of entries:

- **Concept guides (01-08)** explain the ideas, written so a beginner can follow them in order.
- **Build log (10-13)** is what was actually done, step by step, while building the
  compiler: the IR, the PTX, the test results and what went wrong. Later entries continue from 14.

Terms are defined the first time they appear, and all of them are collected in the
[glossary](07-glossary.md).

## Suggested reading order

| # | file | read it to learn |
|---|---|---|
| 01 | [What is a GPU compiler?](01-what-is-a-gpu-compiler.md) | the whole pipeline from PyTorch to the GPU, what PTX and SASS are, where this project fits |
| 02 | [GPU execution model](02-gpu-execution-model.md) | threads, warps, blocks, grids; registers, shared and global memory; how `meta.json` sets them |
| 03 | [The Kernel IR](03-kernel-ir.md) | every IR node this compiler handles, with real snippets |
| 04 | [Compiler architecture](04-compiler-architecture.md) | how `compile.py` is organized, function by function |
| 05 | [Pointwise walkthrough](05-pointwise-walkthrough.md) | `p02_scale_shift` from IR to every PTX line |
| 06 | [Reduction walkthrough](06-reduction-walkthrough.md) | `r01_array_sum`: shared memory, `bar.sync`, the reduction strategy |
| 07 | [Glossary](07-glossary.md) | one or two sentences per term |
| 08 | [Next steps](08-next-steps.md) | what matmul, fused and attention kernels still need |

## Build log

| # | file | what happened |
|---|---|---|
| 10 | [Overview](10-log-overview.md) | first read of the task; mapping the annotated example to the IR |
| 11 | [p00_scalar_add](11-log-p00-scalar-add.md) | first working compiler: Tile, Load, Assign, Write |
| 12 | [Pointwise](12-log-pointwise.md) | p00-p03 pass; elementwise op table, tested op by op |
| 13 | [Reductions](13-log-reductions.md) | r01-r04 pass; loops, shuffles, shared memory, barriers |

## Other useful files

- [`../../TASK.md`](../../TASK.md) - the organizers' task and full IR reference table
- [`../../examples/k_sum_1_reduce.ptx`](../../examples/k_sum_1_reduce.ptx) - hand-written,
  commented PTX for r01
- [`../API.md`](../API.md) - the remote runner's API
