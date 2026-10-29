# Kernel IR -> PTX compiler (Emmy Hackathon 2026)

This repo is a small compiler, `compile.py`, that reads a GPU kernel described in Emmy's
**Kernel IR** (a `*.ir.json` file) and prints **PTX**, NVIDIA's GPU assembly language. I built
it for the RiftStack Emmy Hackathon 2026 as a learning project, working with Claude Code. I
started without knowing PTX, and every step is written up in [`docs/journal/`](docs/journal/README.md)
so it can be followed from scratch.

## What was the hackathon for

[Emmy](https://github.com/cloudrift-ai/emmy) compiles PyTorch code into GPU kernels. Today it
writes each kernel as **CUDA C++** and builds it with **`nvcc`**. `nvcc` first turns C++ into
PTX, then runs **`ptxas`** to turn PTX into machine code. The challenge was to **skip the C++
step and emit PTX directly**, so only `ptxas` has to run. The organizers report this is about
19x faster than the full `nvcc` pipeline (their figure, not measured here). That matters
because Emmy's autotuner compiles **thousands of variants** of each kernel to find the fastest
one, so compile time adds up.

The organizers' original task description is kept in
[docs/HACKATHON-ORIGINAL-README.md](docs/HACKATHON-ORIGINAL-README.md). Its relative links
(`TASK.md`, `docs/API.md`, ...) are written relative to the repo root.

## Status

Verified on 2026-10-29 with `.venv/bin/python submit.py <problem> --compiler compile.py`
(runner GPU: RTX 5090, `sm_120`). "max abs err" is the largest difference from the expected
output.

| problem | category | result | detail |
|---|---|---|---|
| p00_scalar_add | 1 pointwise | **PASS** | max abs err 0 |
| p01_vector_add | 1 pointwise | **PASS** | max abs err 0 |
| p02_scale_shift | 1 pointwise | **PASS** | max abs err 0 |
| p03_relu_mul | 1 pointwise | **PASS** | max abs err 0 |
| r01_array_sum | 2 reduction | **PASS** | max abs err 3.8e-06 (tolerance 1e-4) |
| r02_big_sum | 2 reduction | **PASS** | max abs err 1.5e-05 (tolerance 1e-3) |
| r03_mean | 2 reduction | **PASS** | max abs err 9.3e-10 (tolerance 1e-4) |
| r04_row_sum | 2 reduction | **PASS** | max abs err 5.7e-06 (tolerance 1e-4) |
| f01_rmsnorm | 4 fused | **PASS** | max abs err 1.9e-06. Uses only category-2 node kinds |
| f03_l2norm | 4 fused | **PASS** | max abs err 1.5e-08. Uses only category-2 node kinds |
| f02_row_softmax | 4 fused | FAIL | compiler stops: `NotImplementedError: Accum with base` |
| m01_matmul_512 | 3 matmul | FAIL | compiler stops: `NotImplementedError: statement Init` |
| m02_matmul_rect | 3 matmul | FAIL | compiler stops: `NotImplementedError: statement Init` |
| a01_attention | 5 attention | FAIL | compiler stops: `NotImplementedError: statement Loop` |
| a02_attention_128 | 5 attention | FAIL | compiler stops: `NotImplementedError: statement Loop` |

The MVP (all of categories 1 and 2) passes. The failing problems never reach the GPU: the
compiler exits with an error naming the first IR feature it does not support yet. See
[docs/journal/08-next-steps.md](docs/journal/08-next-steps.md).

## How to run

All commands are run from the repo root.

**Setup** (Python 3.12 or newer; no GPU, CUDA or PyTorch needed):

```bash
python3 -m venv .venv
.venv/bin/pip install emmy-ml==0.3.11 numpy requests
```

These steps are the organizers' instructions adapted to a venv. My environment has Python
3.12.3, emmy-ml 0.3.11, numpy 2.5.4 and requests 2.34.2. I have not re-run the setup from a
fresh clone (TO VERIFY).

**Compile one kernel** (PTX goes to stdout, errors go to stderr):

```bash
.venv/bin/python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json > out.ptx
```

**Test one problem end to end.** `submit.py` runs the compiler, sends the PTX to the
organizers' hosted runner (`runner.riftstack.ai`), which assembles it with `ptxas`, runs it on
an RTX 5090, and compares the output with `expected.npz`:

```bash
.venv/bin/python submit.py p00_scalar_add --compiler compile.py
```

**Test every MVP problem:**

```bash
for p in p00_scalar_add p01_vector_add p02_scale_shift p03_relu_mul \
         r01_array_sum r02_big_sum r03_mean r04_row_sum; do
  .venv/bin/python submit.py $p --compiler compile.py
done
```

**Test a hand-written PTX file** (here, the organizers' annotated example):

```bash
.venv/bin/python submit.py r01_array_sum examples/k_sum_1_reduce.ptx
```

## Repo map

```
compile.py          the compiler (the whole project lives in this one file)
submit.py           organizers' test tool: compile + run on the remote GPU + check
problems/           15 open problems, one folder each:
  <name>/*.ir.json    the input: Kernel IR
  <name>/*.ir.txt     the same IR, pretty-printed for humans
  <name>/*.ref.cu     Emmy's own CUDA C++ for it (reference only, never used by compile.py)
  <name>/meta.json    launch geometry (grid, block), argument order, tolerance
  <name>/expected.npz the correct output
examples/           organizers' hand-written, commented PTX for r01_array_sum
docs/journal/       learning journal: concepts, walkthroughs, build log
docs/API.md         organizers' runner API
docs/HACKATHON-ORIGINAL-README.md   organizers' original README
TASK.md, MVP.md     organizers' task description and minimum bar
tools/              organizers' scripts that generated the problems
CLAUDE.md           working rules for Claude Code in this repo
```

## Start reading here

**[docs/journal/README.md](docs/journal/README.md)** is the index of the journal, in
a suggested reading order. It goes from "what is a GPU compiler" to line-by-line walkthroughs of
the PTX this compiler produces.
