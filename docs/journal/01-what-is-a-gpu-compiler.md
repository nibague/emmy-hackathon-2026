# 01 - What is a GPU compiler?

A **compiler** translates a program from one language into another, usually from something
easy for humans to write into something a machine can run. A GPU compiler does this for code
that runs on a graphics card. It is usually not one step but a **pipeline** of several
translations, each into a slightly lower-level language.

## The full pipeline

```
  PyTorch code                    torch.randn(1024).sum()
       |
       |  Emmy traces it (records which tensor ops run)
       v
  Graph                           nodes: InputOp x, KernelOp sum_1
       |                          (top level of every *.ir.json)
       |  Emmy schedules it: how many threads, how to split the work,
       |  what to unroll, where to use shared memory
       v
  Kernel IR                       Tile / StridedLoop / Load / Accum / ...
       |                          (KernelOp.body inside the *.ir.json)
       |
       +--------------------------+
       |  Emmy's normal path      |  THIS PROJECT
       v                          |
  CUDA C++  (*.ref.cu)            |
       |                          |
       |  nvcc front end          |  compile.py
       v                          |
  PTX  <--------------------------+
       |                          virtual GPU assembly, text
       |  ptxas
       v
  SASS (inside a "cubin" file)    real machine code for one GPU model
       |
       |  CUDA driver loads it and launches it with a grid/block size
       v
  GPU runs it
```

### Each stage and why it exists

| stage | what it is | why it exists |
|---|---|---|
| **PyTorch** | Python code on tensors, e.g. `x * 2.0 + 1.0` | what people write |
| **Graph** | a list of operations and the buffers between them | a clean description of *what* to compute, without Python |
| **Kernel IR** | one GPU kernel as a tree of statements: which thread loads what, loops, writes | decides *how* to compute it on a GPU (thread layout, loops, shared memory) |
| **CUDA C++** | C++ with GPU extensions (`threadIdx.x`, `__shared__`) | a human-readable kernel that `nvcc` can build |
| **PTX** | a text assembly language for a *virtual* NVIDIA GPU | a stable target that works across GPU generations |
| **SASS** | binary machine code for one specific GPU architecture | the only thing the hardware actually executes |

An **IR** ("intermediate representation") is any in-between language a compiler uses
internally. Emmy has several. TASK.md mentions Loop IR and Tile IR before Kernel IR. How
exactly Emmy moves between them is not something this project needed (TO VERIFY if curious).
This project only reads the last one, Kernel IR.

## PTX vs SASS

**PTX** (Parallel Thread eXecution) is a *virtual* instruction set ("virtual ISA"):

- It is plain text you can read and write by hand:
  ```
  ld.global.f32  %f1, [%rd8];     // load a float from GPU memory
  add.f32        %f3, %f1, %f2;   // add two floats
  ```
- It has **unlimited registers**. You name `%f1, %f2, ... %f999` freely.
- It is **stable**. NVIDIA documents it, and PTX written for one GPU generation can be compiled
  for later ones. The line `.target sm_89` says "this needs at least sm_89 features".

**SASS** is the *real* machine code:

- It is binary and specific to one architecture (the code for an RTX 4090 is not the code for an
  RTX 5090).
- It has a **fixed, small number of registers** per thread, so someone must decide which
  values live in which physical register (this is **register allocation**).
- NVIDIA does not fully document it.

**`ptxas`** is the tool that turns PTX into SASS. It is a real optimizing compiler: it does
register allocation, instruction scheduling and dead-code removal. That is why our compiler can
be simple and "wasteful" (fresh registers everywhere, repeated index math). `ptxas` cleans it up.
For example, `submit.py` reports `16 registers` for p00 even though our PTX declares about 40.

## Where nvcc and ptxas fit

`nvcc` is NVIDIA's CUDA compiler driver. Building a `.cu` file runs, roughly:

```
.cu  --[nvcc front end: C++ parsing + LLVM-based optimizer]-->  PTX  --[ptxas]-->  SASS
```

The front end (parsing C++, optimizing) is the slow part. Emmy's autotuner may build thousands
of kernel variants, so skipping it matters. The organizers' README says going straight to `ptxas`
is about 19x faster than full `nvcc` (their measurement, not repeated here).

## Where this project sits

```
Kernel IR (*.ir.json)  -->  compile.py  -->  PTX (stdout)  -->  ptxas  -->  SASS  -->  GPU
                            ^^^^^^^^^^
                            this repo
```

`compile.py` replaces "CUDA C++ + nvcc front end". It reads the Kernel IR tree and prints PTX
text. Everything after that (`ptxas`, loading, running, checking) is done by the organizers'
remote runner, which `submit.py` calls.

### A detail: sm_89 vs sm_120

Our PTX says `.target sm_89` (the problems were scheduled for an RTX 4090, which is sm_89). The
runner is an RTX 5090 (sm_120). Its `/health` endpoint reports
`"default_arch":"sm_120"`. The runner's `ptxas` compiles our sm_89 PTX into sm_120 SASS, and that
works because PTX for an older target can be compiled for a newer GPU. All passing problems in
the README were run this way.
