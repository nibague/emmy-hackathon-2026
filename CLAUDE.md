# Context
Emmy Hackathon 2026: compile.py turns Emmy Kernel IR (*.ir.json) into PTX.
Contract: `python compile.py kernel.ir.json > out.ptx`. PTX goes to stdout ONLY; debug output goes to stderr.
Evaluation uses a HIDDEN set of similar kernels: never hardcode problem names or shapes. Generalize by IR node kind.
Deadline: 23:59 AoE Oct 28 (06:59 Bogota Oct 29). Goal: MVP = categories 1 (pointwise) and 2 (reductions).

# Environment
Repo root: /home/bbyga/GPU_mode/emmy-hackathon-2026. Always use .venv/bin/python. Never use sudo.
Test: .venv/bin/python submit.py <problem> --compiler compile.py

# Working rules (this is a learning project)
- Before coding, explain the plan in plain terms and which IR nodes are involved.
- One step at a time: one node kind or one problem, then test with submit.py.
- After each step, write docs/journal/NN-<topic>.md with: the IR snippet (*.ir.txt), the PTX
  it became and why (registers, .param, ld/st address spaces, %tid/%ctaid/%ntid), the submit.py
  result, and what failed and why.
- Commit after each passing problem.
