#!/usr/bin/env python3
"""Kernel IR -> PTX compiler (stub). Usage: python compile.py kernel.ir.json > out.ptx"""
import sys
import json
from emmy.compiler.graph import Graph


def main():
    if len(sys.argv) != 2:
        print("usage: python compile.py kernel.ir.json > out.ptx", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        graph = Graph.from_dict(json.load(f))   # IR file -> Emmy's Python objects

    # Debug info goes to STDERR, never stdout: stdout IS the PTX file.
    for name, node in graph.nodes.items():
        op = getattr(node, "op", None)
        print(f"node {name}: {type(op).__name__}", file=sys.stderr)

    # TODO: walk the IR tree and print PTX to stdout


if __name__ == "__main__":
    main()
