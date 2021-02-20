#!/usr/bin/env python3

from sys import argv, stderr
import json
from subprocess import run, PIPE, DEVNULL
from time import time
from math import sqrt

assert len(argv) == 5

mult = argv[1]
ham_len = argv[2]
seed = argv[3]

TIMELIMIT = float(argv[4])

OUR_BIN = "./orrz.bin"

ALGO = {
    "OUR_BFCT": ([OUR_BIN, "32"], "ESTP"),
    "OUR_GORC": ([OUR_BIN, "16"], "ESTP"),
    "OUR_BFTO": ([OUR_BIN, "8"], "ESTP"),
    "EPS_BFCT": ([OUR_BIN, "4"], "ESTP"),
    "EPS_GORC": ([OUR_BIN, "2"], "ESTP"),
    "EPS_BFTO": ([OUR_BIN, "1"], "ESTP"),
    "Z3": (["z3", "-smt2", "-in"], "SMT2"),
    "YICES2_SIMPLEX": (["yices", "--arith-solver=simplex", "--logic=QF_RDL"], "YS"),
    "YICES2_FW": (["yices", "--arith-solver=floyd-warshall", "--logic=QF_RDL"], "YS"),
    "CVC4": (["cvc4", "--lang=smt2"], "SMT2"),
    "VERIT": (["veriT", "--disable-print-success", "--input=smtlib2"], "SMT2"),
    "SMTINTERPOL": (["smtinterpol", "-smt2", "-no-success", "-w"], "SMT2"),
    "MATHSAT5": (["mathsat", "-input=smt2"], "SMT2")
}

START = 2**5
active = set(ALGO.keys())
true_n = START
results = []
while len(active) > 0:
    n = int(round(true_n))
    needed = set()
    for algo in active:
        needed.add(ALGO[algo][1])
    inp = dict()
    inp["ESTP"] = run(["pypy3", "hamilton.py", str(n), mult, ham_len, seed], stdout=PIPE, stderr=DEVNULL).stdout.decode()
    if "SMT2" in needed:
        inp["SMT2"] = run(["pypy3", "estp2smt2.py"], stdout=PIPE, stderr=DEVNULL, input=inp["ESTP"].encode()).stdout.decode()
    if "YS" in needed:
        inp["YS"] = run(["pypy3", "estp2ys.py"], stdout=PIPE, stderr=DEVNULL, input=inp["ESTP"].encode()).stdout.decode()
    line = (n, dict())
    to_remove = []
    for algo in active:
        try:
            start = time()
            r = run(ALGO[algo][0], stdout=DEVNULL, stderr=DEVNULL, input=inp[ALGO[algo][1]].encode(), timeout=TIMELIMIT)
            end = time()
            line[1][algo] = end - start
        except:
            to_remove.append(algo)
            print("DISABLING {}".format(algo), file=stderr)
    for algo in to_remove:
        active.remove(algo)
    if len(line[1]) > 0:
        results.append(line)
    print("DONE n={}".format(n), file=stderr)
    true_n = true_n * sqrt(2)
print(json.dumps(results))
