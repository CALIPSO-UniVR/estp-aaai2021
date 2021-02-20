#!/usr/bin/env python3

from sys import argv
from math import log2
from random import seed, randint, randrange, shuffle

assert len(argv) == 5

def clamp(n, mi, ma):
    return min(ma, max(mi, n))

n = int(argv[1])
c = float(argv[2])
ham_len = clamp(float(argv[3]), 0, 1)
seed(int(argv[4]))
MAXE = int(log2(n))

edges = [(n - 1, 0, 0, ham_len > 0)] + [(x, x + 1, 0, False) for x in range(n - 1)]
s = set([(x, x + 1) for x in range(n - 1)] + [(n - 1, 0)])

if ham_len > 0 and ham_len < 1:
    t = int(ham_len * n)
    edges.append((0, t, 0, False))
    s.add((0, t))

for _ in range(int(n * (c - 1))):
    f, t = 0, 0
    while f == t or (f, t) in s:
        f, t = randrange(n), randrange(n)
    edges.append((f, t, randint(1, MAXE), randrange(2) == 0))
    s.add((f, t))

pi = [randint(-int(log2(n) * n), int(log2(n) * n)) for _ in range(n)]
mapper = list(range(n))
shuffle(mapper)
shuffle(edges)

for (f, t, v, s) in edges:
    sign = "<" if s else "<="
    v = pi[f] - pi[t] + v
    if v < 0:
        v = -v
        sign = ">" if s else ">="
        f, t = t, f
    print("x{} - x{} {} {}".format(mapper[t], mapper[f], sign, v))
