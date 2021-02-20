#!/usr/bin/env python3

from sys import stdin

vars = set()

for line in stdin:
    line = line.strip().split(" ")
    assert line[1] == '-'
    v = [line[0], line[2]]
    z = line[4]
    s = line[3]
    for var in v:
        if var not in vars:
            vars.add(var)
            print("(define {} :: real)".format(var))
    print("(assert ({} (- {} {}) {}))".format(s, v[0], v[1], z))
print("(check)")
