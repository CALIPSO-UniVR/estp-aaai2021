#!/usr/bin/env python3

from sys import argv
import json

data = {}
count = {}

for i in argv[1:]:
    res = json.loads(open(i, "r").read())
    for j in res:
        if j[0] not in data:
            data[j[0]] = {}
            count[j[0]] = {}
        for k in j[1]:
            if k not in data[j[0]]:
                data[j[0]][k] = 0
                count[j[0]][k] = 0
            data[j[0]][k] += j[1][k]
            count[j[0]][k] += 1

out = []

for i in data:
    for j in data[i]:
        data[i][j] /= count[i][j]
    out.append((i, data[i]))

print(json.dumps(out))
