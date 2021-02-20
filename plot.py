#!/usr/bin/env python3

from sys import argv
import json
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.figure import Figure

assert len(argv) == 3

data = json.loads(open(argv[1], "r").read())

rc("figure", figsize=(11.69, 8.27))
fig = plt.gcf()

fig = plt.figure()
ax = fig.add_subplot(111)

keys = list(data[0][1].keys())
keys.sort()

ALGO = {
    "OUR_BFCT": ((0, 1, 0), "solid"),
    "OUR_GORC": ((1, 0, 0), "solid"),
    "OUR_BFTO": ((0, 0, 1), "solid"),
    "EPS_BFCT": ((0, 1, 0), "dashed"),
    "EPS_GORC": ((1, 0, 0), "dashed"),
    "EPS_BFTO": ((0, 0, 1), "dashed"),
    "Z3": ((0, 0, 0), "solid"),
    "YICES2_SIMPLEX": ((1, 0, 1), "solid"),
    "YICES2_FW": ((1, 0, 1), "dashed"),
    "CVC4": ((0.5, 0.5, 0.5), "solid"),
    "VERIT": ((1, 1, 0), "solid"),
    "SMTINTERPOL": ((0, 1, 1), "solid"),
    "MATHSAT5": ((0.75, 0.5, 0.25), "solid")
}

assoc = {}

for k, i in enumerate(keys):
    px, py = [], []
    for j in data:
        if i not in j[1]:
            break
        px.append(j[0])
        py.append(j[1][i])
    ll = i
    if "EPS_" == i[:4]:
        ll = "DdM_" + i[4:]
    r = ax.loglog(px, py, label=ll)
    assoc[ll] = (px[-1], -py[-1])
    r[0].set_color(ALGO[i][0])
    r[0].set_linestyle(ALGO[i][1])
plt.xlabel("n")
plt.ylabel("Time (seconds)")

handles, labels = plt.gca().get_legend_handles_labels()
order = sorted(range(len(handles)), key=(lambda x: assoc[labels[x]]))
plt.legend([handles[i] for i in order], [labels[i] for i in order])

plt.savefig(argv[2], format='eps', bbox_inches='tight')
