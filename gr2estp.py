#!/usr/bin/env python3

while True:
    try:
        s = input()
    except:
        break
    s = s.strip().split(" ")
    if s[0] == "c":
        continue
    elif s[0] == "p":
        assert s[1] == "sp"
        n = int(s[2])
        m = int(s[3])
    elif s[0] == "a":
        print("x{} - x{} <= {}".format(*s[1:]));
    else:
        assert False
