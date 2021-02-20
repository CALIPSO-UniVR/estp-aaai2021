#!/bin/bash -xe

ulimit -s unlimited

c++ -o orrz.bin -O3 -march=native -flto -std=c++17 orrz.cpp

TIMELIMIT=100

for ((i=0;i<8;i++)); do
    python3 benchmark.py 8 1 $i $TIMELIMIT > bench.ham100.${i}.json
    python3 benchmark.py 8 0 $i $TIMELIMIT > bench.ham000.${i}.json
    python3 benchmark.py 8 0.01 $i $TIMELIMIT > bench.ham001.${i}.json
    python3 benchmark.py 8 0.25 $i $TIMELIMIT > bench.ham025.${i}.json
done
