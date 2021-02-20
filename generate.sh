#!/bin/bash -xe

for i in ham100 ham000 ham001 ham025; do
    python3 aggregate.py bench.${i}.*.json > bench.${i}.json
    python3 plot.py bench.${i}.json ${i}.eps
done
