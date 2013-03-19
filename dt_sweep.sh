#!/bin/sh

for k in 80 100 120; do
    python convergence_runner.py --scheme hv -dx 1500 1000 -k $k --gpu;
done
