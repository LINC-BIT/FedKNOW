#!/bin/bash

for RUN in 1 2; do
  python main_fedrep.py --dataset sent140 --model res --epochs 50 --alg fedrep --lr 0.01 \
  --num_users 183 --gpu 1 --test_freq 50 --local_ep 15  --frac 0.1 --local_rep_ep 5 --local_bs 4
done
