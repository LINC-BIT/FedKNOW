#!/bin/bash

for RUN in 1 2 3 4 5; do
  python main_fedrep.py --dataset sent140 --model res --num_classes 2 --epochs 50 --alg fedavg --lr 0.01 \
  --num_users 183 --gpu 0 --test_freq 50 --local_ep 10 --frac 0.1 --local_bs 4

done
