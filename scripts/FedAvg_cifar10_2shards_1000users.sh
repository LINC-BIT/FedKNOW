#!/bin/bash

for RUN in 1 2 3 4 5; do
  python main_fedrep.py --dataset cifar10 --model cnn --num_classes 10 --epochs 100 --alg fedavg --lr 0.01 \
  --num_users 1000 --gpu 1 --shard_per_user 2 --test_freq 50 --local_ep 5 --frac 0.1 --local_bs 10

done
