#!/bin/bash

python single/main_FedKNOW.py --alg=fedknow --dataset=MiniImageNet  --num_classes=100 --model=ResNeXt --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=5 --local_local_ep=2 --select_grad_num=10 --store_rate=0.1 --gpu=0

