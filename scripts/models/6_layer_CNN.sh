#!/bin/bash

python single/main_FedKNOW.py --alg=fedknow --dataset=cifar100  --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=5 --local_local_ep=2 --select_grad_num=10 --store_rate=0.1 --gpu=0

python single/main_FedKNOW.py --alg=fedknow --dataset=FC100  --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=5 --local_local_ep=2 --select_grad_num=10 --store_rate=0.1 --gpu=0

python single/main_FedKNOW.py --alg=fedknow --dataset=CORe50  --num_classes=550 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=11 --epoch=150  --local_ep=5 --local_local_ep=2 --select_grad_num=10 --store_rate=0.1 --gpu=0
