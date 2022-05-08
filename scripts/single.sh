# for FedKNOW:
python single/main_FedKNOW.py --alg=fedknow --dataset=cifar100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8 --local_local_ep=4 --select_grad_num=10 --store_rate=0.1 --gpu=0
python single/main_FedKNOW.py --alg=fedknow --dataset=FC100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8 --local_local_ep=4 --select_grad_num=10 --store_rate=0.1 --gpu=0
python single/main_FedKNOW.py --alg=fedknow --dataset=CORe50 --num_classes=550 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=11 --epoch=150  --local_ep=8 --local_local_ep=4 --select_grad_num=10 --store_rate=0.1 --gpu=0
python single/main_FedKNOW.py --alg=fedknow --dataset=MiniImageNet  --num_classes=100 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=5 --local_local_ep=2 --select_grad_num=10 --store_rate=0.1 --gpu=0
python single/main_FedKNOW.py --alg=fedknow --dataset=TinyImageNet --num_classes=200 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=20 --epoch=150  --local_ep=5 --local_local_ep=2 --select_grad_num=10 --store_rate=0.1 --gpu=0

# for AGS:
python single/main_AGS.py --alg=AGS --dataset=cifar100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8  --gpu=0
python single/main_AGS.py --alg=AGS --dataset=FC100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8  --gpu=0
python single/main_AGS.py --alg=AGS --dataset=CORe50 --num_classes=550 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=11 --epoch=150  --local_ep=8  --gpu=0
python single/main_AGS.py --alg=AGS --dataset=MiniImageNet  --num_classes=100 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=5  --gpu=0
python single/main_AGS.py --alg=AGS --dataset=TinyImageNet --num_classes=200 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=20 --epoch=150  --local_ep=5  --gpu=0

# for APFL:
python single/main_apfl.py --alg=APFL --dataset=cifar100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8 --APFLalpha=0.5 --gpu=0
python single/main_apfl.py --alg=APFL --dataset=FC100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8 --APFLalpha=0.5 --gpu=0
python single/main_apfl.py --alg=APFL --dataset=CORe50 --num_classes=550 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=11 --epoch=150  --local_ep=8 --APFLalpha=0.5 --gpu=0
python single/main_apfl.py --alg=APFL --dataset=MiniImageNet  --num_classes=100 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=5 --APFLalpha=0.5 --gpu=0
python single/main_apfl.py --alg=APFL --dataset=TinyImageNet --num_classes=200 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=20 --epoch=150  --local_ep=5 --APFLalpha=0.5 --gpu=0

# for BCN:
python single/main_BCN.py --alg=BCN --dataset=cifar100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8 --BCNbeta=2 --gpu=0
python single/main_BCN.py --alg=BCN --dataset=FC100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8 --BCNbeta=2 --gpu=0
python single/main_BCN.py --alg=BCN --dataset=CORe50 --num_classes=550 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=11 --epoch=150  --local_ep=8 --BCNbeta=2 --gpu=0
python single/main_BCN.py --alg=BCN --dataset=MiniImageNet  --num_classes=100 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=5 --BCNbeta=2 --gpu=0
python single/main_BCN.py --alg=BCN --dataset=TinyImageNet --num_classes=200 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=20 --epoch=150  --local_ep=5 --BCNbeta=2 --gpu=0

# for Co2L:
python single/main_Co2L.py --alg=Co2L --dataset=cifar100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8 --Co2Lis_train=True --gpu=0
python single/main_Co2L.py --alg=Co2L --dataset=FC100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8 --Co2Lis_train=True --gpu=0
python single/main_Co2L.py --alg=Co2L --dataset=CORe50 --num_classes=550 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=11 --epoch=150  --local_ep=8 --Co2Lis_train=True --gpu=0
python single/main_Co2L.py --alg=Co2L --dataset=MiniImageNet  --num_classes=100 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=5 --Co2Lis_train=True --gpu=0
python single/main_Co2L.py --alg=Co2L --dataset=TinyImageNet --num_classes=200 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=20 --epoch=150  --local_ep=5 --Co2Lis_train=True --gpu=0

# for EWC:
python single/main_EWC.py --alg=EWC --dataset=cifar100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8 --lambda=10000 --gpu=0
python single/main_EWC.py --alg=EWC --dataset=FC100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8 --lambda=10000 --gpu=0
python single/main_EWC.py --alg=EWC --dataset=CORe50 --num_classes=550 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=11 --epoch=150  --local_ep=8 --lambda=10000 --gpu=0
python single/main_EWC.py --alg=EWC --dataset=MiniImageNet  --num_classes=100 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=5 --lambda=10000 --gpu=0
python single/main_EWC.py --alg=EWC --dataset=TinyImageNet --num_classes=200 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=20 --epoch=150  --local_ep=5 --lambda=10000 --gpu=0

# for FedAvg:
python single/main_fedavg.py --alg=fedavg --dataset=cifar100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8  --gpu=0
python single/main_fedavg.py --alg=fedavg --dataset=FC100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8  --gpu=0
python single/main_fedavg.py --alg=fedavg --dataset=CORe50 --num_classes=550 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=11 --epoch=150  --local_ep=8  --gpu=0
python single/main_fedavg.py --alg=fedavg --dataset=MiniImageNet  --num_classes=100 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=5  --gpu=0
python single/main_fedavg.py --alg=fedavg --dataset=TinyImageNet --num_classes=200 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=20 --epoch=150  --local_ep=5  --gpu=0

# for FedRep:
python single/main_FedRep.py --alg=FedRep --dataset=cifar100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8 --local_local_ep=4 --gpu=0
python single/main_FedRep.py --alg=FedRep --dataset=FC100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8 --local_local_ep=4 --gpu=0
python single/main_FedRep.py --alg=FedRep --dataset=CORe50 --num_classes=550 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=11 --epoch=150  --local_ep=8 --local_local_ep=4 --gpu=0
python single/main_FedRep.py --alg=FedRep --dataset=MiniImageNet  --num_classes=100 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=5 --local_local_ep=2 --gpu=0
python single/main_FedRep.py --alg=FedRep --dataset=TinyImageNet --num_classes=200 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=20 --epoch=150  --local_ep=5 --local_local_ep=2 --gpu=0

# for GEM:
python single/main_GEM.py --alg=GEM --dataset=cifar100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8 --n_memories=20 --gpu=0
python single/main_GEM.py --alg=GEM --dataset=FC100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8 --n_memories=20 --gpu=0
python single/main_GEM.py --alg=GEM --dataset=CORe50 --num_classes=550 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=11 --epoch=150  --local_ep=8 --n_memories=100 --gpu=0
python single/main_GEM.py --alg=GEM --dataset=MiniImageNet  --num_classes=100 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=5 --n_memories=20 --gpu=0
python single/main_GEM.py --alg=GEM --dataset=TinyImageNet --num_classes=200 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=20 --epoch=150  --local_ep=5 --n_memories=20 --gpu=0

# for MAS:
python single/main_MAS.py --alg=MAS --dataset=cifar100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8  --lambda=400 --gpu=0
python single/main_MAS.py --alg=MAS --dataset=FC100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8  --lambda=400 --gpu=0
python single/main_MAS.py --alg=MAS --dataset=CORe50 --num_classes=550 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=11 --epoch=150  --local_ep=8  --lambda=400 --gpu=0
python single/main_MAS.py --alg=MAS --dataset=MiniImageNet  --num_classes=100 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=5  --lambda=400 --gpu=0
python single/main_MAS.py --alg=MAS --dataset=TinyImageNet --num_classes=200 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=20 --epoch=150  --local_ep=5  --lambda=400 --gpu=0

# for WEIT:
python single/main_FedKNOW.py --alg=WEIT --dataset=cifar100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8  --gpu=0
python single/main_FedKNOW.py --alg=WEIT --dataset=FC100 --num_classes=100 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=8  --gpu=0
python single/main_FedKNOW.py --alg=WEIT --dataset=CORe50 --num_classes=550 --model=6layer_CNN --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=11 --epoch=150  --local_ep=8  --gpu=0
python single/main_FedKNOW.py --alg=WEIT --dataset=MiniImageNet  --num_classes=100 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=150  --local_ep=5  --gpu=0
python single/main_FedKNOW.py --alg=WEIT --dataset=TinyImageNet --num_classes=200 --model=ResNet18 --num_users=20  --shard_per_user=5 --frac=0.4 --local_bs=40 --optim=SGD --lr=0.001 --lr_decay=1e-4 --task=20 --epoch=150  --local_ep=5  --gpu=0

