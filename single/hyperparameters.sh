#### Selection hyperparameters for FedKNOW
#!/bin/bash
cd single/
Agg_round=(5 10 15)
Local_epoch=(5 8 10)
LR=(0.0005 0.0008 0.001 0.005)
Lr_decrease=(1e-6 1e-5 1e-4)
Weight_rate=(0.05 0.1 0.2)
K=(5 10 20)
model=(6CNN ResNet18)
## The learning rate(or learning rate decrease) is mainly related to the model, so it only needs a few iterations to see whether the setting is reasonable
for m in ${model[@]}
do
	for lr in ${LR[@]}
	do
		for lrd in ${Lr_decrease}:
		do
			python main_hyperLR.py --lr=$lr --lr_decrease=$lrd --dataset=SVHN --model=$m
		done
	done
done
## Use the determined learning rate and other parameters to search the fedknow super parameters
for r in ${Agg_round[@]}
do
	for l in ${Local_epoch[@]}
	do
		for k in ${K[@]}
		do
			for wr in ${Weight_rate[@]}
			do
				python main_FedKNOW.py --epochs=$r --num_users=20 --frac=0.4 --dataset=SVHN --model=ResNet18 --lr=0.0008 --lr_decay=1e-5 --local_ep=$l --store_rate=$wr --agg_task=$k
			done
		done
	done
done

## GEM
Task_sample_rate=(0.05 0.1 0.15)
for r in ${Agg_round[@]}
do
	for l in ${Local_epoch[@]}
	do
		for sr in ${Task_sample_rate[@]}
		do
			python main_GEM.py --epochs=$r --num_users=20 --frac=0.4 --dataset=SVHN --model=ResNet18 --lr=0.0008 --lr_decay=1e-5 --local_ep=$l --store_rate=$sr
		done
	done
done

## BCN
for r in ${Agg_round[@]}
do
	for l in ${Local_epoch[@]}
	do
		for sr in ${Task_sample_rate[@]}
		do
			python main_BCN.py --epochs=$r --num_users=20 --frac=0.4 --dataset=SVHN --model=ResNet18 --lr=0.0008 --lr_decay=1e-5 --local_ep=$l --store_rate=$sr
		done
	done
done

## Co^2L
for r in ${Agg_round[@]}
do
	for l in ${Local_epoch[@]}
	do
		for sr in ${Task_sample_rate[@]}
		do
			python main_Co2L.py --epochs=$r --num_users=20 --frac=0.4 --dataset=SVHN --model=ResNet18 --lr=0.0008 --lr_decay=1e-5 --local_ep=$l --store_rate=$sr
		done
	done
done

## EWC
Regulation_parameter=(100 10000 40000)
for r in ${Agg_round[@]}
do
	for l in ${Local_epoch[@]}
	do
		for rp in ${Regulation_parameter[@]}
		do
			python main_EWC.py --epochs=$r --num_users=20 --frac=0.4 --dataset=SVHN --model=ResNet18 --lr=0.0008 --lr_decay=1e-5 --local_ep=$l --rp_alpha=$rp
		done
	done
done

## AGS-CL
for r in ${Agg_round[@]}
do
	for l in ${Local_epoch[@]}
	do
		for rp in ${Regulation_parameter[@]}
		do
			python main_AGS.py --epochs=$r --num_users=20 --frac=0.4 --dataset=SVHN --model=ResNet18 --lr=0.0008 --lr_decay=1e-5 --local_ep=$l --rp_alpha=$rp
		done
	done
done

## MAS
for r in ${Agg_round[@]}
do
	for l in ${Local_epoch[@]}
	do
		for rp in ${Regulation_parameter[@]}
		do
			python main_MAS.py --epochs=$r --num_users=20 --frac=0.4 --dataset=SVHN --model=ResNet18 --lr=0.0008 --lr_decay=1e-5 --local_ep=$l --rp_alpha=$rp
		done
	done
done