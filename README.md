# FedKNOW

English | [简体中文](README_zh-CN.md)
![](https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/FedKNOW_architecture.png)

## Table of contents
- [1 Introduction](#1-introduction)
- [2 How to get started](#2-how-to-get-started)
  * [2.1 Setup](#21-setup)
  * [2.2 Usage](#22-usage)
- [3 Supported models in image classification](#3-supported-models-in-image-classification)
- [4 Experiments](#4-experiments)
  * [4.1 Under different workloads (model and dataset)](#41-under-different-workloads-model-and-dataset)
  * [4.2 Under different network bandwidths](#42-under-different-network-bandwidths)
  * [4.3 Large scale](#43-large-scale)
  * [4.4 Long task sequence](#44-long-task-sequence)
  * [4.5 Under different parameter settings](#45-under-different-parameter-settings)
  * [4.6 Applicability on different networks](#46-applicability-on-different-networks)

## 1 Introduction
FedKNOW is designed to achieve SOTA performance (accuracy, time, and communication cost etc.) in federated continual learning setting. It currently supports eight different networks of image classification: ResNet, ResNeXt, MobileNet, WideResNet, SENet, ShuffleNet, Inception and DenseNet.
- [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html): this model consists of multiple convolutional layers and pooling layers that extract the information in image. Typically, ResNet suffers from gradient vanishing (exploding) and performance degrading when the network is  deep. ResNet thus adds BatchNorm to alleviate gradient vanishing (exploding) and adds residual connection to alleviate the performance degrading. 
- [Inception](https://ieeexplore.ieee.org/document/7780677): To extract high dimensional features, a typical method is to use deeper network which makes the size of the network too large. To address this issue, Inception Module is proposed to widen the network. This can maintain the performance and reduce the number of parameters. Inception Module firstly leverages 1x1 convolution to aggregate the information, and leverages multi-scale convolutions to obtain multi-scale feature maps, finally concatenate them to produce the final feature map.
- [ResNeXt](https://arxiv.org/abs/1611.05431): ResNeXt combines Inception and ResNet. It first simplifies the Inception Module to make each of its branch have the same structure and then constructs the network as ResNet-style.
- [WideResNet](http://www.bmva.org/bmvc/2016/papers/paper087/index.html): WideResNet widens the residual connection of ResNet to improve its performance and reduces the number of its parameters. Besides, WideResNet uses Dropout regularization to further improve its generalization.
- [MobileNet](https://arxiv.org/abs/1801.04381): MobileNet is a lightweight convolutional network which widely uses the depthwise separable convolution.
- [SENet](https://ieeexplore.ieee.org/document/341010): SENet imports channel attention to allow the network focus the more important features. In SENet, a Squeeze & Excitation Module uses the output of a block as input, produces an channel-wise importance vector, and multiplies it into the original output of the block to strengthen the important channels and weaken the unimportant channels.
- [ShuffleNet](https://arxiv.org/abs/1807.11164): ShuffleNet is a lightweight network. It imports the pointwise group convolution and channel shuffle to greatly reduce the computation cost. It replaces the 1x1 convolution of ResBlock with the group convolution and add channel shuffle to it.
- [DenseNet](https://arxiv.org/pdf/1707.06990.pdf): DenseNet extends ResNet by adding connections between each blocks to aggregate all multi-scale features.
## 2 How to get started
### 2.1 Setup
**Requirements**

- Edge devices such as Jetson AGX, Jetson TX2, Jetson Xavier NX and Jetson Nano
- Linux and Windows 
- Python 3.6+
- PyTorch 1.9+
- CUDA 10.2+ 

**Preparing the virtual environment**

1. Create a conda environment and activate it.
	```shell
	conda create -n FedKNOW python=3.7
	conda active FedKNOW
	```
	
2. Install PyTorch 1.9+ in the [offical website](https://pytorch.org/). A NVIDIA graphics card and PyTorch with CUDA are recommended.

  ![image](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ec360791671f4a4ab322eb4e71cc9e62~tplv-k3u1fbpfcp-zoom-1.image)

3. Clone this repository and install the dependencies.
  ```shell
  git clone https://github.com/LINC-BIT/FedKNOW.git
  pip install -r requirements.txt
  ```
### 2.2 Usage
- **Single device**

    Run FedKNOW or the baselines:
    ```shell
    python single/main_FedKNOW.py --alg fedknow --dataset [dataset] --model [mdoel]
    --num_users [num_users]  --shard_per_user [shard_per_user] --frac [frac] 
    --local_bs [local_bs] --lr [lr] --task [task] --epoch [epoch]  --local_ep 
    [local_ep] --local_local_ep [local_local_ep] --store_rate [store_rate] 
    --select_grad_num [select_grad_num] --gpu [gpu]
    ```
    Arguments:
    - `alg`: the algorithm, e.g. `FedKNOW`, `FedRep`, `GEM`
    
    - `dataset` : the dataset, e.g. `cifar100`, `FC100`, `CORe50`, `MiniImagenet`, `Tinyimagenet`
    
    - `model`: the model, e.g. `6-Layers CNN`, `ResNet18`
    
    - `num_users`: the number of clients
    
    - `shard_per_user`: the number of classes in each client
    
    - `frac`: the percentage of clients participating in training in each epoch
    
    - `local_bs`: the batch size in each client
    
    - `lr`: the learning rate
    
    - `task`: the number of tasks
    
    - `epochs`: the number of communications between clients and the server
    
    - `local_ep`:the number of epochs in clients
    
    - `local_local_ep`:the number of updating the local parameters in clients
    
    - `store_rate`: the store rate of model parameters in FedKNOW
    
    - `select_grad_num`: the number of choosing the old grad in FedKNOW
    
    - `gpu`: GPU id
    
      More details refer to `utils/option.py`. The configurations of all algorithms are located in `scripts/single.sh`.
- **Multiple devices**
  
    1. Limit the network bandwidth to emulate the real long distance transmission:
    
        ```shell
        sudo wondershaper [adapter] [download rate] [upload rate]
        ```
    2. Launch the server:
        ```shell
        python multi/server.py --num_users [num_users] --frac [frac] --ip [ip]
        ```
    3. Launch the clients:
        ```shell
        python multi/main_FedKNOW.py --client_id [client_id] --alg [alg]
        --dataset [dataset] --model[mdoel]  --shard_per_user [shard_per_user] 
        --local_bs [local_bs] --lr [lr] --task [task] --epoch [epoch]  --local_ep 
        [local_ep] --local_local_ep [local_local_ep]  --store_rate [store_rate] 
        --select_grad_num [select_grad_num] --gpu [gpu] --ip [ip]
        ```
        Arguments:
        - `client_id`: the id of the client
        
        - `ip`: IP address of the server
        
          The other arguments is the same as the one in single device setting. More details refer to `utils/option.py`. The configurations of all algorithms are located in `scripts/multi.sh`. 
## 3 Supported models in image classification
||Model Name|Data|Script|
|--|--|--|--|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[6 layer_CNN (NeurIPS'2020)](https://proceedings.neurips.cc/paper/2020/hash/258be18e31c8188555c2ff05b4d542c3-Abstract.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;[Cifar100](http://www.cs.toronto.edu/~kriz/cifar.html)&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; <br>&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;[FC100](https://paperswithcode.com/dataset/fc100) &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;<br>&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp;[CORe50](https://vlomonaco.github.io/core50/index.html#download) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/6_layer_CNN.sh) &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[ResNet (CVPR'2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp;<br>&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[TinyImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;| &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/ResNet.sh) &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[MobileNetV2 (CVPR'2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/MobileNet.sh) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[ResNeXt (CVPR'2017)](https://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/ResNext.sh) &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[InceptionV3(CVPR'2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/inception_v3.py) &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[WideResNet (BMVC'2016)](https://arxiv.org/abs/1605.07146) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php)  &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/WideResNet.sh)&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[ShuffleNetV2 (ECCV'2018)](https://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php)  &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/ShuffleNet.sh)&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[DenseNet(CVPR'2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/DenseNet.sh)&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[SENet (CVPR'2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/SENet.sh)&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;|
## 4 Experiemts
### 4.1 Under different workloads (model and dataset)

1. **Run**
   
    **Launch the server：**
    
    ```shell
    python multi/server.py --epochs=150 --num_users=20 --frac=0.4 --ip=127.0.0.1:8000
    ```
    **Launch the clients：**
    
    - 6-layer CNN on Cifar100
        ```shell
        for ((i=0;i<20;i++));
        do
            python multi/main_FedKNOW.py --client_id=$i --model=6_layerCNN --dataset=cifar100 --num_classes=100 --task=10 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
        done
        ```
    - 6-layer CNN on FC100
      
        ```shell
        for ((i=0;i<20;i++));
        do
            python multi/main_FedKNOW.py --client_id=$i --model=6_layerCNN --dataset=FC100 --num_classes=100 --task=10 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
        done
        ```
    - 6-layer CNN on CORe50
        ```shell
        for ((i=0;i<20;i++));
        do
            python multi/main_FedKNOW.py --client_id=$i --model=6_layerCNN --dataset=CORe50 --num_classes=550 --task=11 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
        done
        ```
    - ResNet18 on MiniImageNet
        ```shell
        for ((i=0;i<20;i++));
        do
            python multi/main_FedKNOW.py --client_id=$i --model=ResNet --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 --ip=127.0.0.1:8000
        done
        ```
    - ResNet18 on TiniImageNet
        ```shell
        for ((i=0;i<20;i++));
        do
            python multi/main_FedKNOW.py --client_id=$i --model=ResNet --dataset=TinyImageNet --num_classes=200 --task=20 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 --ip=127.0.0.1:8000
        done
        ```
    **Note:** Keep the IP address of the server and clients the same. `--ip=127.0.0.1:8000` represents testing locally. If there're multiple edge devices, you should do `--ip=<IP of the server>`.
    
2. **Result**

    - **The accuracy trend overtime time under different workloads**(X-axis represents the time and Y-axis represents the inference accuracy)
    ![](https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/difworkerloader.png)
### 4.2 Under different network bandwidths
1. **Run**

    **Limit the network bandwidth of the server:**
    
    ```shell
    # The maximal download rate and upload rate are 1000KB/s. 
    # In practice this is not so precise so you can adjust it.
    sudo wondershaper [adapter] 1000 1000 
    ```
    **Check the network state of the server**
    
    ```shell
    sudo nload -m
    ```
    
    **Launch the server：**
    ```shell
    python multi/server.py --epochs=150 --num_users=20 --frac=0.4 --ip=127.0.0.1:8000
    ```
    **Launch the clients：**
    
    - 6-layer CNN on Cifar100
      
        ```shell
        for ((i=0;i<20;i++));
        do
            python multi/main_FedKNOW.py --client_id=$i --model=6_layerCNN --dataset=cifar100 --num_classes=100 --task=10 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
        done
        ```
    - 6-layer CNN on FC100
        
        ```shell
        for ((i=0;i<20;i++));
        do
            python multi/main_FedKNOW.py --client_id=$i --model=6_layerCNN --dataset=FC100 --num_classes=100 --task=10 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
        done
        ```
    - 6-layer CNN on CORe50
        ```shell
        for ((i=0;i<20;i++));
        do
            python multi/main_FedKNOW.py --client_id=$i --model=6_layerCNN --dataset=CORe50 --num_classes=550 --task=11 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
        done
        ```
    - ResNet18 on MiniImageNet
        ```shell
        for ((i=0;i<20;i++));
        do
            python multi/main_FedKNOW.py --client_id=$i --model=ResNet18 --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 --ip=127.0.0.1:8000
        done
        ```
    - ResNet18 on TiniImageNet
        ```shell
        for ((i=0;i<20;i++));
        do
            python multi/main_FedKNOW.py --client_id=$i --model=ResNet18 --dataset=TinyImageNet --num_classes=200 --task=20 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 --ip=127.0.0.1:8000
        done
        ```
2. **Result**

    - **The communication time under different workloads and maximal network bandwidth 1MB/s** (X-axis represents the dataset and Y-axis represents the communication time)
        
        <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/difctime.png" width="50%">
        
    - **The communication time under different network bandwidths** (X-axis represents the network bandwidth and Y-axis represents the communication time)
        
        <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/difbandwidth.png" width="50%">
        
### 4.3 Large scale
1. **Run**

    ```shell
    # 50 clients
    python single/main_FedKNOW.py --epochs=150 --num_users=50 --frac=0.4 --model=ResNet18 --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 
    # 100 clients
    python single/main_FedKNOW.py --epochs=150 --num_users=100 --frac=0.4 --model=ResNet18 --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 
    ```
2. **Result**

    - **The accuracy under 50 clients and 100 clients** (X-axis represents the task and Y-axis represents the accuracy)
    
        <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/bigscale_acc.png" width="50%">
        
    - **The average forgetting rate under 50 clients and 100 clients** (X-axis represents the task and Y-axis represents the average forgetting rate)
        
        <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/bigscale_fr.png" width="50%">
        
### 4.4 Long task sequence
1. **Run**

    ```shell
    # dataset = MiniImageNet + TinyImageNet + cifar100 + FC100, task = 80 ,per_task_class = 5
    python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=ResNet18 --dataset=All --num_classes=400 --task=80 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 
    ```
2. **Result**

    - **The average accuracy under 80 tasks** (X-axis represents the task and Y-axis represents the accuracy)
        
        <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/moretask_acc.png" width="50%">
        
    - **The average forgetting rate under 80 tasks** (X-axis represents the task and Y-axis represents the average forgetting rate)
    
        <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/moretask_fr.png" width="50%">
        
    - **The time under 80 tasks** (X-axis represents the task and Y-axis represents the time on current task)
    
        <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/moretask_time.png" width="50%">

### 4.5 Under different parameter settings
1. **Run**
    ```shell
    # store_rate = 0.05
    python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=ResNet18 --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 --store_rate=0.05
    # store_rate = 0.1
    python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=ResNet18 --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 --store_rate=0.1
    # store_rate = 0.2
    python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=ResNet18 --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 --store_rate=0.2
    ```
2. **Result**

    - **The accuracy under different parameter storage ratios** (X-axis represents the task and Y-axis represents the accuracy)
        
        <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/difporpotion_acc.png" width="50%">
        
    - **The time under different parameter storage ratios** (X-axis represents the task and Y-axis represents the time on current task)
    
        <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/difporpotion_time.png" width="50%">
    
### 4.6 Applicability on different networks
1. **Run**

    ```shell
    # WideResNet50
    python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=WideResNet --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5
    # ResNeXt50
    python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=ResNeXt --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5
    # ResNet152
    python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=ResNet152 --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5
    # SENet18
    python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=SENet --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5
    # MobileNetV2
    python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=MobileNet --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-5
    # ShuffleNetV2
    python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=ShuffleNet --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0005 --optim=Adam --lr_decay=1e-5
    # InceptionV3
    python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=Inception --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0005 --optim=Adam --lr_decay=1e-5
    # DenseNet
    python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=DenseNet --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-5
    ```
2. **Result**

    - **The accuracy on different networks**(X-axis represents the task and Y-axis represents the accuracy)
        ![](https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/moremodel.png)
