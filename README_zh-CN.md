# FedKNOW
[English](README.md) | 简体中文
![在这里插入图片描述](https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/FedKNOW_architecture.png)
## Table of contents
- [1 介绍](#1-介绍)
- [2 代码相关](#2-代码相关)
  * [2.1 代码](#21-代码)
  * [2.2 安装](#22-安装)
- [3 在图像分类中支持的模型](#3-支持的模型)
- [4 实验细节描述](#4-实验细节描述)
  * [4.1 在不同的工作负载测试](#41-在不同的工作负载测试-models)
  * [4.2 在不同带宽下测试](#42-在不同带宽下测试)
  * [4.3 大规模测试](#43-大规模测试)
  * [4.4 多任务测试](#44-多任务测试)
  * [4.5 参数存储比例测试](#45-参数存储比例测试)
  * [4.6 适用性测试](#46-适用性测试)
## 1 介绍
目前，在联邦学习的场景下FedKNOW对图像分类的8种网络模型下取得了不错的效果，包括：ResNet，ResNeXt，MobileNet，WideResNet，SENet，ShuffleNet，Inception，DenseNet。
- [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) : 在图像分类中，由多个卷积层和池化层进行提取图片信息。但随着网络的深度增加，深度神经网络出现梯度消失(爆炸)问题以及网络退化。对于梯度消失(爆炸)的问题，ResNet添加了BatchNorm层进行约束，而对于网络退化，ResNet则建立了残差网络结果，将每一层的输入添加映射到下一层的输入中。
- [Inception](https://ieeexplore.ieee.org/document/7780677)：卷积网络为了提取高维特征，一般方式进行更深层卷积，但是随之带来网络变大的问题。Inception模块提出可以使网络变宽，在保证模型质量的前提下，减少参数个数，提取高维特征。在Inception结构，首先通过1x1卷积来降低通道数把信息聚集一下，再进行不同尺度的特征提取以及池化，得到多个尺度的信息，最后将特征进行叠加输出。
- [RepNeXt](https://arxiv.org/abs/1611.05431)：将Inception和ResNet进行结合。首先对Inception的分解进行简化，使每一个分解结构都具有相同的拓扑形状，之后按照ResNet的方式搭建网络。
- [WideResNet](http://www.bmva.org/bmvc/2016/papers/paper087/index.html)：对ResNet的残差层拓宽了结构，从增加网络宽度角度改善ResNet，同时为了减少增加的参数量，WideResNet利用Dropout来进行正则化。
- [MobileNet](https://arxiv.org/abs/1801.04381)： MobileNet是一个轻量级卷积神经网络，它进行卷积的参数比标准卷积要少很多，其基本单元为深度级可分离卷积。
- [SENet](https://ieeexplore.ieee.org/document/341010)：SENet着手于优化维度上的特征，通过引入注意力机制，增加少量参数，使模型可以更好地获取不同度上的特征。在SENet中，对每一个卷积产生的输出进行压缩-激活操作，该操作能判断每一个通道上重要程度，从而优化提取特征。
- [ShuffleNet](https://arxiv.org/abs/1807.11164)：ShuffleNet的设计目标也是如何利用有限的计算资源来达到最好的模型精度，通过使用pointwise group convolution和channel shuffle两种操作来进行解决。ShuffleNet基本单元由ResNet出发，将ResNet中1x1的卷积层替换为group convolution，同时添加channel shuffle操作，用于降低参数量。
- [DenseNet](https://arxiv.org/pdf/1707.06990.pdf)：DenseNet模型，它的基本思路与ResNet一致，但是它建立的是前面所有层与后面层的密集连接，通过特征在channel上的连接来实现特征重用。
## 2 代码和安装
### 2.1 代码
- **单设备运行**

    FedKNOW 根据下面的命令进行运行：
    ```shell
    python single/main_FedKNOW.py --alg fedknow --dataset [dataset] --model [mdoel]
    --num_users [num_users]  --shard_per_user [shard_per_user] --frac [frac] 
    --local_bs [local_bs] --lr [lr] --task [task] --epoch [epoch]  --local_ep 
    [local_ep] --local_local_ep [local_local_ep] --store_rate [store_rate] 
    --select_grad_num [select_grad_num]--gpu [gpu]
    ```
    参数解释：
    - `alg`： 需要跑的算法，例如：`FedKNOW`,`FedRep`,`GEM`等
    - `dataset` : 数据集，例如：`cifar100`,`FC100`,`CORe50`,`MiniImagenet`, `Tinyimagenet`
    - `model`: 网络模型，例如：`6-Layers CNN`, `ResNet18`
    - `num_users` : 客户端数量
    - `shard_per_user`: 每个客户端拥有的类
    - `frac`：每一轮参与训练的客户端
    - `local_bs`：每一个客户端的batch_size
    - `lr`：学习率
    - `task`：任务数
    - `epochs`: 客户端和服务器通信的总次数
    - `local_ep`：本地客户端迭代数
    - `local_local_ep`：本地客户端更新本地参数的迭代数
    - `store_rate`: FedKNOW中选择存储参数的比例
    - `select_grad_num`: FedKNOW中用于计算旧梯度的数目
    - `gpu`：GPU ID
- **多设备运行**
    1. 控制传输网速
    
        ```shell
        sudo wondershaper [网卡名] [下载速度] [上传速度]
        ```
    2. 运行服务器
        ```shell
        python multi/server.py --num_users [num_users] --frac [frac] --ip [ip]
        ```
    3. 运行客户端
        ```shell
        python multi/main_FedKNOW.py --client_id [client_id] --alg [alg]
        --dataset [dataset] --model[mdoel]  --shard_per_user [shard_per_user] 
        --local_bs [local_bs] --lr [lr] --task [task] --epoch [epoch]  --local_ep 
        [local_ep] --local_local_ep [local_local_ep]  --store_rate [store_rate] 
        --select_grad_num [select_grad_num] --gpu [gpu] --ip [ip]
        ```
        这里的参数描述和单设备参数描述一致，新添加的`client_id`表示客户端标识符, `ip`表示客户端连接服务器的ip地址，用于服务器确定具体的标识。对于每一个算法的配置信息我们放到了`multi/scripts/`。
        完整的参数信息解释在`utils/option.py`。对于每一个算法的详细配置我们放到了`single/scripts/`下。
### 2.2 安装
**相关准备**
- Linux and Windows 
- Python 3.6+
- PyTorch 1.9+
- CUDA 10.2+ 



**准备虚拟环境**
1. 准备conda环境并进行激活.
	```shell
	conda create -n FedKNOW python=3.7
	conda active FedKNOW
	```
2. 在[官网](https://pytorch.org/)安装对应版本的pytorch
![image](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ec360791671f4a4ab322eb4e71cc9e62~tplv-k3u1fbpfcp-zoom-1.image)
可以在终端直接使用官方提供的code
   **Note:  安装前请确保电脑上是否有显卡且显卡算力和pytorch版本匹配**
3.  安装FedKNOW所需要的包
	```shell
	git clone https://github.com/LINC-BIT/FedKNOW.git
	pip install -r requirements.txt
	```
## 3 支持的模型和数据集
||Model Name|Data|Script|
|--|--|--|--|
|&#9745;|[6 layer_CNN (NeurIPS'2020)](https://proceedings.neurips.cc/paper/2020/hash/258be18e31c8188555c2ff05b4d542c3-Abstract.html)|[Cifar100](http://www.cs.toronto.edu/~kriz/cifar.html) <br>[FC100](https://paperswithcode.com/dataset/fc100) <br>[CORe50](https://vlomonaco.github.io/core50/index.html#download)|[Demo](models/Nets.py)|
|&#9745;|[ResNet (CVPR'2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)|[MiniImageNet](https://image-net.org/download.php) <br>[TinyImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip)|[Demo](models/ResNet.py)|
|&#9745;|[MobileNetV2 (CVPR'2018)](https://arxiv.org/abs/1801.04381)|[MiniImageNet](https://image-net.org/download.php)|[Demo](models/mobilenet.py)|
|&#9745;|[ResNeXt (CVPR'2017)](https://arxiv.org/abs/1611.05431)|[MiniImageNet](https://image-net.org/download.php)|[Demo](models/ResNet.py)|
|&#9745;|[InceptionV3(CVPR'2016)](https://ieeexplore.ieee.org/document/7780677/)|[MiniImageNet](https://image-net.org/download.php)|[Demo](models/inception_v3.py)|
|&#9745;|[WideResNet (BMVC'2016)](https://dx.doi.org/10.5244/C.30.87)|[MiniImageNet](https://image-net.org/download.php)|[Demo](models/ResNet.py)|
|&#9745;|[ShuffleNetV2 (ECCV'2018)](https://arxiv.org/abs/1807.11164)|[MiniImageNet](https://image-net.org/download.php)|[Demo](models/shufflenetv2.py)|
|&#9745;|[DenseNet](https://arxiv.org/pdf/1707.06990.pdf)|[MiniImageNet](https://image-net.org/download.php)|[Demo](models/Densenet.py)|
|&#9745;|[SENet (CVPR'2018)](https://ieeexplore.ieee.org/document/341010)|[MiniImageNet](https://image-net.org/download.php)|[Demo](models/SENet.py)|
## 4 实验细节描述
### 4.1 在不同的工作负载运行结果

1. **Experiment code**
    **运行服务器：**
    
    ```shell
    python multi/server.py --epochs=150 --num_users=20 --frac=0.4 --ip=127.0.0.1:8000
    ```
    **运行客户端：**
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
    **Note:** 服务器和客户端的ip地址请保持一致，127.0.0.1表示在本机上测试，如果有多个设备进行运行的话，将其替换为服务器的ip地址。其他baseline的运行说明在`scripts/DifWork`中。
2. **Experiment result**

    - **不同算法在不同工作负载上的运行时间和准确率**(x axis for time and y axis for inference accuracy)
    ![在这里插入图片描述](https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/difworkerloader.png)
### 4.2 在不同带宽下运行结果
1. **Experiment code**

    **限制服务器网速：**
    ```shell
    sudo wondershaper [网卡名] 1000 1000 # 1000表示为最大速度为1000KB/s, 实际中网络会存在波动，可以调整上限值。
    ```
    **查看运行过程中服务器网络情况：**
    ```shell
    sudo nload -m
    ```
    
    **运行服务器：**
    ```shell
    python multi/server.py --epochs=150 --num_users=20 --frac=0.4 --ip=127.0.0.1:8000
    ```
    **运行客户端：**
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
2. **Experiment result**

    - **算法在最大带宽为1MB/s下在不同工作负载的通信时间(x axis for dataset and y axis for communication time)**
         
        <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/difctime.png" width="50%">
        
    - **算法在不同网络带宽的总通信时间(x axis for bandwidth and y axis for communication time)**
        
        <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/difbandwidth.png" width="50%">
        
### 4.3 大规模测试
1. **Experiment code**

    ```shell
    # 50 clients
    python single/main_FedKNOW.py --epochs=150 --num_users=50 --frac=0.4 --model=ResNet18 --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 
    # 100 clients
    python single/main_FedKNOW.py --epochs=150 --num_users=100 --frac=0.4 --model=ResNet18 --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 
    ```
2. **Experiment result**

    - **算法在50个客户端以及100个客户端的下的准确率**(x axis for task and y axis for accuracy)
        
        <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/bigscale_acc.png" width="50%">
        
    - **算法在50个客户端以及100个客户端的下的平均遗忘率**(x axis for task and y axis for average forgetting rate)
        
        <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/bigscale_fr.png" width="50%">
        
### 4.4 多任务测试
1. **Experiment code**

    ```shell
    # dataset = MiniImageNet + TinyImageNet + cifar100 + FC100, task = 80 ,per_task_class = 5
    python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=ResNet18 --dataset=All --num_classes=400 --task=80 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 
    ```
2. **Experiment result**

    - **算法在80个任务的平均准确率**(x axis for task and y axis for accuracy)
        
        <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/moretask_acc.png" width="50%">
        
    - **算法在80个任务的平均遗忘率**(x axis for task and y axis for average forgetting rate)
        
        <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/moretask_fr.png" width="50%">
        
    - **算法在80个任务的任务时间**(x axis for task and y axis for current task time)
        
        <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/moretask_time.png" width="50%">

### 4.5 参数存储比例测试
1. **Experiment code**
    ```shell
    # store_rate = 0.05
    python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=ResNet18 --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 --store_rate=0.05
    # store_rate = 0.1
    python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=ResNet18 --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 --store_rate=0.1
    # store_rate = 0.2
    python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=ResNet18 --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 --store_rate=0.2
    ```
2. **Experiment result**

    - **算法使用不同存储比例时准确率**(x axis for task and y axis for accuracy)
         
        <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/difporpotion_acc.png" width="50%">
        
    - **算法使用不同存储比例时任务时间**(x axis for task and y axis for current task time)
        
        <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/difporpotion_time.png" width="50%">
    
    
### 4.6 适用性测试
1. **Experiment code**

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
2. **Experiment result**

    - **算法在不同网络模型上的正确率**(x axis for task and y axis for accuracy)
        ![在这里插入图片描述](https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/moremodel.png)