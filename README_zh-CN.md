# FedKNOW
[English](README.md) | 简体中文
![在这里插入图片描述](https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/FedKNOW_architecture.png)
## Table of contents
- [1 介绍](#1-介绍)
- [2 代码相关](#2-代码相关)
  * [2.1 代码](#21-代码)
  * [2.2 安装](#22-安装)
- [3 在图像分类中支持的模型](#3-支持的模型)
- [4 实验设置](#4-实验设置)
  * [4.1 任务生成](#41-任务生成)
  * [4.2 超参数的选择](#42-超参数的选择)
- [5 实验细节描述](#5-实验细节描述)
  * [5.1 在不同的工作负载测试](#51-在不同的工作负载测试-models)
  * [5.2 在不同带宽下测试](#52-在不同带宽下测试)
  * [5.3 大规模测试](#53-大规模测试)
  * [5.4 多任务测试](#54-多任务测试)
  * [5.5 参数存储比例测试](#55-参数存储比例测试)
  * [5.6 适用性测试](#56-适用性测试)
## 1 介绍
在持续学习的北京下，FedKNOW针对准确率，时间，通信等等取得不错的效果。目前，在联邦学习的场景下FedKNOW对图像分类的8种网络模型下取得了不错的效果，包括：ResNet，ResNeXt，MobileNet，WideResNet，SENet，ShuffleNet，Inception，DenseNet。
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
    
       更多细节在 `utils/option.py`. 对所有算法的描述在 `scripts/single.sh`.
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
        这里的参数描述和单设备参数描述一致，新添加的`client_id`表示客户端标识符, `ip`表示客户端连接服务器的ip地址，用于服务器确定具体的标识。对于每一个算法的配置信息我们放到了`scripts/multi.sh`。
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
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[6 layer_CNN (NeurIPS'2020)](https://proceedings.neurips.cc/paper/2020/hash/258be18e31c8188555c2ff05b4d542c3-Abstract.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;[Cifar100](http://www.cs.toronto.edu/~kriz/cifar.html)&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; <br>&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;[FC100](https://paperswithcode.com/dataset/fc100) &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;<br>&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp;[CORe50](https://vlomonaco.github.io/core50/index.html#download) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/6_layer_CNN.sh) &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[ResNet (CVPR'2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp;<br>&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[TinyImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;| &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/ResNet.sh) &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[MobileNetV2 (CVPR'2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/MobileNet.sh) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[ResNeXt (CVPR'2017)](https://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/ResNext.sh) &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[InceptionV3(CVPR'2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/inception_v3.py) &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[WideResNet (BMVC'2016)](https://arxiv.org/abs/1605.07146) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php)  &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/WideResNet.sh)&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[ShuffleNetV2 (ECCV'2018)](https://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php)  &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/ShuffleNet.sh)&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[DenseNet(CVPR'2017)](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/DenseNet.sh)&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;|
|&nbsp; &nbsp; &nbsp; &nbsp;&#9745;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;[SENet (CVPR'2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html) &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;[MiniImageNet](https://image-net.org/download.php) &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|&nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;[Demo](scripts/models/SENet.sh)&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;|
## 4 实验设置
### 4.1 任务生成
#### 4.1.1 数据集介绍
- [Cifar100](http://www.cs.toronto.edu/~kriz/cifar.html): Cifar100 数据集共有100个不同类别共计50000个训练样本（每个类有500个）以及10000个测试样本（每个类有100个）。
- [FC100](https://paperswithcode.com/dataset/fc100) : FC100 数据集共有100个不同类别共计50000个训练样本（每个类有500个）以及10000个测试样本（每个类有100个）。
- [CORe50](https://vlomonaco.github.io/core50/index.html#download) :CORe50数据集共有550个不同类别共计165000个训练样本（每个类有300个）以及55000个测试样本（每个类有100个）。
- [MiniImageNet](https://image-net.org/download.php):MiniImageNet 数据集共有100个不同类别共计50000个训练样本（每个类有500个）以及10000个测试样本（每个类有100个）。
- [TinyImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip): MiniImageNet 数据集共有200个不同类别共计100000个训练样本（每个类有500个）以及10000个测试样本（每个类有50个）。

#### 4.1.2 任务划分方法
在构建不同的dataloader前，我们需要将各个数据集拆分成多个任务。我们使用[持续学习数据集拆分方法](https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html)将这些数据集拆分为多个任务。每个任务都有不同类别的数据样本，并分配了一个唯一的任务ID，如下：

- 将Cifar100拆分为10个任务
	```shell
	python dataset/Cifar100.py --task_number=10 --class_number=100
	```
- 将FC100拆分为10个任务
	```shell
	python dataset/FC100.py --task_number=10 --class_number=100
	```
- 将CORe50拆分为11个任务
	```shell
	python dataset/core50.py --task_number=11 --class_number=550
	```
- 将MiniImageNet拆分为10个任务
	```shell
	python dataset/miniimagenet.py --task_number=10 --class_number=100
	```
- 将TinyImageNet拆分为20个任务
	```shell
	python dataset/tinyimagenet.py --task_number=20 --class_number=200
	```
#### 4.1.3 任务的分配方法
在联邦持续设置下，每个客户端都有自己的私有任务序列，因此我们根据[FedRep](http://proceedings.mlr.press/v139/collins21a)方法将每个任务以Non-IID的形式分配给所有客户端。
具体来说，我们将每一个数据集拆分的任务序列分配给所有的客户端。对于每个任务，每个客户端随机选择2-5个类别的数据，从被选中类别中随机获得10%的训练样本和测试样本。
```shell
def noniid(dataset, num_users, shard_per_user, num_classes, dataname, rand_set_all=[]):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    count = 0
    for i in range(len(dataset)):
        if dataname == 'miniimagenet' or dataname == 'FC100' or dataname == 'tinyimagenet':
            label = torch.tensor(dataset.data[i]['label']).item()
        elif dataname == 'Corn50':
            label = torch.tensor(dataset.data['label'][i]).item()
        else:
            label = torch.tensor(dataset.data[i][1]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1

    shard_per_class = int(shard_per_user * num_users / num_classes)
    samples_per_user = int( count/num_users )
    # whether to sample more test samples per user
    if (samples_per_user < 20):
        double = True
    else:
        double = False

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    testb = False
    for i in range(num_users):
        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            if (samples_per_user < 100 and testb):
                rand_set.append(idxs_dict[label][idx])
            else:
                rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    return dict_users, rand_set_all
```
### 4.2 超参数的选择
**选择方法** 

为了确保每种方法都能有效工作，我们使用额外的图像分类数据集SVHN来搜索每种方法的超参数，以避免测试数据泄漏，并确保所有方法的公平性。

**选择指标** 
    
在内存限制(每个客户端4G内存)、时间限制(每个任务运行时间不超过20分钟)下，选择正确率最高的超参数。

**各个超参数描述及搜索空间**
1. 基本超参数

    基本超参数用于保证在各个方法下模型能有足够的时间进行训练和更新，使得模型能够收敛。
	- `Aggregation round`: 每个任务的聚合轮数，搜索空间为[5, 10, 15]。
	- `Local epoch`: 每轮客户端本地训练的次数，搜索空间为[5, 8, 10]。
	- `Learning rate`: 学习率，搜索空间为[0.0005, 0.0008, 0.001, 0.005]。
	- `Learning rate decrease`: 学习率衰减量，搜索空间为[1e-6, 1e-5, 1e-4]
2. 各个方法不同的超参数
    
    每个方法（baseline）都有其独有的超参数，这些参数保证了方法能够正常的工作。对于这些超参数，我们根据其论文中设定值的1/2和2倍为搜索空间的下界以及上界，从中选择3个中间值进行搜索。
	- 基于存储样本的持续学习算法（GEM，BCN，Co2L）
		* `Task sample rate`: 每个任务保存样本的比例，用于计算过去任务的损失，避免灾难性以往，搜索空间为[5%, 10%, 15%]。
	- 基于正则化的持续学习（EWC，AGS-CL，MAS）
		* `Regulation parameter`: 正则化参数，用于固定住部分关键的参数权重，搜索空间为[100, 10000, 40000]。
	- FLCN
		* `Center sample rate`: 服务器存储样本的比例，用于调整正则化参数，搜索空间为[5%, 10%, 15%]。
	- FedWEIT
		* `sparseness parameter`: 损失稀疏参数，用于分开基本参数以及自适应参数，搜索空间为[1e-5, 1e-4, 1e-3]。
		* `weight-decay parameter`: 权重衰减参数，搜索空间为[1e-5, 1e-4, 1e-3]。
		* `past task loss parameter`: 过去任务损失参数，搜索空间为[0.5, 1, 2]。
	- FedKNOW(Our)
		* `Weight rate`: 每个任务存储参数的比例，用作每个任务的任务知识，搜索空间为[5%, 10%, 20%]。
		* `k`: 参与聚合的过去任务梯度数，搜索空间为[5, 10, 20]。

**运行脚本**
```shell
./main_hyperparameters.sh
```

## 5 实验细节描述
### 5.1 在不同的工作负载运行结果
#### 5.1.1 Experiment code
1. 在20个Jetson设备上运行
    
    我们选择了20个拥有不同内存不同计算速度的Jetson设备在5个不同工作负载上进行了测试，包括：8个4G内存的Jetson Nano设备，2个8G内存的Jetson TX2，8个16GB内存的Jetson Xavier NX以及2个32GB的Jetson AGX。
    - **运行服务器：**
        ```shell
        ## 在20个jetson设备上运行
        python multi/server.py --epochs=150 --num_users=20 --frac=0.4 --ip=127.0.0.1:8000
        ```
        **Note：这里的ip=127.0.0.1：8000表示利用本地机器充当服务器，如果有现成的服务器则可以替换为服务器的ip地址。**
    
   - **运行客户端：**
       * 6-layer CNN on Cifar100
            ```shell
           ## 在20个jetson设备上运行
           for ((i=0;i<20;i++));
           do
               python multi/main_FedKNOW.py --client_id=$i --model=6_layerCNN --dataset=cifar100 --num_classes=100 --task=10 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
           done
           ```
       * 6-layer CNN on FC100
           ```shell
           ## 在20个jetson设备上运行
           for ((i=0;i<20;i++));
           do
               python multi/main_FedKNOW.py --client_id=$i --model=6_layerCNN --dataset=FC100 --num_classes=100 --task=10 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
           done
           ```
       * 6-layer CNN on CORe50
           ```shell
           ## 在20个jetson设备上运行
           for ((i=0;i<20;i++));
           do
               python multi/main_FedKNOW.py --client_id=$i --model=6_layerCNN --dataset=CORe50 --num_classes=550 --task=11 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
           done
           ```
       * ResNet18 on MiniImageNet
           ```shell
             ## 在20个jetson设备上运行
           for ((i=0;i<20;i++));
           do
               python multi/main_FedKNOW.py --client_id=$i --model=ResNet --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 --ip=127.0.0.1:8000
           done
           ```
       * ResNet18 on TiniImageNet
           ```shell
           ## 在20个jetson设备上运行
           for ((i=0;i<20;i++));
           do
               python multi/main_FedKNOW.py --client_id=$i --model=ResNet --dataset=TinyImageNet --num_classes=200 --task=20 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 --ip=127.0.0.1:8000
           done
           ```
       **Note:** 服务器和客户端的ip地址请保持一致，127.0.0.1表示在本机上测试。如果有多个设备进行运行的话，则直接在对应的边缘设备上运行相应的代码，同时将其替换为服务器的ip地址。其他baseline的运行说明在`scripts/DifWork`中。

2. 在10个树莓派以及20个Jetson设备上运行
    
    为了加大边缘设备的异构性，我们添加了10个不同内存（2G内存的1个，4G内存的5个，8G内存的4个）的树莓派。相比于利用GPU进行计算的Jetson设备，利用CPU计算的树莓派的计算速度会大大降低，同时由于内存限制，在训练过程中如果不控制存储数据的大小，则会出现内存溢出的现象。
    - **运行服务器：**
        ```shell
        ## 在10个树莓派以及20个jetson设备运行
        python multi/server.py --epochs=150 --num_users=30 --frac=0.4 --ip=127.0.0.1:8000
        ```
        **Note：这里的ip=127.0.0.1：8000表示利用本地机器充当服务器，如果有现成的服务器则可以替换为服务器的ip地址。**
   - **运行客户端：**
       * 6-layer CNN on Cifar100
           ```shell
           ## 在20个jetson设备上运行
           for ((i=0;i<20;i++));
           do
               python multi/main_FedKNOW.py --client_id=$i --model=6_layerCNN --dataset=cifar100 --num_classes=100 --task=10 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
           done
           for ((i=0;i<10;i++));
           do
               python multi/main_FedKNOW.py --client_id=10+$i --model=6_layerCNN --dataset=cifar100 --num_classes=100 --task=10 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000 --gpu=-1
           done
           ```
       * 6-layer CNN on FC100
           ```shell
           ## 在10个树莓派以及20个jetson设备运行
           for ((i=0;i<20;i++));
           do
               python multi/main_FedKNOW.py --client_id=$i --model=6_layerCNN --dataset=FC100 --num_classes=100 --task=10 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
           done
           for ((i=0;i<10;i++));
           do
               python multi/main_FedKNOW.py --client_id=10+$i --model=6_layerCNN --dataset=FC100 --num_classes=100 --task=10 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000 --gpu=-1
           done
           ```
       * 6-layer CNN on CORe50
           ```shell
           ## 在10个树莓派以及20个jetson设备运行
           for ((i=0;i<20;i++));
           do
               python multi/main_FedKNOW.py --client_id=$i --model=6_layerCNN --dataset=CORe50 --num_classes=550 --task=11 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
           done
           for ((i=0;i<10;i++));
           do
               python multi/main_FedKNOW.py --client_id=10+$i --model=6_layerCNN --dataset=CORe50 --num_classes=550 --task=11 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000 --gpu=-1
           done
           ```
           **Note:** 树莓派上只能使用CPU进行运行，保证超参数gpu=-1。

#### 5.2 **Experiment result**
- **The accuracy trend overtime time under different workloads**(X-axis represents the time and Y-axis represents the inference accuracy)
    ![](https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/difworkerloader.png)
### 5.2 在不同带宽下运行结果
#### 5.2.1 Experiment code
- **限制服务器网速：**
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
- **运行客户端：**
    * 6-layer CNN on Cifar100
        ```shell
        for ((i=0;i<20;i++));
        do
            python multi/main_FedKNOW.py --client_id=$i --model=6_layerCNN --dataset=cifar100 --num_classes=100 --task=10 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
        done
        ```
    * 6-layer CNN on FC100
        ```shell
        for ((i=0;i<20;i++));
        do
            python multi/main_FedKNOW.py --client_id=$i --model=6_layerCNN --dataset=FC100 --num_classes=100 --task=10 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
        done
        ```
    * 6-layer CNN on CORe50
        ```shell
        for ((i=0;i<20;i++));
        do
            python multi/main_FedKNOW.py --client_id=$i --model=6_layerCNN --dataset=CORe50 --num_classes=550 --task=11 --alg=FedKNOW --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
        done
        ```
    * ResNet18 on MiniImageNet
        ```shell
        for ((i=0;i<20;i++));
        do
            python multi/main_FedKNOW.py --client_id=$i --model=ResNet18 --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 --ip=127.0.0.1:8000
        done
        ```
    * ResNet18 on TiniImageNet
        ```shell
        for ((i=0;i<20;i++));
        do
            python multi/main_FedKNOW.py --client_id=$i --model=ResNet18 --dataset=TinyImageNet --num_classes=200 --task=20 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 --ip=127.0.0.1:8000
        done
        ```
#### 5.2.2 Experiment result

- **算法在最大带宽为1MB/s下在不同工作负载的通信时间(x axis for dataset and y axis for communication time)**
     
    <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/difctime.png" width="50%">
    
- **算法在不同网络带宽的总通信时间(x axis for bandwidth and y axis for communication time)**
    
    <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/difbandwidth.png" width="50%">
    
### 5.3 大规模测试
#### 5.3.1 Experiment code

```shell
# 50 clients
python single/main_FedKNOW.py --epochs=150 --num_users=50 --frac=0.4 --model=ResNet18 --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 
# 100 clients
python single/main_FedKNOW.py --epochs=150 --num_users=100 --frac=0.4 --model=ResNet18 --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 
```
#### 5.3.2 Experiment result

- **算法在50个客户端以及100个客户端的下的准确率**(x axis for task and y axis for accuracy)
    
    <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/bigscale_acc.png" width="50%">
    
- **算法在50个客户端以及100个客户端的下的平均遗忘率**(x axis for task and y axis for average forgetting rate)
    
    <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/bigscale_fr.png" width="50%">
        
### 5.4 多任务测试
#### 5.4.1 Experiment code

```shell
# dataset = MiniImageNet + TinyImageNet + cifar100 + FC100, task = 80 ,per_task_class = 5
python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=ResNet18 --dataset=All --num_classes=400 --task=80 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 
```
#### 5.4.2 Experiment result

- **算法在80个任务的平均准确率**(x axis for task and y axis for accuracy)
    
    <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/moretask_acc.png" width="50%">
    
- **算法在80个任务的平均遗忘率**(x axis for task and y axis for average forgetting rate)
    
    <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/moretask_fr.png" width="50%">
    
- **算法在80个任务的任务时间**(x axis for task and y axis for current task time)
    
    <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/moretask_time.png" width="50%">

### 5.5 参数存储比例测试
#### 5.5.1 Experiment code
```shell
# store_rate = 0.05
python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=ResNet18 --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 --store_rate=0.05
# store_rate = 0.1
python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=ResNet18 --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 --store_rate=0.1
# store_rate = 0.2
python single/main_FedKNOW.py --epochs=150 --num_users=20 --frac=0.4 --model=ResNet18 --dataset=MiniImageNet --num_classes=100 --task=10 --alg=FedKNOW --lr=0.0008 --optim=SGD --lr_decay=1e-5 --store_rate=0.2
```
#### 5.5.2 Experiment result**

- **算法使用不同存储比例时准确率**(x axis for task and y axis for accuracy)
     
    <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/difporpotion_acc.png" width="50%">
    
- **算法使用不同存储比例时任务时间**(x axis for task and y axis for current task time)
    
    <img src="https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/difporpotion_time.png" width="50%">
    
    
### 5.6 适用性测试
#### 5.6.1 Experiment code

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
#### 5.6.2 Experiment result

- **算法在不同网络模型上的正确率**(x axis for task and y axis for accuracy)
    ![在这里插入图片描述](https://github.com/LINC-BIT/FedKNOW/blob/main/Experiment%20images/moremodel.png)