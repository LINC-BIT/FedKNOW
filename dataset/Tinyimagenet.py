import os
import json
from torchvision import models
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from models.Nets import RepTail
import cv2
import numpy as np
class TinyimageTask():
    def __init__(self,root,task_num=1):
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        self.root = root
        self.csv_name = {
            'train':"train.csv",
            'test': "test.csv"
        }
        self.task_num = task_num
        self.data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "test": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
        # self.transform_train = data_transform['train']
        # self.transform = data_transform['val']
        # self.data_transform = {
        #     "train": transforms.Compose([transforms.ToTensor(),
        #                                  transforms.Normalize(mean, std)]),
        #     "test": transforms.Compose([
        #                                transforms.ToTensor(),
        #                                transforms.Normalize(mean, std)])}
    def getTaskDataSet(self):
        trainDataset = MyTinyimageDataSet(root_dir=self.root, csv_name=self.csv_name['train'],name='train',task=self.task_num)
        train_task_datasets = [TinyimageDataSet(data, transform=self.data_transform['train']) for data in trainDataset.task_datasets]
        testDataset = MyTinyimageDataSet(root_dir=self.root, csv_name=self.csv_name['test'],name='val',task=self.task_num)
        test_task_datasets = [TinyimageDataSet(data, transform=self.data_transform['test']) for data in testDataset.task_datasets]
        return train_task_datasets,test_task_datasets
class TinyimageDataSet(Dataset):
    def __init__(self,data,transform):
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int):
        imgpath, target = self.data[item]['image'], self.data[item]['label']

        img = Image.open(imgpath)
        img1 = np.array(img)
        if len(img1.shape) == 2:
            img1 = cv2.imread(imgpath)
            gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img1 = np.zeros_like(img1)
            img1[:, :, 0] = gray
            img1[:, :, 1] = gray
            img1[:, :, 2] = gray
            img = Image.fromarray(img1)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
class MyTinyimageDataSet(Dataset):
    """自定义数据集"""
    def __init__(self,
                 root_dir: str,
                 csv_name: str,
                 task:int,
                 name,
                 transform = None):
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        self.task=task
        csv_path = root_dir+'/'+csv_name
        # csv_path = csv_name
        root_dir = os.path.join(root_dir,name)
        assert os.path.exists(csv_path), "file:'{}' not found.".format(csv_path)
        csv_data = pd.read_csv(csv_path)
        self.total_num = csv_data.shape[0]
        self.img_paths = [os.path.join(root_dir, i)for i in csv_data["dir"].values]
        self.img_label = [i for i in csv_data["label"].values]
        samples = self.total_num//task
        self.task_datasets = []
        for i in range(task):
            task_dataset = []
            for j in range(samples):
                zipped = {}
                zipped['image'] = self.img_paths[i*samples+j]
                zipped['label'] = self.img_label[i*samples+j]
                task_dataset.append(zipped)
            self.task_datasets.append(task_dataset)
class testmodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net_glob = models.resnet18()
        self.net_glob.load_state_dict(torch.load('../pre_train/resnet18.pth'))
        self.net_glob.fc = torch.nn.Linear(self.net_glob.fc.in_features,10)
    def forward(self,x):
        out = self.net_glob(x)
        return out
@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 验证集样本个数
    num_samples = len(data_loader.dataset)

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 等待所有进程计算完毕
    acc = sum_num.item() / num_samples

    return acc
# if __name__ == '__main__':
#     m = TinyimageTask(root='../data/tiny-imagenet-200',task_num=20)
#     train,test =m.getTaskDataSet()
#     train_dataset = train[0]
#     train = train[0]
#     val_dataset = test[0]
#     # for i in t:
#     #     a =i[0]
#     # train_dataset = MiniImageDataSet(root_dir='../data/mini-imagenet',
#     #                           csv_name="new_train.csv",
#     #                           json_path="../data/mini-imagenet/classes_name.json",
#     #                           task=10,
#     #                           transform=data_transform["train"])
#     # val_dataset = MiniImageDataSet(root_dir='../data/mini-imagenet',
#     #                         csv_name="new_val.csv",
#     #                         json_path="../data/mini-imagenet/classes_name.json",
#     #                         task=10,
#     #                         transform=data_transform["val"])
#     train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                batch_size=128,
#                                                shuffle=True,
#                                                pin_memory=True,
#                                                num_workers=1,
#                                                collate_fn=train_dataset.collate_fn)
#
#     val_loader = torch.utils.data.DataLoader(val_dataset,
#                                              batch_size=256,
#                                              shuffle=False,
#                                              pin_memory=True,
#                                              num_workers=1,
#                                              collate_fn=val_dataset.collate_fn)
#     net_glob = testmodel()
#     net_glob.cuda()
#     net_glob.train()
#     opt = torch.optim.SGD(net_glob.parameters(), 0.001,momentum=0.9)
#     ce = torch.nn.CrossEntropyLoss()
#     for epoch in range(100):
#         for x,y in train_loader:
#             x = x.cuda()
#             y = y.cuda()
#             out = net_glob(x)
#             loss = ce(out,y)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#         if epoch % 10 ==0:
#             acc = evaluate(net_glob,val_loader,'cuda:0')
#             print('The epochs:'+ str(epoch)+'  the acc:'+ str(acc))





