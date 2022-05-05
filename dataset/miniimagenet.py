import os
import json
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms, models
from models.Nets import RepTail
class MiniImageTask():
    def __init__(self,root,json_path,task_num=1,data_transform=None):
        self.root = root
        self.csv_name = {
            'train':"new_train.csv",
            'test': "new_val.csv"
        }
        self.task_num = task_num
        self.json_path = json_path
        self.data_transform = data_transform

    def getTaskDataSet(self):
        trainDataset = MyMiniImageDataSet(root_dir=self.root, csv_name=self.csv_name['train'], json_path = self.json_path,task=self.task_num)
        train_task_datasets = [MiniImageDataset(data, transform=self.data_transform['train']) for data in trainDataset.task_datasets]
        testDataset = MyMiniImageDataSet(root_dir=self.root, csv_name=self.csv_name['test'], json_path = self.json_path,task=self.task_num)
        test_task_datasets = [MiniImageDataset(data, transform=self.data_transform['test']) for data in testDataset.task_datasets]
        return train_task_datasets,test_task_datasets
class MiniImageDataset(Dataset):
    def __init__(self,data,transform):
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int):
        imgpath, target = self.data[item]['image'], self.data[item]['label']

        img = Image.open(imgpath)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
class MyMiniImageDataSet():
    """自定义数据集"""
    def __init__(self,
                 root_dir: str,
                 csv_name: str,
                 json_path: str,
                 task:int):
        images_dir = os.path.join(root_dir, "images")
        self.task=task
        assert os.path.exists(images_dir), "dir:'{}' not found.".format(images_dir)

        assert os.path.exists(json_path), "file:'{}' not found.".format(json_path)
        self.label_dict = json.load(open(json_path, "r"))

        csv_path = os.path.join(root_dir, csv_name)
        assert os.path.exists(csv_path), "file:'{}' not found.".format(csv_path)
        csv_data = pd.read_csv(csv_path)
        self.total_num = csv_data.shape[0]
        self.img_paths = [os.path.join(images_dir, i)for i in csv_data["filename"].values]
        self.img_label = [self.label_dict[i][0] for i in csv_data["label"].values]
        self.labels = set(csv_data["label"].values)
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
class testmodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # SENET
        self.net = models.shufflenet_v2_x0_5()
        # models.DenseNet()
        # models.mobilenet_v2()
        # models.resnet152()
        # models.shufflenet_v2_x0_5()
        self.net.load_state_dict(torch.load('../pre_train/shufflenetv2_x0.5-f707e7126e.pth'))
        self.net.fc = torch.nn.Linear(self.net._stage_out_channels[-1],100)
        self.weight_keys =[]
        for name,para in self.named_parameters():
            temp=[]
            if 'fc' not in name:
                temp.append(name)
                self.weight_keys.append(temp)
    def forward(self,x):
        out = self.net(x)
        return out
if __name__ == '__main__':
    m = MiniImageTask(root='../data/mini-imagenet',json_path="../data/mini-imagenet/classes_name.json",task_num=1)
    t,te =m.getTaskDataSet()
    for i in t:
        a =i[0]
    train_dataset = t[0]
    val_dataset = te[0]
    # train_dataset = MiniImageTask(root_dir='../data/mini-imagenet',
    #                           csv_name="new_train.csv",
    #                           json_path="../data/mini-imagenet/classes_name.json",
    #                           task=10,
    #                           transform=data_transform["train"])
    # val_dataset = MiniImageDataSet(root_dir='../data/mini-imagenet',
    #                         csv_name="new_val.csv",
    #                         json_path="../data/mini-imagenet/classes_name.json",
    #                         task=10,
    #                         transform=data_transform["val"])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=256,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=1,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=256,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=1,
                                             collate_fn=val_dataset.collate_fn)
    net_glob = testmodel()
    net_glob.cuda()
    net_glob.train()
    opt = torch.optim.Adam(net_glob.parameters(), 0.001)
    ce = torch.nn.CrossEntropyLoss()
    for epoch in range(100):
        for x,y in train_loader:
            x = x.cuda()
            y = y.cuda()
            out = net_glob(x)
            loss = ce(out,y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 1 ==0:
            acc = evaluate(net_glob,val_loader,'cuda:0')
            print('The epochs:'+ str(epoch)+'  the acc:'+ str(acc))





