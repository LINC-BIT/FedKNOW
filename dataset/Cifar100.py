import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets,models
from PIL import Image
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple
from models.inception_v3 import inception_v3
from torch.utils.data import DataLoader
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

class Cifar100Task():
    def __init__(self,root,task_num=1):
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        self.root = root
        self.task_num = task_num
        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
        self.transform_train = data_transform['train']
        self.transform = data_transform['val']
        # self.transform_train = transforms.Compose([
        #     # transforms.RandomCrop(32, padding=4),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std),
        #     # Cutout(n_holes=1, length=16)
        # ])
        # self.transform = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize(mean, std)]
        # )

    def getTaskDataSet(self):
        trainDataset = myCIFAR100(root=self.root, train=True, transform=self.transform_train, download=True, task_num=self.task_num)
        train_task_datasets = [CifarDataset(data, self.transform_train) for data in trainDataset.task_datasets]
        testDataset = myCIFAR100(root=self.root, train=False, transform=self.transform, download=True, task_num=self.task_num)
        test_task_datasets = [CifarDataset(data, self.transform) for data in testDataset.task_datasets]
        return train_task_datasets,test_task_datasets

class CifarDataset(Dataset):
    def __init__(self,data,transform):
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index][0], self.data[index][1]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
class myCIFAR100(datasets.CIFAR100):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            task_num:int = 1
    ) -> None:

        super(myCIFAR100, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()
        self.ldata=[]
        for x in self.data:
            self.ldata.append(x)
        zipped = zip(self.ldata,self.targets)
        self.sort_zipped = sorted(zipped, key=lambda x: x[1])
        self.task_datasets = []
        samples = len(self.data)//task_num
        for i in range(task_num):
            task_dataset = []
            for j in range(samples):
                task_dataset.append(self.sort_zipped[i*samples+j])
            self.task_datasets.append(task_dataset)

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.sort_zipped[index][0], self.sort_zipped[index][1]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

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
        # models.wide_resnet50_2()
        # models.DenseNet()
        # models.mobilenet_v2()
        self.net = inception_v3()
        models.inception_v3()
        # models.resnet152()
        # models.shufflenet_v2_x0_5()
        self.net.load_state_dict(torch.load('../pre_train/inception_v3_google-1a9a5a14.pth'))
        self.net.fc = torch.nn.Linear(self.net.fc.in_features,10)
        self.weight_keys =[]
        for name,para in self.named_parameters():
            temp=[]
            if 'fc' not in name:
                temp.append(name)
                self.weight_keys.append(temp)
    def forward(self,x):
        out = self.net(x)
        return out

def main():
    task = Cifar100Task('../data/cifar100',task_num=10)
    train, test = task.getTaskDataSet()
    train_dataset = train[0]
    val_dataset = test[0]
    # for i in t:
    #     a =i[0]
    # train_dataset = MiniImageDataSet(root_dir='../data/mini-imagenet',
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
                                               batch_size=128,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=2)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=64,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=2)

    net_glob = testmodel()
    net_glob.cuda()
    opt = torch.optim.Adam(net_glob.parameters(), 0.0005)
    ce = torch.nn.CrossEntropyLoss()
    print(net_glob.weight_keys)
    for epoch in range(100):
        net_glob.train()
        for x, y in train_loader:
            x = x.cuda()
            y = y.cuda()
            out,_ = net_glob(x)
            loss = ce(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 1 == 0:
            acc = evaluate(net_glob, val_loader, 'cuda:0')
            print('The epochs:' + str(epoch) + '  the acc:' + str(acc))

if __name__ == "__main__":
    main()