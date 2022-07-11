# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/utils/train_utils.py
# credit goes to: Paul Pu Liang

from torchvision import datasets, transforms
from models.Nets import RepTailInception_v3,RepTailResNet18,RepTailResNet152,RepTailWideResNet,RepTailResNext,RepTailMobilenet,RepTailshufflenet,RepTail,RepTailSENet,RepTailDensnet
from utils.sampling import noniid
import os
import json
from dataset.Cifar100 import Cifar100Task
from dataset.miniimagenet import MiniImageTask
from dataset.fc100 import FC100Task
from dataset.core50 import Core50Task
from dataset.Tinyimagenet import TinyimageTask

complex_data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "test": transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

easy_data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])]),
    "test": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])])
}

def get_data(args):
    if args.dataset == 'cifar100':
        if args.model == '6layer_CNN':
            data_transform = easy_data_transform
        else:
            data_transform = complex_data_transform
        cifar100 = Cifar100Task('../data/cifar100',task_num=10,data_transform=data_transform)
        dataset_train,dataset_test = cifar100.getTaskDataSet()
        dict_users_train, rand_set_all = noniid(dataset_train[0], args.num_users, args.shard_per_user, args.num_classes // args.task)
        dict_users_test, rand_set_all = noniid(dataset_test[0], args.num_users, args.shard_per_user, args.num_classes // args.task, rand_set_all=rand_set_all)
    elif args.dataset == 'MiniImageNet':
        Miniimagenet = MiniImageTask(root='../data/mini-imagenet',json_path="data/mini-imagenet/classes_name.json",task_num=10,data_transform = complex_data_transform)
        dataset_train, dataset_test = Miniimagenet.getTaskDataSet()
        dict_users_train, rand_set_all = noniid(dataset_train[0], args.num_users, args.shard_per_user, args.num_classes // args.task,dataname='miniimagenet')
        dict_users_test, rand_set_all = noniid(dataset_test[0], args.num_users, args.shard_per_user, args.num_classes // args.task,
                                               rand_set_all=rand_set_all,dataname='miniimagenet')
    elif args.dataset == 'FC100':
        if args.model == '6layer_CNN':
            data_transform = easy_data_transform
        else:
            data_transform = complex_data_transform
        Fc100 = FC100Task(root='../data/FC100',task_num=10,data_transform=data_transform)
        dataset_train, dataset_test = Fc100.getTaskDataSet()
        dict_users_train, rand_set_all = noniid(dataset_train[0], args.num_users, args.shard_per_user, args.num_classes // args.task,
                                                dataname='FC100')
        dict_users_test, rand_set_all = noniid(dataset_test[0], args.num_users, args.shard_per_user, args.num_classes // args.task,
                                               rand_set_all=rand_set_all, dataname='FC100')
    elif args.dataset == 'CORe50':
        if args.model == '6layer_CNN':
            data_transform = easy_data_transform
        else:
            data_transform = complex_data_transform
        CORe50 = Core50Task(root='../data', task_num=11,data_transform = data_transform)
        dataset_train, dataset_test = CORe50.getTaskDataSet()
        dict_users_train, rand_set_all = noniid(dataset_train[0], args.num_users, args.shard_per_user, args.num_classes // args.task,
                                                dataname='Corn50')
        dict_users_test, rand_set_all = noniid(dataset_test[0], args.num_users, args.shard_per_user, args.num_classes // args.task,
                                               rand_set_all=rand_set_all, dataname='Corn50')
    elif args.dataset == 'TinyImageNet':
        Tinyimagenet = TinyimageTask(root='../data/tiny-imagenet-200',task_num=20,data_transform = easy_data_transform)
        dataset_train, dataset_test = Tinyimagenet.getTaskDataSet()
        dict_users_train, rand_set_all = noniid(dataset_train[0], args.num_users, args.shard_per_user, args.num_classes // args.task,
                                                dataname='tinyimagenet')
        dict_users_test, rand_set_all = noniid(dataset_test[0], args.num_users, args.shard_per_user, args.num_classes // args.task,
                                               rand_set_all=rand_set_all, dataname='tinyimagenet')
    else:
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test, dict_users_train, dict_users_test

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(train_data.keys())

    return clients, groups, train_data, test_data


def get_model(args):
    net_glob = None
    if args.model == '6layer_CNN' :
        if 'cifar100' in args.dataset or 'FC100' in args.dataset or 'CORe50' in args.dataset:
            image_size = [3,32,32]
        else:
            image_size = [3, 224, 224]
        net_glob = RepTail(image_size,output=args.num_classes,nc_per_task=args.num_classes // args.task).to(args.device)
    elif args.model == 'ResNet18' :
        net_glob = RepTailResNet18(output=args.num_classes,nc_per_task=args.num_classes // args.task).to(args.device)
    elif args.model == 'ResNet152' :
        net_glob = RepTailResNet152(output=args.num_classes,nc_per_task=args.num_classes // args.task).to(args.device)
    elif args.model == 'Inception' :
        net_glob = RepTailInception_v3(output=args.num_classes,nc_per_task=args.num_classes // args.task).to(args.device)
    elif args.model == 'WideResNet' :
        net_glob = RepTailWideResNet(output=args.num_classes,nc_per_task=args.num_classes // args.task).to(args.device)
    elif args.model == 'ResNeXt' :
        net_glob = RepTailResNext(output=args.num_classes,nc_per_task=args.num_classes // args.task).to(args.device)
    elif args.model == 'MobileNet' :
        net_glob = RepTailMobilenet(output=args.num_classes,nc_per_task=args.num_classes // args.task).to(args.device)
    elif args.model == 'ShuffleNet' :
        net_glob = RepTailshufflenet(output=args.num_classes,nc_per_task=args.num_classes // args.task).to(args.device)
    elif args.model == 'DenseNet' :
        net_glob = RepTailDensnet(output=args.num_classes,nc_per_task=args.num_classes // args.task).to(args.device)
    elif args.model == 'SENet' :
        net_glob = RepTailSENet(output=args.num_classes,nc_per_task=args.num_classes // args.task).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    return net_glob
