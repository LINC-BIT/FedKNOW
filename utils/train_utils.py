# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/utils/train_utils.py
# credit goes to: Paul Pu Liang

from torchvision import datasets, transforms
from models.Nets import CNNCifar, CNNCifar100, RNNSent, MLP
from utils.sampling import noniid
import os
import json
from dataset.Cifar100 import Cifar100Task
from dataset.miniimagenet import MiniImageTask
from dataset.fc100 import FC100Task
from dataset.core50 import Core50Task
from dataset.Tinyimagenet import TinyimageTask
trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])


def get_data(args):
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all, testb=True)
    elif args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
        dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all)
    elif args.dataset == 'cifar100':
        cifar100 = Cifar100Task('data/cifar100',task_num=10)
        # dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        # dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
        dataset_train,dataset_test = cifar100.getTaskDataSet()
        # dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user, args.num_classes)
        # dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, args.num_classes,
        #                                        rand_set_all=rand_set_all)
        # for dataset_train,dataset_test in zip(dataset_trains,dataset_tests):
        dict_users_train, rand_set_all = noniid(dataset_train[0], args.num_users, args.shard_per_user, args.num_classes)
        dict_users_test, rand_set_all = noniid(dataset_test[0], args.num_users, args.shard_per_user, args.num_classes, rand_set_all=rand_set_all)
    elif args.dataset == 'miniimagenet':
        Miniimagenet = MiniImageTask(root='data/mini-imagenet',json_path="data/mini-imagenet/classes_name.json",task_num=10)
        dataset_train, dataset_test = Miniimagenet.getTaskDataSet()
        dict_users_train, rand_set_all = noniid(dataset_train[0], args.num_users, args.shard_per_user, args.num_classes,dataname='miniimagenet')
        dict_users_test, rand_set_all = noniid(dataset_test[0], args.num_users, args.shard_per_user, args.num_classes,
                                               rand_set_all=rand_set_all,dataname='miniimagenet')
    elif args.dataset == 'FC100':
        Fc100 = FC100Task(root='data/FC100',task_num=10)
        dataset_train, dataset_test = Fc100.getTaskDataSet()
        dict_users_train, rand_set_all = noniid(dataset_train[0], args.num_users, args.shard_per_user, args.num_classes,
                                                dataname='FC100')
        dict_users_test, rand_set_all = noniid(dataset_test[0], args.num_users, args.shard_per_user, args.num_classes,
                                               rand_set_all=rand_set_all, dataname='FC100')
    elif args.dataset == 'Corn50':
        Corn50 = Core50Task(root='data', task_num=11)
        dataset_train, dataset_test = Corn50.getTaskDataSet()
        dict_users_train, rand_set_all = noniid(dataset_train[0], args.num_users, args.shard_per_user, args.num_classes,
                                                dataname='Corn50')
        dict_users_test, rand_set_all = noniid(dataset_test[0], args.num_users, args.shard_per_user, args.num_classes,
                                               rand_set_all=rand_set_all, dataname='Corn50')
    elif args.dataset == 'tinyimagenet':
        Tinyimagenet = TinyimageTask(root='data/tiny-imagenet-200',task_num=20)
        dataset_train, dataset_test = Tinyimagenet.getTaskDataSet()
        dict_users_train, rand_set_all = noniid(dataset_train[0], args.num_users, args.shard_per_user, args.num_classes,
                                                dataname='tinyimagenet')
        dict_users_test, rand_set_all = noniid(dataset_test[0], args.num_users, args.shard_per_user, args.num_classes,
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
    if args.model == 'cnn' and 'cifar100' in args.dataset:
        net_glob = CNNCifar100(args=args).to(args.device)
    elif args.model == 'cnn' and 'cifar10' in args.dataset:
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'mlp' and 'mnist' in args.dataset:
        net_glob = MLP(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
    elif args.model == 'cnn' and 'femnist' in args.dataset:
        net_glob = CNN_FEMNIST(args=args).to(args.device)
    elif args.model == 'mlp' and 'cifar' in args.dataset:
        net_glob = MLP(dim_in=3072, dim_hidden=512, dim_out=args.num_classes).to(args.device)
    elif 'sent140' in args.dataset:
        net_glob = model = RNNSent(args,'LSTM', 2, 25, 128, 1, 0.5, tie_weights=False).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    return net_glob
