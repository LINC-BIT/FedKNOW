import copy
import numpy as np
import torch
from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import DatasetSplit
from multi.ContinualLearningMethod.FedKNOW import Appr,LongLifeTest,LongLifeTrain
from torch.utils.data import DataLoader
import time
from models.Packnet import PackNet
import flwr as fl
from collections import OrderedDict
import datetime
class FedKNOWClient(fl.client.NumPyClient):
    def __init__(self,appr,args):
        self.appr= appr
        self.args = args
        self.curTask = 0
    def get_parameters(self):
        net = appr.get_featurenet()
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        net = appr.get_featurenet()
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        train_round = config['round']

        begintime = datetime.datetime.now()
        print('cur round{} begin training ,time is {}'.format(train_round,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        self.set_parameters(parameters)
        loss, totalnum = LongLifeTrain(self.args,appr,train_round,None,args.client_id)
        endtime =time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print('cur round {} end training ,time is {}'.format(train_round, endtime))
        return self.get_parameters(), totalnum, {}

    def evaluate(self, parameters, config):
        print('eval:')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        test_round = config['round']
        self.set_parameters(parameters)
        loss, accuracy,totalnum = LongLifeTest(args, appr, test_round)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        return float(accuracy), totalnum, {"accuracy": float(accuracy)}
if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist' or 'miniimagenet' in args.dataset or 'FC100' in args.dataset or 'Corn50' in args.dataset:
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])

    net_glob = get_model(args)
    net_glob.train()

    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    net_keys = [net_glob.state_dict().keys()]

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    print(total_num_layers)
    print(net_keys)

    # generate list of local models for each user

    # training
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    start = time.time()
    task=-1
    appr = Appr(copy.deepcopy(net_glob),PackNet(args.task,local_ep=args.local_ep,local_rep_ep=args.local_local_ep,device=args.device),copy.deepcopy(net_glob), None,lr=args.lr, nepochs=args.local_ep, args=args)
    for i in range(args.task):
        tr_dataloaders = DataLoader(DatasetSplit(dataset_train[i], dict_users_train[args.client_id]),
                                    batch_size=args.local_bs, shuffle=True, num_workers=0)
        te_dataloader = DataLoader(DatasetSplit(dataset_test[i], dict_users_test[args.client_id]), batch_size=args.local_test_bs, shuffle=False,num_workers=0)
        appr.traindataloaders.append(tr_dataloaders)
        appr.testdataloaders.append(te_dataloader)
    client = FedKNOWClient(appr,args)
    fl.client.start_numpy_client(args.ip, client=client)