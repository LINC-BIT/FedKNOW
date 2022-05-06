import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data
from models.Update import DatasetSplit
from models.test import test_img_local_all
from single.ContinualLearningMethod.APFL import Appr,LongLifeTrain
from torch.utils.data import DataLoader
import time

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or 'MiniImageNet' in args.dataset or 'FC100' in args.dataset or 'CORe50' in args.dataset or 'TinyImageNet' in args.dataset:
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])
    else:
        print('Not this dataset!!')
    write = SummaryWriter('./log/APFL_' + args.dataset+'_'+'round' + str(args.round) + '_frac' + str(args.frac) + '_model_'+args.model)
    # build model
    # net_glob = get_model(args)
    net_glob = get_model(args)
    net_glob.train()
    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    w_glob_keys = []

    # generate list of local models for each user
    net_local_list = []
    w_locals = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict

    # training
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()
    task=-1
    apprs = [Appr(net_glob.to(args.device),100,10, args) for i in range(args.num_users)]
    print(args.round)
    for iter in range(args.epochs):
        if iter % (args.round) == 0:
            task+=1
        w_glob = {}
        fisher_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        w_keys_epoch = w_glob_keys
        times_in = []
        total_len = 0
        tr_dataloaders= None
        for ind, idx in enumerate(idxs_users):
            glob_fisher = None
            start_in = time.time()
            tr_dataloaders = DataLoader(DatasetSplit(dataset_train[task],dict_users_train[idx][:args.m_ft]),batch_size=args.local_bs, shuffle=True)
            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            net_local.load_state_dict(w_local)
            appr = apprs[idx]
            appr.set_model(net_local.to(args.device))
            # appr.set_trData(tr_dataloaders)
            last = iter == args.epochs
            w_local,loss, indd = LongLifeTrain(args,appr,tr_dataloaders,iter,idx)
            loss_locals.append(copy.deepcopy(loss))
            total_len += lens[idx]
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_glob[key] * lens[idx]
                    w_locals[idx][key] = w_local[key]
            else:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    if key in w_glob_keys:
                        w_glob[key] += w_local[key] * lens[idx]
                    else:
                        w_glob[key] += w_local[key] * lens[idx]
                    w_locals[idx][key] = w_local[key]
            # if len(fisher_glob) == 0:
            #     fisher_glob = copy.deepcopy(fisher)
            #     for k, key in enumerate(net_glob.feature_net.state_dict().keys()):
            #         fisher_glob[key] = fisher[key] * lens[idx]
            # else:
            #     for k, key in enumerate(net_glob.feature_net.state_dict().keys()):
            #         fisher_glob[key] += fisher[key] * lens[idx]
            times_in.append(time.time() - start_in)
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)
        # for k in net_glob.feature_net.state_dict().keys():
        #     fisher_glob[k] = torch.div(fisher_glob[k], total_len)
        # for i in range(args.num_users):
        #     apprs[i].set_fisher(fisher_glob)
        w_local = net_glob.state_dict()
        for k in w_glob.keys():
            w_local[k] = w_glob[k]
        if args.epochs != iter:
            net_glob.load_state_dict(w_glob)

        if iter % args.round == args.round-1:
            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
            acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,task,
                                                     w_glob_keys=w_glob_keys, w_locals=w_locals, indd=indd,
                                                     dataset_train=dataset_train, dict_users_train=dict_users_train,
                                                     return_all=False,write=write,apprs = apprs)
            accs.append(acc_test)
            # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
            if iter != args.epochs:
                print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    iter, loss_avg, loss_test, acc_test))
            else:
                # in the final round, we sample all users, and for the algs which learn a single global model, we fine-tune the head for 10 local epochs for fair comparison with FedRep
                print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    loss_avg, loss_test, acc_test))
            if iter >= args.epochs - 10 and iter != args.epochs:
                accs10 += acc_test / 10
            if iter >= args.epochs - 10 and iter != args.epochs:
                accs10_glob += acc_test / 10

            model_save_path = './save/APFL/0.4/accs_APFL_memory_'+str(args.lamb) +str('_') + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) + '_iter' + str(iter) + '_frac_'+str(args.frac)+'.pt'
            torch.save(net_glob.state_dict(), model_save_path)

    # print('Average accuracy final 10 rounds: {}'.format(accs10))
    # if args.alg == 'fedavg' or args.alg == 'prox':
    #     print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
    end = time.time()
    print(end - start)
    print(times)
    print(accs)
    base_dir = './save/APFL/accs_APFL_lambda_' + str(args.lamb) + str(
        '_') + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
        args.shard_per_user) + '_iterFinal' + '_frac_' + str(args.frac) + '_model_' + args.model + '.csv'
    user_save_path = base_dir
    accs = np.array(accs)
    accs = pd.DataFrame(accs, columns=['accs'])
    accs.to_csv(base_dir, index=False)
