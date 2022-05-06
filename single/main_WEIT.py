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
from models.test import test_img_local_all, test_img_local_all_WEIT
from single.ContinualLearningMethod.WEIT import Appr,LongLifeTrain
from models.Nets import WEITResNet
from torch.utils.data import DataLoader
import time

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.wd = 1e-4
    args.lambda_l1 = 1e-3
    args.lambda_l2 = 1
    args.lambda_mask = 0
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist' or 'miniimagenet' in args.dataset or 'FC100' in args.dataset or 'Corn50' in args.dataset:
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])

    print(args.alg)
    write = SummaryWriter('./log/WEIT_' + args.dataset+'_'+'round' + str(args.round) + '_frac' + str(args.frac) + '_model_'+args.model)
    # build model
    # net_glob = get_model(args)
    net_glob = WEITResNet()
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
    apprs = [Appr(copy.deepcopy(net_glob).cuda(), None,lr=args.lr, nepochs=args.local_ep, args=args,num_classes=args.num_classes) for i in range(args.num_users)]
    print(args.round)
    from_kb =[]
    for name,para in net_glob.named_parameters():
        if 'aw' in name:
            shape = np.concatenate([para.shape, [int(round(args.num_users * args.frac))]], axis=0)
            from_kb_l = np.zeros(shape)
            from_kb_l = torch.from_numpy(from_kb_l)
            from_kb.append(from_kb_l)
    w_glob=[]
    for iter in range(args.epochs):
        if iter % (args.round) == 0:
            task+=1
        w_agg=w_glob
        w_glob = []
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        w_keys_epoch = w_glob_keys
        times_in = []
        total_len = 0
        tr_dataloaders= None
        if iter % args.round == 0:
            for i in range(args.num_users):
                apprs[i].model.set_knowledge(task,from_kb)
        for ind, idx in enumerate(idxs_users):
            glob_fisher = None
            start_in = time.time()
            tr_dataloaders = DataLoader(DatasetSplit(dataset_train[task],dict_users_train[idx][:args.m_ft]),batch_size=args.local_bs, shuffle=True)
            w_local = []
            appr = apprs[idx]
            appr.set_sw(w_agg)
            appr.set_trData(tr_dataloaders)
            last = iter == args.epochs
            # if 'femnist' in args.dataset or 'sent140' in args.dataset:
            #     w_local, loss, indd = local.train(net=net_local.to(args.device), ind=idx, idx=clients[idx],
            #                                       w_glob_keys=w_glob_keys, lr=args.lr, last=last)
            # else:
            w_local, aws,loss, indd = LongLifeTrain(args,appr,iter,from_kb,idx)
            if iter % args.round == args.round -1:
                from_kb = []
                for aw in aws:
                    shape = np.concatenate([aw.shape, [int(round(args.num_users * args.frac))]], axis=0)
                    from_kb_l = np.zeros(shape)
                    if len(shape) == 5:
                        from_kb_l[:, :, :, :, ind] = aw.cpu().detach().numpy()
                    else:
                        from_kb_l[:, :, ind] = aw.cpu().detach().numpy()
                    from_kb_l = torch.from_numpy(from_kb_l)
                    from_kb.append(from_kb_l)
            loss_locals.append(copy.deepcopy(loss))
            total_len += lens[idx]
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for i in range(len(w_glob)):
                    w_glob[i] = w_glob[i] * lens[idx]
            else:
                for i in range(len(w_glob)):
                    w_glob[i] += w_local[i]*lens[idx]
            times_in.append(time.time() - start_in)
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # get weighted average for global weights
        for i in range(len(w_glob)):
            w_glob[i] = torch.div(w_glob[i], total_len)
        if iter % args.round == args.round-1:
            for i in range(args.num_users):

                if len(apprs[i].pre_weight['aw']) < task+1:
                    print("client " + str(i) + " not train")
                    tr_dataloaders = DataLoader(DatasetSplit(dataset_train[task], dict_users_train[i][:args.m_ft]),
                                                batch_size=args.local_bs, shuffle=True)
                    apprs[i].set_sw(w_agg)
                    apprs[i].set_trData(tr_dataloaders)
                    LongLifeTrain(args, apprs[i], iter, from_kb, i)

            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
            acc_test, loss_test = test_img_local_all_WEIT(apprs, args, dataset_test, dict_users_test,task,
                                                     w_glob_keys=w_glob_keys, w_locals=w_locals, indd=indd,
                                                     dataset_train=dataset_train, dict_users_train=dict_users_train,
                                                     return_all=False,write=write,num_classes=args.num_classes)
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


    # print('Average accuracy final 10 rounds: {}'.format(accs10))
    # if args.alg == 'fedavg' or args.alg == 'prox':
    #     print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
    end = time.time()
    print(end - start)
    print(times)
    print(accs)
    base_dir = './save/WEIT/accs_WEIT_lambda_'+str(args.lamb) +str('_') + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) + '_iterFinal' + '_frac_'+str(args.frac)+ '_model_'+args.model+'.csv'
    user_save_path = base_dir
    accs = np.array(accs)
    accs = pd.DataFrame(accs, columns=['accs'])
    accs.to_csv(base_dir, index=False)
