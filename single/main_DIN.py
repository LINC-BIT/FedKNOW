import copy

import numpy as np

import torch

from torch.utils.tensorboard import SummaryWriter

from utils.options import args_parser

from ContinualLearningMethod.DIN import Appr, LongLifeTrain


from torch.utils.data import DataLoader
import time


import os


from dataset.Amazon_Book import DINDataSet,split_data
import pickle as pk
from models.DIN import DIN


def eval_test(apprs,te_dataloader,write=None,round=None):
    print('test begin' + '*' * 100)
    num_idxxs = len(apprs)
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    for idx in range(num_idxxs):

        appr = apprs[idx]
        dataloder = te_dataloader[idx]
        loss,acc = appr.eval(dataloder)


        acc_test_local[idx] = acc
        loss_test_local[idx] = loss
    if write is not None:
        write.add_scalar('task_finish_and _agg', sum(acc_test_local) / num_idxxs, round)


    return sum(acc_test_local) / num_idxxs, sum(loss_test_local) / num_idxxs






if __name__ == '__main__':
    # parse args
    args = args_parser()

    args.model = 'DIN'
    args.dataset = 'book'

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICE'] = f'{args.gpu}'  # compatible to cuda()

    MAX_LEN = 100
    EMBEDDING_DIM = 18


    # Adam
    LR = 1e-3
    BETA1 = 0.5
    BETA2 = 0.99

    # Train
    BATCH_SIZE = 128
    EPOCH_TIME = 20
    TEST_ITER = 1000

    user_map = pk.load(open('../data/Amazon_Book/uid_voc.pkl', 'rb'));
    n_uid = len(user_map)
    material_map = pk.load(open('../data/Amazon_Book/mid_voc.pkl', 'rb'));
    n_mid = len(material_map)
    category_map = pk.load(open('../data/Amazon_Book/cat_voc.pkl', 'rb'));
    n_cat = len(category_map)

    train_file = '../data/Amazon_Book/local_train_splitByUser'
    test_file = '../data/Amazon_Book/local_test_splitByUser'
    split_data(args.num_users,train_file,test_file)

    tr_dataloader = []
    te_dataloader = []
    tr_data_path = '../data/Amazon_Book/data_' + str(args.num_users) + 'clients/train_data_'
    te_data_path = '../data/Amazon_Book/data_' + str(args.num_users) + 'clients/test_data_'
    for i in range(args.num_users):
        tr_set = DINDataSet(tr_data_path+str(i),user_map,material_map,category_map,MAX_LEN)
        tr_dataloader.append(DataLoader(tr_set,batch_size=BATCH_SIZE,shuffle=True))
        te_set = DINDataSet(te_data_path+str(i),user_map,material_map,category_map,MAX_LEN)
        te_dataloader.append(DataLoader(te_set,batch_size=BATCH_SIZE,shuffle=False))

    write = SummaryWriter(
        f'./log/explogs-{args.dataset}/{args.alg}/' + args.dataset + '_' + args.model + '_' + 'round' + str(
            args.round) + "_epoch_" + str(args.local_ep) + '_frac' + str(args.frac) + '_' + str(args.seed)
        + "_" + str(int(time.time()))[-4:])

    net_glob = DIN(n_uid,n_mid,n_cat,EMBEDDING_DIM).to(args.device)
    net_glob.train()
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
    #task = -1
    # apprs = [Appr(net_glob.to(args.device),3*32*32,100,10, args) for i in range(args.num_users)]
    apprs = [Appr(copy.deepcopy(net_glob).to(args.device), tr_dataloader[i], lr=args.lr, nepochs=args.local_ep, args=args) for i in
            range(args.num_users)]

    w_globals = []
    for iter in range(args.epochs):
        # if iter % (args.round) == 0:
        #     task += 1
        #     w_globals = []
        w_glob = {}
        loss_locals = []

        m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users

        # if iter % (args.round) == args.round - 1:
        #     print("*"*100)
        #     print("Last Train")
        #     idxs_users = [i for i in range(args.num_users)]
        # else:
        all_users = [i for i in range(args.num_users)]
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(idxs_users)
        times_in = []
        total_len = 0
        #tr_dataloaders = None
        all_local_models = []

        Client_tr_dataloaders = []

        for ind, idx in enumerate(idxs_users):
            start_in = time.time()

            # tr_dataloaders = DataLoader(
            #     DatasetSplit(dataset_train[client_task[idx][task]], dict_users_train[idx][:args.m_ft],
            #                  tran_task=[task, client_task[idx][task]]), batch_size=args.local_bs, shuffle=True)
            # tr_dataloaders = DataLoader(DatasetSplit(dataset_train[client_task[idx][task]],dict_users_train[idx],tran_task=[task,client_task[idx][task]]),batch_size=args.local_bs, shuffle=True)
            #Client_tr_dataloaders.append(tr_dataloaders)
            # if args.epochs == iter:
            #     local = LocalUpdate(args=args, dataset=dataset_train[task], idxs=dict_users_train[idx][:args.m_ft])
            # else:
            #     local = LocalUpdate(args=args, dataset=dataset_train[task], idxs=dict_users_train[idx][:args.m_tr])

            # appr = Appr(net, sbatch=args.batch_size, lr=args.lr, nepochs=args.nepochs, args=args, log_name=log_name)

            appr = apprs[idx]

            # appr.set_model(net_local.to(args.device))
            # last = iter == args.epochs

            # local_model,loss, indd = LongLifeTrain(args,appr,tr_dataloaders,iter,idx)
            # loss_locals.append(copy.deepcopy(loss))

            appr.set_trData(tr_dataloader[idx])
            #net_local = copy.deepcopy(net_glob).to(args.device)
            #appr.set_model(net_local)
            last = iter == args.epochs

            w_local, loss, indd = LongLifeTrain(args, appr, iter, None, idx)
            loss_locals.append(copy.deepcopy(loss))

            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_glob[key] * 1/m
                    w_locals[idx][key] = w_local[key]
            else:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    if key in w_glob_keys:
                        w_glob[key] += w_local[key] * 1/m
                    else:
                        w_glob[key] += w_local[key] * 1/m
                    w_locals[idx][key] = w_local[key]



        # get weighted average for global weights
        w_local = net_glob.state_dict()
        for k in w_glob.keys():
            w_local[k] = w_glob[k]
        if args.epochs != iter:
            net_glob.load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # acc_test, loss_test = test_img_local_all(None, args, dataset_test, dict_users_test, task, apprs=apprs,
        #                                          w_locals=None, return_all=False, write=write, round=iter,
        #                                          client_task=client_task)

        acc_test, loss_test = eval_test(apprs,te_dataloader,write,round=iter)

        accs.append(acc_test)

        print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
            iter, loss_avg, loss_test, acc_test))

        #fedavg
        if args.alg == 'Fedavg':
            if w_globals is not None:
                for i in range(args.num_users):
                    apprs[i].model.load_state_dict(w_globals)





        # DisGOSSIP
        else:
            clients_global = []
            all_users = [i for i in range(args.num_users)]

            for ind, idx in enumerate(all_users):
                temp = copy.deepcopy(all_users)
                temp.remove(idx)
                cur_local_models = []
                idxs_users = np.random.choice(temp, args.neibour, replace=False)
                for idx_user in idxs_users:
                    cur_local_models.append(apprs[idx_user].model.state_dict())

                # 这里需要同时把自己的模型也加进去
                cur_local_models.append(apprs[idx].model.state_dict())
                w_tmp = {}
                for w_local in cur_local_models:
                    if len(w_tmp) == 0:
                        w_tmp = copy.deepcopy(w_local)
                        for k, key in enumerate(net_glob.state_dict().keys()):
                            w_tmp[key] = w_tmp[key] * 1 / m
                            w_locals[idx][key] = w_local[key]
                    else:
                        for k, key in enumerate(net_glob.state_dict().keys()):
                            if key in w_glob_keys:
                                w_tmp[key] += w_local[key] * 1 / m
                            else:
                                w_tmp[key] += w_local[key] * 1 / m
                            w_locals[idx][key] = w_local[key]
                clients_global.append(copy.deepcopy(w_tmp))

            for ind, idx in enumerate(all_users):
                apprs[idx].model.load_state_dict(clients_global[idx])



        # if iter >= args.epochs - 10 and iter != args.epochs:
        #     accs10 += acc_test / 10
        # if iter >= args.epochs - 10 and iter != args.epochs:
        #     accs10_glob += acc_test / 10

        # model_save_path = './save/Baseline/0.4/accs_Fedavg_lambda_'+str(args.lamb) +str('_') + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
        #     args.shard_per_user) + '_iter' + str(iter) + '_frac_'+str(args.frac)+'.pt'
        # torch.save(net_glob.state_dict(), model_save_path)

    # print('Average accuracy final 10 rounds: {}'.format(accs10))
    # if args.alg == 'fedavg' or args.alg == 'prox':
    #     print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
    end = time.time()
    print(end - start)