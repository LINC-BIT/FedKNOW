# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/test.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y
from copy import deepcopy
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        d = int(self.idxs[item])
        image, label = self.dataset[d]
        return image, label

class DatasetSplit_leaf(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label
def compute_offsets(task, nc_per_task, is_cifar=True):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2
def test_img_local(net_g, dataset, args,t,idx=None,indd=None, user_idx=-1, idxs=None,appr = None,num_classes=10,glob_classify =None,device=None):
    net_g.to(device)
    net_g.eval()
    if glob_classify is not None:
        glob_classify.to(device)
        glob_classify.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(DatasetSplit(dataset,idxs), batch_size=args.local_test_bs,shuffle=False)
    count = 0
    for idx, (data, target) in enumerate(data_loader):
        offset1, offset2 = compute_offsets(t, num_classes)
        data = data.to(device)
        target = (target - num_classes * t).to(device)
        if appr is not None:
            appr.pernet.to(device)
            output1 = appr.pernet(data,t)[:, offset1:offset2]
            output2 = net_g(data,t)[:, offset1:offset2]
            log_probs = appr.alpha * output1 + (1-appr.alpha)*output2
        else:
            if glob_classify is None:
                log_probs = net_g(data,t)[:, offset1:offset2]
            else:
                features = net_g(data, return_feat=False)
                log_probs = glob_classify.forward(features, t)[:, offset1:offset2]
        # sum up batch loss

        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= count
    accuracy = 100.00 * float(correct) / count
    return  accuracy, test_loss
def test_img_local_meta(net_g, dataset, args, t, idx=None, indd=None, user_idx=-1, idxs=None, appr=None,num_classes = 10,device=None):
    opt = torch.optim.Adam(net_g.parameters(), args.lr)
    test_loss = 0
    correct = 0
    # put LEAF data into proper format

    data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_test_bs, shuffle=False)
    count = 0
    for idx, (data, target) in enumerate(data_loader):
        if idx == 0:
            data = data.to(device)
            target = (target - num_classes * t).to(device)
            offset1, offset2 = compute_offsets(t, num_classes)
            log_probs = net_g(data, t)[:, offset1:offset2]
            loss = F.cross_entropy(log_probs, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            data = data.to(device)
            target = (target - num_classes * t).to(device)
            offset1, offset2 = compute_offsets(t, num_classes)
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            net_g.to(device)
            if appr is not None:
                appr.pernet.to(device)
                output1 = appr.pernet(data, t)[:, offset1:offset2]
                output2 = net_g(data, t)[:, offset1:offset2]
                log_probs = appr.alpha * output1 + (1 - appr.alpha) * output2
            else:
                log_probs = net_g(data, t)[:, offset1:offset2]
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    test_loss /= count
    accuracy = 100.00 * float(correct) / count
    return accuracy, test_loss
def test_img_local_all(net, args, dataset_test, dict_users_test,t,w_locals=None,w_glob_keys=None, indd=None,dataset_train=None,dict_users_train=None, return_all=False,write =None,apprs = None,meta=False,num_classes = 10,glob_classify=None,device=None):
    print('test begin'+'*'*100)
    print('task '+str(t)+' finish train')
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    for idx in range(num_idxxs):
        net_local = copy.deepcopy(net)
        if w_locals is not None:
            w_local = net_local.state_dict()
            for k in w_locals[idx].keys():
                w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
        net_local.eval()
        all_task_acc = 0
        all_task_loss = 0
        for u in range(t+1):
            if apprs is not None:
                appr = apprs[idx]
            else:
                appr = None
            if meta:
                a, b = test_img_local_meta(deepcopy(net_local), dataset_test[u], args, u, user_idx=idx, idxs=dict_users_test[idx],
                                      appr=appr,device=device)
            else:
                a, b = test_img_local(net_local, dataset_test[u], args,u, user_idx=idx, idxs=dict_users_test[idx],appr = appr,num_classes=num_classes,glob_classify = glob_classify,device=device)
            all_task_acc+=a
            all_task_loss+=b
        all_task_acc /= (t+1)
        all_task_loss /= (t+1)
        tot += len(dict_users_test[idx])
        acc_test_local[idx] = all_task_acc*len(dict_users_test[idx])
        loss_test_local[idx] = all_task_loss*len(dict_users_test[idx])
        del net_local
    
    if return_all:
        return acc_test_local, loss_test_local
    if write is not None:
        write.add_scalar('task_finish_and _agg', sum(acc_test_local)/tot, t + 1)
    return  sum(acc_test_local)/tot, sum(loss_test_local)/tot


def test_img_local_all_WEIT(appr, args, dataset_test, dict_users_test, t, w_locals=None, w_glob_keys=None, indd=None,
                       dataset_train=None, dict_users_train=None, return_all=False, write=None,num_classes = 10,device=None):
    print('test begin' + '*' * 100)
    print('task ' + str(t) + ' finish train')
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    for idx in range(num_idxxs):
        net_local = copy.deepcopy(appr[idx].model)
        net_local.eval()
        all_task_acc = 0
        all_task_loss = 0
        for u in range(t + 1):
            a, b = test_img_local(net_local, dataset_test[u], args, u, user_idx=idx, idxs=dict_users_test[idx],num_classes=num_classes)
            all_task_acc += a
            all_task_loss += b
        all_task_acc /= (t + 1)
        all_task_loss /= (t + 1)
        tot += len(dict_users_test[idx])
        acc_test_local[idx] = all_task_acc * len(dict_users_test[idx])
        loss_test_local[idx] = all_task_loss * len(dict_users_test[idx])
        del net_local

    if return_all:
        return acc_test_local, loss_test_local
    if write is not None:
        write.add_scalar('task_finish_and _agg', sum(acc_test_local) / tot, t + 1)
    return sum(acc_test_local) / tot, sum(loss_test_local) / tot
