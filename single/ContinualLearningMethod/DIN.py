import sys, time, os
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
#from ClientTrain.utils import *
from torch.utils.tensorboard import SummaryWriter
import quadprog


class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self, model, tr_dataloader, nepochs=100, lr=0.001, lr_min=1e-6, lr_factor=3, lr_patience=5,
                 clipgrad=100,
                 args=None, kd_model=None):
        self.model = model
        self.model_old = model
        self.device = args.device
        self.fisher = None
        self.nepochs = nepochs
        self.tr_dataloader = tr_dataloader
        self.lr = lr
        self.lr_min = lr_min * 1 / 3
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.args = args
        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        #self.optimizer = torch.optim.Adam(model.parameters(), lr, (0.5, 0.99))

        self.lamb = args.lamb  # 20000
        self.e_rep = args.local_local_ep
        self.old_task = -1
        self.grad_dims = []
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())
        self.first_train = True

        return

    def set_model(self, model):
        self.model = model

    def set_fisher(self, fisher):
        self.fisher = fisher

    def set_trData(self, tr_dataloader):
        self.tr_dataloader = tr_dataloader

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if 'vit' in self.args.model or 'pit' in self.args.model:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=0.05)
        # self.momentum = 0.9
        # self.weight_decay = 0.0001
        #
        # optimizer =  torch.optim.SGD(self.model.parameters(), lr=lr, momentum=self.momentum,
        #                       weight_decay=self.weight_decay)
        return optimizer

    def train(self):
        def eval_output(scores, target, dtype,loss_function=torch.nn.functional.binary_cross_entropy_with_logits):
            loss = loss_function(scores.type(dtype), target.type(dtype))

            y_pred = scores.sigmoid().round()
            accuracy = (y_pred == target).type(dtype).mean()

            return loss, accuracy
        train_loss = 0
        train_acc = 0
        self.optimizer = self._get_optimizer()
        for e in range(self.nepochs):
            total = 0
            loss_sum = 0
            acc_sum = 0

            for i, data in enumerate(self.tr_dataloader):

                # transform data to target device
                data = [item.to(self.device) if item != None else None for item in data]
                target = data.pop(-1)

                self.model.zero_grad()

                scores = self.model(data, neg_sample=False)

                loss, accuracy = eval_output(scores, target,torch.cuda.FloatTensor)
                total += len(target)
                loss_sum += loss.item()*len(target)
                acc_sum += accuracy*len(target)
                loss.backward()
                self.optimizer.step()

            if e % self.e_rep == self.e_rep - 1:
                train_loss = loss_sum/total
                train_acc = 100 * acc_sum/total
                print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | \n'.format(e + 1,  loss_sum/total, 100 * acc_sum/total), end='')

        return train_loss,train_acc



    # def train(self, t):
    #     if t != self.old_task:
    #         self.model_old = deepcopy(self.model)
    #         self.model_old.train()
    #         freeze_model(self.model_old)  # Freeze the weights
    #         self.old_task = t
    #         self.first_train = True
    #
    #     lr = self.lr
    #     self.optimizer = self._get_optimizer(lr)
    #
    #     # Loop epochs
    #     for e in range(self.nepochs):
    #         # Train
    #         self.train_epoch_rep(t, e)
    #         # train_loss, train_acc = self.eval(t)
    #         # if e % self.e_rep == self.e_rep -1:
    #         #     print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | \n'.format(
    #         #         e + 1,  train_loss, 100 * train_acc), end='')
    #     # Fisher ops
    #     fisher_old = {}
    #     if t > 0:
    #         for n, _ in self.model.feature_net.named_parameters():
    #             fisher_old[n] = self.fisher[n].clone()
    #     self.fisher = fisher_matrix_diag(t, self.tr_dataloader, self.model, self.device)
    #     if t > 0:
    #         # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
    #         for n, _ in self.model.feature_net.named_parameters():
    #             self.fisher[n] = (self.fisher[n] + fisher_old[n] * t) / (
    #                     t + 1)  # Checked: it is better than the other option
    #
    #     self.first_train = False
    #     # return train_loss, train_acc
    #     return 0, 0

    # def train_kd(self, images, targets, t):
    #     self.cur_kd.train()
    #     kd_optimizer = torch.optim.Adam(self.cur_kd.parameters(), lr=0.0005)
    #     # Forward current model
    #     offset1, offset2 = compute_offsets(t, 10)
    #     kd_outputs = self.cur_kd.forward(images, t)[:, offset1:offset2]
    #     outputs = self.model.forward(images, t)[:, offset1:offset2]
    #     loss = self.ce(kd_outputs, targets) + self.softloss(outputs, kd_outputs)
    #     kd_optimizer.zero_grad()
    #     loss.backward()
    #     kd_optimizer.step()

    # def train_epoch(self, t):
    #     self.model.train()
    #     for images, targets in self.tr_dataloader:
    #         images = images.to(self.device)
    #         targets = (targets - 10 * t).to(self.device)
    #         # Forward current model
    #         offset1, offset2 = compute_offsets(t, 10)
    #         outputs = self.model.forward(images, t)[:, offset1:offset2]
    #         loss = self.ce(outputs, targets)
    #         ## 根据这个损失计算梯度，变换此梯度
    #         # Backward
    #         loss.backward()
    #         self.optimizer.step()
    #     return

    # def train_epoch_rep(self, t, epoch, kd_lambda=0.0):
    #     self.model.train()
    #     # Loop batches
    #     for images, targets in self.tr_dataloader:
    #         # Forward current model
    #         images = images.to(self.device)
    #         targets = (targets - 10 * t).to(self.device)
    #         pre_loss = 0
    #         grads = torch.Tensor(sum(self.grad_dims), 2)
    #         offset1, offset2 = compute_offsets(t, 10)
    #         # if t > 0:
    #         #     preLabels = self.model_old.forward(images, t, pre=True)[:, 0: offset1]
    #         #     preoutputs = self.model.forward(images, t, pre=True)[:, 0: offset1]
    #         #     self.model.zero_grad()
    #         #     self.optimizer.zero_grad()
    #         #     pre_loss=MultiClassCrossEntropy(preoutputs,preLabels,t,T=2)
    #         #     pre_loss.backward()
    #         #     store_grad(self.model.feature_net.parameters,grads, self.grad_dims,0)
    #         #     ## 求出每个分类器算出来的梯度
    #
    #         outputs = self.model.forward(images, t)[:, offset1:offset2]
    #         loss = self.criterion(t, outputs, targets)
    #         ## 根据这个损失计算梯度，变换此梯度
    #         # Backward
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #
    #     return

    # def eval(self, t, train=True, model=None):
    #     total_loss = 0
    #     total_acc = 0
    #     total_num = 0
    #     if train:
    #         dataloaders = self.tr_dataloader
    #     if model is None:
    #         model = self.model
    #     # Loop batches
    #     model.eval()
    #     with torch.no_grad():
    #         for images, targets in dataloaders:
    #             images = images.to(self.device)
    #             targets = (targets - 10 * t).to(self.device)
    #             # Forward
    #             offset1, offset2 = compute_offsets(t, 10)
    #             output = model.forward(images, t)[:, offset1:offset2]
    #
    #             loss = self.criterion(t, output, targets)
    #             _, pred = output.max(1)
    #             hits = (pred == targets).float()
    #
    #             # Log
    #             total_loss += loss.data.cpu().numpy() * len(images)
    #             total_acc += hits.sum().data.cpu().numpy()
    #             total_num += len(images)
    #
    #     return total_loss / total_num, total_acc / total_num

    def eval(self,loader_test):
        def eval_output(scores, target, dtype, loss_function=torch.nn.functional.binary_cross_entropy_with_logits):
            loss = loss_function(scores.type(dtype), target.type(dtype))

            y_pred = scores.sigmoid().round()
            accuracy = (y_pred == target).type(dtype).mean()

            return loss, accuracy

        self.model.eval()
        with torch.no_grad():
            score_list = [];
            target_list = []
            for data in loader_test:
                data = [item.to(self.device) if item != None else None for item in data]

                target = data.pop(-1)

                scores = self.model(data, neg_sample=False)
                score_list.append(scores)
                target_list.append(target)
            scores = torch.cat(score_list, dim=-1)
            target = torch.cat(target_list, dim=-1)
            loss, accuracy= eval_output(scores, target,torch.cuda.FloatTensor)
        self.model.train()
        return loss,accuracy


    def criterion(self, t, output, targets):
        # Regularization for all previous tasks
        loss_reg = 0
        if t > 0:
            for (name, param), (_, param_old) in zip(self.model.feature_net.named_parameters(),
                                                     self.model_old.feature_net.named_parameters()):
                loss_reg += torch.sum(self.fisher[name] * (param_old - param).pow(2)) / 2
        return self.ce(output, targets) + self.lamb * loss_reg


def LongLifeTrain(args, appr, aggNum, writer, idx):
    print('cur round :' + str(aggNum) + '  cur client:' + str(idx))
    # taskcla = []
    # for i in range(10):
    #     taskcla.append((i, 10))
    # # acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    # # lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    # t = aggNum // args.round
    # print('cur task:' + str(t))
    # r = aggNum % args.round
    # # for t, ncla in taskcla:
    #
    # print('*' * 100)
    # print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    # Get data
    # task = t

    # Train
    #loss, _ = appr.train(task)
    loss, acc = appr.train()

    print('-' * 100)

    return appr.model.state_dict(), loss, acc


def LongLifeTest(args, appr, t, testdatas, aggNum, writer):
    acc = np.zeros((1, t), dtype=np.float32)
    lss = np.zeros((1, t), dtype=np.float32)
    t = aggNum // args.round
    r = aggNum % args.round
    for u in range(t + 1):
        xtest = testdatas[u][0].cuda()
        ytest = (testdatas[u][1] - u * 10).cuda()
        test_loss, test_acc = appr.eval(u, xtest, ytest)
        print('>>> Test on task {:2d} : loss={:.3f}, acc={:5.1f}% <<<'.format(u, test_loss,
                                                                              100 * test_acc))
        acc[0, u] = test_acc
        lss[0, u] = test_loss
    # Save
    mean_acc = np.mean(acc[0, :t + 1])
    mean_lss = np.mean(lss[0, :t])
    print('Average accuracy={:5.1f}%'.format(100 * np.mean(acc[0, :t + 1])))
    print('Average loss={:5.1f}'.format(np.mean(lss[0, :t + 1])))
    print('Save at ' + args.output)
    if r == args.round - 1:
        writer.add_scalar('task_finish_and _agg', mean_acc, t + 1)
    # np.savetxt(args.agg_output + 'aggNum_already_'+str(aggNum)+args.log_name, acc, '%.4f')
    return mean_lss, mean_acc