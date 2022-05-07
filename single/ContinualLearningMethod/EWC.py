import sys, time, os
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from utils import *
from torch.utils.tensorboard import SummaryWriter
import quadprog
sys.path.append('..')
import torch.nn.functional as F
import torch.nn as nn

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
def fisher_matrix_diag(t,dataloader, model):
    # Init
    fisher = {}
    for n, p in model.feature_net.named_parameters():
        fisher[n] = 0 * p.data
    # Compute
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    offset1, offset2 = compute_offsets(t, 10)
    all_num = 0
    for images,target in dataloader:
        images = images.cuda()
        target = (target - 10 * t).cuda()
        all_num += images.shape[0]
        # Forward and backward
        model.zero_grad()
        outputs = model.forward(images, t)[:, offset1: offset2]
        loss = criterion(outputs, target)
        loss.backward()
        # Get gradients
        for n, p in model.feature_net.named_parameters():
            if p.grad is not None:
                fisher[n] += images.shape[0] * p.grad.data.pow(2)
    # Mean
    with torch.no_grad():
        for n, _ in model.feature_net.named_parameters():
            fisher[n] = fisher[n] / all_num
    return fisher
def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1
def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1
def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))
def MultiClassCrossEntropy(logits, labels, t,T=2):
    # Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
    outputs = torch.log_softmax(logits / T, dim=1)  # compute the log of softmax values
    label = torch.softmax(labels / T, dim=1)
        # print('outputs: ', outputs)
        # print('labels: ', labels.shape)
    outputs = torch.sum(outputs * label, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)

    # print('OUT: ', outputs)
    return outputs

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return
class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self, model, tr_dataloader,nepochs=100, lr=0.001, lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=100,
                 args=None):
        self.model = model
        self.model_old = model
        self.fisher = None
        self.nepochs = nepochs
        self.tr_dataloader = tr_dataloader
        self.lr = lr
        self.lr_min = lr_min * 1 / 3
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad

        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.lamb = args.lamb
        self.e_rep = args.local_rep_ep
        self.old_task=-1
        self.grad_dims = []
        for param in self.model.feature_net.parameters():
            self.grad_dims.append(param.data.numel())

        return
    def set_model(self,model):
        self.model = model
    def set_fisher(self,fisher):
        self.fisher = fisher
    def set_trData(self,tr_dataloader):
        self.tr_dataloader = tr_dataloader
    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.momentum = 0.9
        # self.weight_decay = 0.0001
        #
        # optimizer =  torch.optim.SGD(self.model.parameters(), lr=lr, momentum=self.momentum,
        #                       weight_decay=self.weight_decay)
        return optimizer

    def train(self, t):
        if t!=self.old_task:
            self.model_old = deepcopy(self.model)
            self.model_old.train()
            freeze_model(self.model_old)  # Freeze the weights
            self.old_task=t
        lr = self.lr
        self.optimizer = self._get_optimizer(lr)

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            self.train_epoch(t)


            train_loss, train_acc = self.eval(t)
            if e % self.e_rep == self.e_rep -1:
                print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | \n'.format(
                    e + 1,  train_loss, 100 * train_acc), end='')
        # Fisher ops
        fisher_old = {}
        if t>0:
            for n, _ in self.model.feature_net.named_parameters():
                fisher_old[n] = self.fisher[n].clone()
        self.fisher = fisher_matrix_diag(t,self.tr_dataloader, self.model)
        if t > 0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
            for n, _ in self.model.feature_net.named_parameters():
                self.fisher[n] = (self.fisher[n] + fisher_old[n] * t) / (
                        t + 1)  # Checked: it is better than the other option

        return train_loss, train_acc

    def train_epoch(self,t):
        self.model.train()
        for images,targets in self.tr_dataloader:
            images = images.cuda()
            targets = (targets - 10 * t).cuda()
            # Forward current model
            offset1, offset2 = compute_offsets(t, 10)
            outputs = self.model.forward(images,t)[:,offset1:offset2]
            loss = self.criterion(t, outputs, targets)
            ## 根据这个损失计算梯度，变换此梯度
            # Backward
            loss.backward()
            self.optimizer.step()
        return

    def eval(self, t,train=True):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()
        if train:
            dataloaders = self.tr_dataloader

        # Loop batches
        with torch.no_grad():
            for images,targets in dataloaders:
                images = images.cuda()
                targets = (targets - 10*t).cuda()
                # Forward
                offset1, offset2 = compute_offsets(t, 10)
                output = self.model.forward(images,t)[:,offset1:offset2]

                loss = self.criterion(t, output, targets)
                _, pred = output.max(1)
                hits = (pred == targets).float()

                # Log
                total_loss += loss.data.cpu().numpy() * len(images)
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(images)

        return total_loss / total_num, total_acc / total_num

    def criterion(self, t, output, targets):
        # Regularization for all previous tasks
        loss_reg = 0
        if t > 0:
            for (name, param), (_, param_old) in zip(self.model.feature_net.named_parameters(), self.model_old.feature_net.named_parameters()):
                loss_reg += torch.sum(self.fisher[name] * (param_old - param).pow(2)) / 2
        return self.ce(output, targets) + self.lamb * loss_reg


def LongLifeTrain(args, appr, aggNum, writer,idx):
    print('cur round :' + str(aggNum)+'  cur client:' + str(idx))
    taskcla = []
    for i in range(10):
        taskcla.append((i, 10))
    # acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    # lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    t = aggNum // args.round
    print('cur task:'+ str(t))
    r = aggNum % args.round
    # for t, ncla in taskcla:

    print('*' * 100)
    # print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    # Get data
    task = t

    # Train
    loss,_ = appr.train(task)
    print('-' * 100)
    return appr.model.state_dict(),loss,0

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
    mean_acc = np.mean(acc[0, :t+1])
    mean_lss = np.mean(lss[0, :t])
    print('Average accuracy={:5.1f}%'.format(100 * np.mean(acc[0, :t+1])))
    print('Average loss={:5.1f}'.format(np.mean(lss[0, :t+1])))
    print('Save at ' + args.output)
    if r == args.round - 1:
        writer.add_scalar('task_finish_and _agg', mean_acc, t + 1)
    # np.savetxt(args.agg_output + 'aggNum_already_'+str(aggNum)+args.log_name, acc, '%.4f')
    return mean_lss, mean_acc
