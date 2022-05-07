# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import numpy as np
import quadprog


# Auxiliary functions useful for GEM's inner optimization.

def compute_offsets(task, nc_per_task, is_cifar):
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

class Appr():
    def __init__(self,
                 model,
                 n_outputs,
                 n_tasks,
                 args):
        super(Appr, self).__init__()
        self.margin = args.memory_strength
        self.is_cifar = True
        self.net = model
        self.pernet = deepcopy(self.net)
        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
        self.opt = optim.Adam(self.net.parameters(), args.lr)
        self.peropt = optim.Adam(self.pernet.parameters(), args.lr)
        self.gpu = True
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.optim_type = args.optim
        self.alpha = args.APFLalpha
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        self.num_classes = args.num_classes // args.task
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs
    def set_model(self,model):
        self.net = model
    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        if "SGD" in self.optim_type:
            optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, weight_decay=self.lr_decay)
        else:
            optimizer = torch.optim.Adam(self.net.parameters(), lr=lr,weight_decay=self.lr_decay)
        return optimizer
    def observe(self, x, t, y):
        self.opt = self._get_optimizer()
        # update w
        if t != self.old_task:
            self.old_task = t
        self.net.zero_grad()
        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        loss = self.ce(self.net.forward(x, t)[:, offset1: offset2], y - offset1)
        loss.backward()
        self.opt.step()
        # update v
        output1 = self.pernet.forward(x, t)[:, offset1: offset2]
        output2 = self.net.forward(x, t)[:, offset1: offset2]
        output = self.alpha*output1 + (1-self.alpha)*output2
        loss = self.ce(output, y - offset1)
        loss.backward()
        self.peropt.step()
        self.pernet.zero_grad()
    def alpha_update(self):
        grad_alpha = 0
        for l_params, p_params in zip(self.net.parameters(), self.pernet.parameters()):
            ## 这里为 v - w
            dif = p_params.data - l_params.data
            ## 这里为f(\bar{v}的损失)
            grad = self.alpha * p_params.grad.data + (1 - self.alpha) * l_params.grad.data
            ## 乘起来即可
            grad_alpha += dif.view(-1).T.dot(grad.view(-1))
        grad_alpha += 0.02 * self.alpha
        ## 进行更新
        alpha_n = self.alpha - self.lr * grad_alpha
        ## 确保在0，1之间
        alpha_n = np.clip(alpha_n.item(), 0.0, 1.0)
        self.alpha = alpha_n
    def validTest(self, t,tr_dataloader,sbatch=20):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.net.eval()
        # Loop batches
        with torch.no_grad():
            for images,targets in tr_dataloader:
                images = images.cuda()
                targets = targets.cuda()

                # Forward
                offset1, offset2 = compute_offsets(t, self.nc_per_task,
                                                   self.is_cifar)
                output1 = self.pernet.forward(images, t)[:, offset1: offset2]
                output2 = self.net.forward(images, t)[:, offset1: offset2]
                output = self.alpha * output1 + (1 - self.alpha) * output2

                loss = self.ce(output, targets - offset1)
                _, pred = output.max(1)
                hits = (pred == targets).float()

                # Log
                total_loss += loss.data.cpu().numpy() * len(targets)
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(targets)

        return total_loss / total_num, total_acc / total_num
def life_experience(task,appr,tr_dataloader,epochs,sbatch=10):
    for name,para in appr.net.named_parameters():
        para.requires_grad = True
    for e in range(epochs):
        for i,(images,targets) in enumerate(tr_dataloader):
            images = images.cuda()
            targets = targets.cuda()
            appr.net.train()
            appr.observe(images, task, targets)
            if i == 0:
                appr.alpha_update()

    loss,acc = appr.validTest(task,tr_dataloader)
    print('| Train finish, | Train: loss={:.3f}, acc={:5.1f}% | \n'.format( loss, 100 * acc), end='')
    return loss


def LongLifeTrain(args, appr, tr_dataloader, aggNum,idx):
    print('cur round :' + str(aggNum)+'  cur client:' + str(idx))
    # acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    # lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
    t = aggNum // args.round
    r = aggNum % args.round
    # for t, ncla in taskcla:

    print('*' * 100)
    # print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    # Get data
    task = t

    # Train
    loss = life_experience(task,appr,tr_dataloader,args.local_ep,args.local_bs)
    print('-' * 100)
    return appr.net.state_dict(),loss,0


def LongLifeTest(args, appr, t, testdatas, aggNum, writer):
    acc = np.zeros((1, t), dtype=np.float32)
    lss = np.zeros((1, t), dtype=np.float32)
    t = aggNum // args.round
    r = aggNum % args.round
    for u in range(t + 1):
        xtest = testdatas[u][0].cuda()
        ytest = (testdatas[u][1]).cuda()
        test_loss, test_acc = appr.validTest(u, xtest, ytest)
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