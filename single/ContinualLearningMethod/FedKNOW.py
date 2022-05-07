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
def fisher_matrix_diag(t,dataloader, model,num_classes = 10,device=None):
    # Init
    fisher = {}
    for n, p in model.feature_net.named_parameters():
        fisher[n] = 0 * p.data
    # Compute
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    offset1, offset2 = compute_offsets(t, num_classes)
    all_num = 0
    for images,target in dataloader:
        images = images.to(device)
        target = (target - num_classes * t).to(device)
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
def project2cone2(gradient, memories,memory, margin=0.5, eps=1e-3):
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
    try:
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        gradient.copy_(torch.Tensor(x).view(-1, 1))
    except ValueError:
        memory_np = memory.cpu().t().double().numpy()
        t = memory_np.shape[0]
        P = np.dot(memory_np, memory_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
        q = np.dot(memory_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + margin
        try:
            v = quadprog.solve_qp(P, q, G, h)[0]
            x = np.dot(v, memory_np) + gradient_np
            gradient.copy_(torch.Tensor(x).view(-1, 1))
        except ValueError:
            gradient.copy_(torch.Tensor(gradient_np).view(-1, 1))
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
    def __init__(self, model, packnet,packmodel, tr_dataloader,nepochs=100, lr=0.001, lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=100,
                 args=None):
        self.device = args.device
        self.num_classes = args.num_classes
        self.model = model
        self.model_old = model
        self.pack = packnet
        self.packmodel = packmodel
        self.fisher = None
        self.nepochs = nepochs
        self.tr_dataloader = tr_dataloader
        self.lr = lr
        self.lr_decay = args.lr_decay
        self.lr_min = lr_min * 1 / 3
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.ce = torch.nn.CrossEntropyLoss()
        self.optim_type = args.optim
        self.optimizer = self._get_optimizer()
        self.pack_optimizer = self._get_optimizer(packmodel)
        self.lamb = args.lamb
        self.e_rep = args.local_local_ep
        self.old_task=-1
        self.grad_dims = []
        self.num_classes = args.num_classes // args.task
        for param in self.model.feature_net.parameters():
            self.grad_dims.append(param.data.numel())
        self.select_grad_num = args.select_grad_num
        return
    def set_model(self,model):
        self.model = model
    def set_fisher(self,fisher):
        self.fisher = fisher
    def set_trData(self,tr_dataloader):
        self.tr_dataloader = tr_dataloader
    def _get_optimizer(self, model=None,lr=None):
        if lr is None: lr = self.lr
        optimizer =None
        if model == None:
            if "SGD" in self.optim_type:
                optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.lr_decay)
            else:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,weight_decay=self.lr_decay)
        else:
            if "SGD" in self.optim_type:
                optimizer = torch.optim.SGD(self.packmodel.parameters(), lr=lr, weight_decay=self.lr_decay)
            else:
                optimizer = torch.optim.Adam(self.packmodel.parameters(), lr=lr,weight_decay=self.lr_decay)
        return optimizer
    def train(self, t):
        self.model.to(self.device)
        self.model_old.to(self.device)
        self.packmodel.to(self.device)
        oldpackmodel = deepcopy(self.packmodel)
        if t!=self.old_task:
            self.model_old = deepcopy(self.model)
            self.model_old.train()
            freeze_model(self.model_old)  # Freeze the weights
            self.old_task=t
        self.optimizer = self._get_optimizer()
        self.pack.on_init_end(self.packmodel,t)
        # trian model
        if len(self.pack.masks) > t:
            self.pack.masks.pop()
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()

            if e < self.e_rep:
                for name,para in self.model.named_parameters():
                    if 'feature_net' in name:
                        para.requires_grad = False
                    else:
                        para.requires_grad = True
            else :
                for name,para in self.model.named_parameters():
                    if 'feature_net' in name:
                        para.requires_grad = True
                    else:
                        para.requires_grad = False
            if t == 0:
                self.train_epoch_rep(t, e, oldpackmodel)
            else:
                if e < self.e_rep:
                    self.train_epoch_head(t)
                else:
                    self.train_epoch_rep(t, e,oldpackmodel)
            self.train_packnet(t)
            self.pack.on_epoch_end(self.packmodel.feature_net,e,t)


            train_loss, train_acc = self.eval(t)
            if e % self.e_rep == self.e_rep -1:
                print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | \n'.format(
                    e + 1,  train_loss, 100 * train_acc), end='')
        # Fisher ops
        # fisher_old = {}
        # if t>0:
        #     for n, _ in self.model.feature_net.named_parameters():
        #         fisher_old[n] = self.fisher[n].clone()
        # self.fisher = fisher_matrix_diag(t,self.tr_dataloader, self.model,num_classes=self.num_classes)
        # if t > 0:
        #     # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
        #     for n, _ in self.model.feature_net.named_parameters():
        #         self.fisher[n] = (self.fisher[n] + fisher_old[n] * t) / (
        #                 t + 1)  # Checked: it is better than the other option
        return self.fisher,train_loss, train_acc
    def train_packnet(self,t):
        self.packmodel.train()
        for images, targets in self.tr_dataloader:
            images = images.to(self.device)
            targets = (targets - self.num_classes * t).to(self.device)
            offset1, offset2 = compute_offsets(t, self.num_classes)
            outputs = self.packmodel.forward(images, t)[:, offset1:offset2]
            loss = self.ce(outputs, targets)
            self.pack_optimizer.zero_grad()
            loss.backward()
            self.pack.on_after_backward(self.packmodel.feature_net,t)
            self.pack_optimizer.step()

    def train_epoch_head(self,t):
        self.model.train()
        for images,targets in self.tr_dataloader:
            images = images.to(self.device)
            targets = (targets - self.num_classes * t).to(self.device)
            # Forward current model
            offset1, offset2 = compute_offsets(t, self.num_classes)
            preLabels = self.model_old.forward(images,t,pre=True)[:, 0: offset1]
            preoutputs = self.model.forward(images,t,pre=True)[:, 0: offset1]
            # self.model.zero_grad()
            self.optimizer.zero_grad()
            self.model.zero_grad()
            memoryloss=MultiClassCrossEntropy(preoutputs,preLabels,t,T=1)
            memoryloss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.model.zero_grad()
            # store_grad(self.model.parameters,grads, self.grad_dims,0)
            outputs = self.model.forward(images,t)[:,offset1:offset2]
            loss = self.ce(outputs, targets)
            ## 根据这个损失计算梯度，变换此梯度

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return
    def train_epoch_rep(self, t, epoch,oldpackmodel):
        self.model.train()
        self.packmodel.train()
        # Loop batches
        for images,targets in self.tr_dataloader:
            # Forward current model
            images = images.to(self.device)
            targets = (targets - self.num_classes * t).to(self.device)
            pre_loss = 0
            grads = torch.Tensor(sum(self.grad_dims), 2+t)
            offset1, offset2 = compute_offsets(t, self.num_classes)
            grads = grads.to(self.device)
            if t > 0:
                preLabels = self.model_old.forward(images, t, pre=True)[:, 0: offset1]
                preoutputs = self.model.forward(images, t, pre=True)[:, 0: offset1]
                self.model.zero_grad()
                self.optimizer.zero_grad()
                pre_loss=MultiClassCrossEntropy(preoutputs,preLabels,t,T=2)
                pre_loss.backward()
                store_grad(self.model.feature_net.parameters,grads, self.grad_dims,0)
                if t >= self.select_grad_num:
                    t = self.select_grad_num -1
                for i in range(t):
                    self.model.zero_grad()
                    self.optimizer.zero_grad()
                    begin, end = compute_offsets(i, self.num_classes)
                    temppackmodel = deepcopy(oldpackmodel).to(self.device)
                    temppackmodel.train()
                    self.pack.apply_eval_mask(task_idx=i, model=temppackmodel.feature_net)
                    preoutputs = self.model.forward(images, t, pre=True)[:, begin:end]
                    with torch.no_grad():
                        oldLabels = temppackmodel.forward(images, i)[:, begin:end]
                    memoryloss = MultiClassCrossEntropy(preoutputs, oldLabels, i, T=2)
                    memoryloss.backward()
                    store_grad(self.model.feature_net.parameters, grads, self.grad_dims, i+1)
                    del temppackmodel
                ## 求出每个分类器算出来的梯度

            outputs = self.model.forward(images,t)[:,offset1:offset2]
            loss = self.ce(outputs, targets)
            ## 根据这个损失计算梯度，变换此梯度

            # Backward
            self.model.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            if t>0:
                # copy gradient
                store_grad(self.model.feature_net.parameters, grads, self.grad_dims, t+1)
                taskl = [i for i in range(t+2)]
                indx = torch.LongTensor(taskl[:-1]).to(self.device)
                errindx = torch.LongTensor(0).to(self.device)
                dotp = torch.mm(grads[:, 1].unsqueeze(0),
                                grads.index_select(1, indx))
                if (dotp < 0).sum() != 0:
                    project2cone2(grads[:, t+1].unsqueeze(1),
                                  grads.index_select(1, indx), grads.index_select(1,errindx))
                    # copy gradients back
                    overwrite_grad(self.model.feature_net.parameters, grads[:, t+1],
                                   self.grad_dims)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=50, norm_type=2)
            self.optimizer.step()
        return

    def moretrain(self,t):
        self.packmodel.to(self.device)
        for e in range(self.nepochs):
            self.train_packnet(t)
            self.pack.on_epoch_end(self.packmodel.feature_net, e, t)

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
                images = images.to(self.device)
                targets = (targets - self.num_classes*t).to(self.device)
                # Forward
                offset1, offset2 = compute_offsets(t, self.num_classes)
                output = self.model.forward(images,t)[:,offset1:offset2]

                loss = self.ce(output, targets)
                _, pred = output.max(1)
                hits = (pred == targets).float()

                # Log
                total_loss += loss.data.cpu().numpy() * len(images)
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(images)

        return total_loss / total_num, total_acc / total_num

    def criterion(self, t):
        # Regularization for all previous tasks
        loss_reg = 0
        if t > 0:
            for (name, param), (_, param_old) in zip(self.model.feature_net.named_parameters(), self.model_old.feature_net.named_parameters()):
                loss_reg += torch.sum(self.fisher[name] * (param_old - param).pow(2)) / 2
        return self.lamb * loss_reg


def LongLifeTrain(args, appr, aggNum, writer,idx):
    print('cur round :' + str(aggNum)+'  cur client:' + str(idx))
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
    fisher,loss,_ = appr.train(task)
    print('-' * 100)
    return appr.model.state_dict(),fisher,loss,0

def LongLifeTest(args, appr, t, testdatas, aggNum, writer):
    acc = np.zeros((1, t), dtype=np.float32)
    lss = np.zeros((1, t), dtype=np.float32)
    t = aggNum // args.round
    r = aggNum % args.round
    # for u in range(t + 1):
    #     xtest = testdatas[u][0].to(self.device)
    #     ytest = (testdatas[u][1] - u * 10).to(self.device)
    #     test_loss, test_acc = appr.eval(u, xtest, ytest)
    #     print('>>> Test on task {:2d} : loss={:.3f}, acc={:5.1f}% <<<'.format(u, test_loss,
    #                                                                           100 * test_acc))
    #     acc[0, u] = test_acc
    #     lss[0, u] = test_loss
    # # Save
    # mean_acc = np.mean(acc[0, :t+1])
    # mean_lss = np.mean(lss[0, :t])
    # print('Average accuracy={:5.1f}%'.format(100 * np.mean(acc[0, :t+1])))
    # print('Average loss={:5.1f}'.format(np.mean(lss[0, :t+1])))
    # print('Save at ' + args.output)
    # if r == args.round - 1:
    #     writer.add_scalar('task_finish_and _agg', mean_acc, t + 1)
    # # np.savetxt(args.agg_output + 'aggNum_already_'+str(aggNum)+args.log_name, acc, '%.4f')
    # return mean_lss, mean_acc

# def main():
#     # cifar100 = Cifar100Task('../data',batch_size=900,num_clients=5,cur_client=4,task_num=10,isFed=True)
#     cifar100 = Cifar100Task('../data/cifar-100-python', batch_size=4500, task_num=10, num_clients=5, cur_client=0,
#                       isFed=True)
#     TaskDatas = cifar100.getDatas()
#     net = network.RepTail([3, 32, 32]).to(self.device)
#
#
# if __name__ == "__main__":
#     main()