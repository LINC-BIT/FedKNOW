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
from torch.nn.parameter import Parameter
from models.Nets import RepTail

def gs_cal(t,tr_dataloader, model, num_classes,sbatch=20):
    # Init
    param_R = {}

    for name, param in model.named_parameters():
        if len(param.size()) <= 1:
            continue
        name = name.split('.')[:-1]
        name = '.'.join(name)
        param = param.view(param.size(0), -1)
        param_R['{}'.format(name)] = torch.zeros((param.size(0))).cuda()

    # Compute
    model.train()
    total_num = 0
    for images, targets in tr_dataloader:
        total_num += len(targets)
        images = images.cuda()
        targets = (targets - num_classes * t).cuda()
        # Forward current model
        offset1, offset2 = compute_offsets(t, num_classes)
        outputs = model.forward(images, t,avg_act=True)[:, offset1:offset2]
        cnt = 0

        for idx, j in enumerate(model.feature_net.act):
            j = torch.mean(j, dim=0)
            if len(j.size()) > 1:
                j = torch.mean(j.view(j.size(0), -1), dim=1).abs()
            model.feature_net.act[idx] = j

        for name, param in model.named_parameters():
            if len(param.size()) <= 1 or 'last' in name or 'downsample' in name:
                continue
            name = name.split('.')[:-1]
            name = '.'.join(name)
            param_R[name] += model.feature_net.act[cnt].abs().detach() * sbatch
            cnt += 1

    with torch.no_grad():
        for key in param_R.keys():
            param_R[key] = (param_R[key] / total_num)
    return param_R
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return
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
class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self, model, tr_dataloader,nepochs=100, lr=0.001, lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=100,
                 args=None):
        self.model = model
        self.model_old = deepcopy(model)
        self.omega = None
        self.nepochs = nepochs
        self.tr_dataloader = tr_dataloader
        self.lr = lr
        self.lr_decay = args.lr_decay
        self.lr_min = lr_min * 1 / 3
        self.eta  = 0.9
        self.rho =0.3
        self.ce = torch.nn.CrossEntropyLoss()
        self.optim_type = args.optim
        self.optimizer = self._get_optimizer()
        self.old_task=-1
        self.lamb = 8
        self.initail_mu = 10
        self.mu = 10
        self.freeze = {}
        self.mask = {}
        self.num_classes = args.num_classes // args.task
        for (name,p) in self.model.named_parameters():
            if len(p.size())<2:
                continue
            name = name.split('.')[:-1]
            name = '.'.join(name)
            self.mask[name] = torch.zeros(p.shape[0]).cuda()

        # if len(args.parameter)>=1:
        #     params=args.parameter.split(',')
        #     print('Setting parameters to',params)
        #     self.lamb=float(params[0])
        return
    def set_model(self,model):
        self.model = model
    def set_fisher(self,fisher):
        self.fisher = fisher
    def set_trData(self,tr_dataloader):
        self.tr_dataloader = tr_dataloader

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        if "SGD" in self.optim_type:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.lr_decay)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.lr_decay)
        return optimizer
    def train(self, t):
        if t!=self.old_task:
            self.old_task=t
            self.model_old = deepcopy(self.model)
            self.model_old.train()
            freeze_model(self.model_old)  # Freeze the weights
        lr = self.lr
        self.optimizer = self._get_optimizer()
        # Loop epochs
        if t>0:
            self.freeze = {}
            for name, param in self.model.named_parameters():
                if 'bias' in name or 'last' in name:
                    continue
                key = name.split('.')
                key = key[0]+'.'+key[1]
                if 'conv1' not in name:
                    if 'conv' in name: #convolution layer
                        temp = torch.ones_like(param)
                        temp[:, self.omega[prekey] == 0] = 0
                        temp[self.omega[key] == 0] = 1
                        self.freeze[key] = temp
                    else:#linear layer
                        temp = torch.ones_like(param)
                        temp = temp.reshape((temp.size(0), self.omega[prekey].size(0) , -1))
                        temp[:, self.omega[prekey] == 0] = 0
                        temp[self.omega[key] == 0] = 1
                        self.freeze[key] = temp.reshape(param.shape)
                prekey = key
        for e in range(self.nepochs):
            self.train_epoch(t)
            train_loss, train_acc = self.eval(t)
        self.model.feature_net.act = None
        temp = gs_cal(t, self.tr_dataloader, self.model,self.num_classes)
        for n in temp.keys():
            if t > 0:
                self.omega[n] = self.eta * self.omega[n] + temp[n]
            else:
                self.omega = temp
            self.mask[n] = (self.omega[n] > 0).float()

        test_loss, test_acc = self.eval(t)
        print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(test_loss, 100 * test_acc))

        dummy = RepTail([3,32,32]).cuda()

        pre_name = 0

        for (name, dummy_layer), (_, layer) in zip(dummy.named_children(), self.model.named_children()):
            with torch.no_grad():
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    if pre_name != 0:
                        temp = (self.omega[pre_name] > 0).float()
                        if isinstance(layer, nn.Linear) and 'conv' in pre_name:
                            temp = temp.unsqueeze(0).unsqueeze(-1)
                            weight = layer.weight
                            weight = weight.view(weight.size(0), temp.size(1), -1)
                            weight = weight * temp
                            layer.weight.data = weight.view(weight.size(0), -1)
                        elif len(weight.size()) > 2:
                            temp = temp.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                            layer.weight *= temp
                        else:
                            temp = temp.unsqueeze(0)
                            layer.weight *= temp

                    weight = layer.weight.data
                    bias = layer.bias.data

                    if len(weight.size()) > 2:
                        norm = weight.norm(2, dim=(1, 2, 3))
                        mask = (self.omega[name] == 0).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                    else:
                        norm = weight.norm(2, dim=(1))
                        mask = (self.omega[name] == 0).float().unsqueeze(-1)

                    zero_cnt = int((mask.sum()).item())
                    indice = np.random.choice(range(zero_cnt), int(zero_cnt * (1 - self.rho)), replace=False)
                    indice = torch.tensor(indice).long()
                    idx = torch.arange(weight.shape[0])[mask.flatten(0) == 1][indice]
                    mask[idx] = 0

                    layer.weight.data = (1 - mask) * layer.weight.data + mask * dummy_layer.weight.data
                    mask = mask.squeeze()
                    layer.bias.data = (1 - mask) * bias + mask * dummy_layer.bias.data

                    pre_name = name

                if isinstance(layer, nn.ModuleList):

                    weight = layer[t].weight
                    weight[:, self.omega[pre_name] == 0] = 0
                    # if 'omniglot' in args.experiment:
                    #     weight = weight.view(weight.shape[0], self.omega[pre_name].shape[0], -1)
                    #     weight[:, self.omega[pre_name] == 0] = 0
                    #     weight = weight.view(weight.shape[0], -1)
                    # else:
                    #     weight[:, self.omega[pre_name] == 0] = 0
        # Fisher ops

        return train_loss, train_acc

    def train_epoch(self, t):
        self.model.train()
        # Loop batches
        for images,targets in self.tr_dataloader:
            images = images.cuda()
            targets = (targets - self.num_classes * t).cuda()
            # Forward current model
            offset1, offset2 = compute_offsets(t, self.num_classes)
            outputs = self.model.forward(images, t)[:, offset1:offset2]

            # Forward current model

            loss = self.ce(outputs, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Freeze the outgoing weights
            if t > 0:
                for name, param in self.model.named_parameters():
                    if 'bias' in name or 'last' in name or 'conv1' in name:
                        continue
                    key = name.split('.')
                    key = key[0]+'.'+key[1]
                    param.data = param.data * self.freeze[key]

        self.proxy_grad_descent(t, self.lr)

    def proxy_grad_descent(self, t, lr):
        with torch.no_grad():
            for (name, module), (_, module_old) in zip(self.model.named_children(), self.model_old.named_children()):
                if not isinstance(module, torch.nn.Linear) and not isinstance(module, torch.nn.Conv2d):
                    continue

                mu = self.mu

                key = name
                weight = module.weight
                bias = module.bias
                weight_old = module_old.weight
                bias_old = module_old.bias

                if len(weight.size()) > 2:
                    norm = weight.norm(2, dim=(1, 2, 3))
                else:
                    norm = weight.norm(2, dim=(1))
                norm = (norm ** 2 + bias ** 2).pow(1 / 2)

                aux = F.threshold(norm - mu * lr, 0, 0, False)
                alpha = aux / (aux + mu * lr)
                coeff = alpha * (1 - self.mask[key])

                if len(weight.size()) > 2:
                    sparse_weight = weight.data * coeff.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                else:
                    sparse_weight = weight.data * coeff.unsqueeze(-1)
                sparse_bias = bias.data * coeff

                penalty_weight = 0
                penalty_bias = 0

                if t > 0:
                    if len(weight.size()) > 2:
                        norm = (weight - weight_old).norm(2, dim=(1, 2, 3))
                    else:
                        norm = (weight - weight_old).norm(2, dim=(1))

                    norm = (norm ** 2 + (bias - bias_old) ** 2).pow(1 / 2)

                    aux = F.threshold(norm - self.omega[key] * self.lamb * lr, 0, 0, False)
                    boonmo = lr * self.lamb * self.omega[key] + aux
                    alpha = (aux / boonmo)
                    alpha[alpha != alpha] = 1

                    coeff_alpha = alpha * self.mask[key]
                    coeff_beta = (1 - alpha) * self.mask[key]

                    if len(weight.size()) > 2:
                        penalty_weight = coeff_alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * weight.data + \
                                         coeff_beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * weight_old.data
                    else:
                        penalty_weight = coeff_alpha.unsqueeze(-1) * weight.data + coeff_beta.unsqueeze(
                            -1) * weight_old.data
                    penalty_bias = coeff_alpha * bias.data + coeff_beta * bias_old.data

                diff_weight = (sparse_weight + penalty_weight) - weight.data
                diff_bias = sparse_bias + penalty_bias - bias.data

                weight.data = sparse_weight + penalty_weight
                bias.data = sparse_bias + penalty_bias

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
                targets = (targets - self.num_classes*t).cuda()
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
def LongLifeTrain(args, appr, aggNum, writer,idx):
    print('cur round :' + str(aggNum)+'  cur client:' + str(idx))
    taskcla = []
    for i in range(10):
        taskcla.append((i, 10))
    t = aggNum // args.round
    print('cur task:'+ str(t))
    r = aggNum % args.round

    print('*' * 100)
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

# def main():
#     # cifar100 = Cifar100Task('../data',batch_size=900,num_clients=5,cur_client=4,task_num=10,isFed=True)
#     cifar100 = Cifar100Task('../data/cifar-100-python', batch_size=4500, task_num=10, num_clients=5, cur_client=0,
#                       isFed=True)
#     TaskDatas = cifar100.getDatas()
#     net = network.RepTail([3, 32, 32]).cuda()
#
#
# if __name__ == "__main__":
#     main()