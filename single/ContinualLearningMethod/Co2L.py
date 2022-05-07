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
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
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
def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, target_labels=None, reduction='mean'):
        assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        curr_class_mask = torch.zeros_like(labels)
        for tc in target_labels:
            curr_class_mask += (labels == tc)
        curr_class_mask = curr_class_mask.view(-1).to(device)
        loss = curr_class_mask * loss.view(anchor_count, batch_size)

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'none':
            loss = loss.mean(0)
        else:
            raise ValueError('loss reduction not supported: {}'.
                             format(reduction))

        return loss
class Appr(object):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self, model, tr_dataloader,nepochs=100, lr=0.001, lr_min=1e-6, lr_factor=3, lr_patience=5, clipgrad=100,
                 args=None):
        self.model = model
        self.nepochs = nepochs
        self.tr_dataloader = tr_dataloader
        self.lr = lr
        self.lr_min = lr_min * 1 / 3
        self.lr_decay = args.lr_decay
        self.optim_type = args.optim
        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.old_task=-1
        self.num_classes = args.num_classes
        self.current_temp = 0.2
        self.pre_model = deepcopy(self.model)
        self.criterion = SupConLoss(temperature=0.5)
        self.past_temp = 0.01
        self.distill_power = 1
        return
    def set_model(self,model):
        self.model = deepcopy(model)
    def set_classify(self,classify):
        self.classify = deepcopy(classify)
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
            self.pre_model = deepcopy(self.model)
        lr = self.lr
        # Loop epochs
        for e in range(self.nepochs):
            losses = self.new_train(t,e)
            if e == self.nepochs -1:
                losses = self.new_train(t, e,endEpoch=True)
        # Fisher ops
        return losses
    def tune(self, t):
        self.model.cuda()
        self.classify.cuda()
        if t!=self.old_task:
            self.old_task=t
        lr = self.lr
        # Loop epochs
        for e in range(self.nepochs):
            losses = self.tune_epoch(t)
            train_loss, train_acc = self.eval(t)
            if e % self.nepochs == self.nepochs - 1:
                print('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | \n'.format(
                    e + 1, train_loss, 100 * train_acc), end='')
        # Fisher ops
        return losses
    def tune_epoch(self,t):
        self.optimizer = self._get_optimizer()
        self.model.train()
        self.classify.cuda()
        self.classify.train()
        for images, targets in self.tr_dataloader:
            images = images.cuda()
            targets = (targets - 10 * t).cuda()
            # Forward current model
            offset1, offset2 = compute_offsets(t, 10)
            with torch.no_grad():
                features = self.model(images)
            outputs = self.classify.forward(features, t)[:, offset1:offset2]
            loss = self.ce(outputs, targets)
            ## 根据这个损失计算梯度，变换此梯度
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return
    def new_train(self,t,epoch,endEpoch=False):
        self.optimizer = self._get_optimizer()
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        distill = AverageMeter()

        end = time.time()
        for idx, (images, labels) in enumerate(self.tr_dataloader):
            data_time.update(time.time() - end)

            if torch.cuda.is_available():
                images2 = images.detach()
                images = torch.cat([images, images2], dim=0)
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            with torch.no_grad():
                prev_task_mask = labels < t * self.num_classes
                prev_task_mask = prev_task_mask.repeat(2)

            # warm-up learning rate

            # compute loss
            features, encoded = self.model(images, return_feat=True)

            # IRD (current)
            if t > 0:
                features1_prev_task = features

                features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T),
                                          self.current_temp)
                logits_mask = torch.scatter(
                    torch.ones_like(features1_sim),
                    1,
                    torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                    0
                )
                logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
                features1_sim = features1_sim - logits_max1.detach()
                row_size = features1_sim.size(0)
                logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
                    features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

            # Asym SupCon
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = self.criterion(features, labels, target_labels=list(
                range(t * self.num_classes, (t + 1) * self.num_classes)))

            # IRD (past)
            if t> 0:
                with torch.no_grad():
                    features2_prev_task = self.pre_model(images)

                    features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T),
                                              self.past_temp)
                    logits_max2, _ = torch.max(features2_sim * logits_mask, dim=1, keepdim=True)
                    features2_sim = features2_sim - logits_max2.detach()
                    logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(
                        features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

                loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
                loss += self.distill_power * loss_distill
                distill.update(loss_distill.item(), bsz)

            # update metric
            losses.update(loss.item(), bsz)

            # SGD
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if endEpoch and idx + 1 == len(self.tr_dataloader):
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f} {distill.avg:.3f})'.format(
                    epoch, idx + 1, len(self.tr_dataloader), batch_time=batch_time,
                    data_time=data_time, loss=losses, distill=distill))

        return losses.avg
    def train_epoch(self,t):
        self.model.train()
        for images,targets in self.tr_dataloader:
            images = images.cuda()
            targets = (targets - 10 * t).cuda()
            # Forward current model
            offset1, offset2 = compute_offsets(t, 10)
            outputs = self.model.forward(images,t)[:,offset1:offset2]
            loss = self.ce(outputs, targets)
            ## 根据这个损失计算梯度，变换此梯度
            # Backward
            self.optimizer.zero_grad()
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

                features = self.model(images, return_feat=False)
                output = self.classify.forward(features, t)[:, offset1:offset2]
                loss = self.ce(output, targets)
                _, pred = output.max(1)
                hits = (pred == targets).float()

                # Log
                total_loss += loss.data.cpu().numpy() * len(images)
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(images)

        return total_loss / total_num, total_acc / total_num
def LongLifeTrain(args, appr, aggNum, writer,idx,is_train=True):
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
    if is_train:
        loss = appr.train(task)
        return appr.model.state_dict(), loss, 0
    else:
        loss = appr.tune(task)
        return appr.classify.state_dict(), loss, 0
    print('-' * 100)

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