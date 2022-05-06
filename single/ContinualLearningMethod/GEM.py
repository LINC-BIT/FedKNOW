# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim

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
    try:
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        gradient.copy_(torch.Tensor(x).view(-1, 1))
    except ValueError:
        print('无法求解')
        gradient.copy_(torch.Tensor(gradient_np).view(-1, 1))



class Appr():
    def __init__(self,
                 model,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Appr, self).__init__()
        self.margin = args.memory_strength
        self.is_cifar = True
        self.net = model
        # if self.is_cifar:
        #     self.net = ResNet18(n_outputs)
        # else:
        #     self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.Adam(self.net.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.gpu = True
        self.lr = args.lr
        # allocate episodic memory
        self.memory_data = torch.FloatTensor(
            n_tasks, self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        if self.gpu:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.net.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if self.gpu:
            self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        if self.is_cifar:
            self.nc_per_task = int(n_outputs / n_tasks)
        else:
            self.nc_per_task = n_outputs
    def set_model(self,model):
        self.net = model

    def observe(self, x, t, y):
        self.opt = optim.Adam(self.net.parameters(), self.lr)
        # update memory
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        self.memory_data[t, self.mem_cnt: endcnt].copy_(
            x.data[: effbsz].view(effbsz,-1))
        if bsz == 1:
            self.memory_labs[t, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.net.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]

                offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                                            self.is_cifar)
                output = self.net(
                        self.memory_data[past_task],
                        past_task)[:, offset1: offset2]
                ptloss = self.ce(output,self.memory_labs[past_task] - offset1)
                ptloss.backward()
                store_grad(self.net.parameters, self.grads, self.grad_dims,
                           past_task)

        # now compute the grad on the current minibatch
        self.net.zero_grad()

        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        loss = self.ce(self.net.forward(x, t)[:, offset1: offset2], y - offset1)
        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.net.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                else torch.LongTensor(self.observed_tasks[:-1])
            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                            self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1),
                              self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                overwrite_grad(self.net.parameters, self.grads[:, t],
                               self.grad_dims)
        self.opt.step()
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
                output = self.net.forward(images,t)

                loss = self.ce(output[:, offset1: offset2], targets - offset1)
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
        for images,targets in tr_dataloader:
            images = images.cuda()
            targets = targets.cuda()
            appr.net.train()
            appr.observe(images, task, targets)
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
    # Test
    # for u in range(t + 1):
    #     xtest = testdatas[u][0].cuda()
    #     ytest = (testdatas[u][1]).cuda()
    #     test_loss, test_acc = appr.validTest(u, xtest, ytest)
    #     print('>>> Test on task {:2d} : loss={:.3f}, acc={:5.1f}% <<<'.format(u, test_loss,
    #                                                                           100 * test_acc))
    #     acc[t, u] = test_acc
    #     lss[t, u] = test_loss
    #
    # # Save
    #
    # print('Average accuracy={:5.1f}%'.format(100 * np.mean(acc[t, :t + 1])))
    # if r == args.round - 1:
    #     writer.add_scalar('task_finish_not_agg', np.mean(acc[t, :t + 1]), t + 1)

    # save_path = args.output + '/aggNum' + str(aggNum)
    # print('Save at ' + save_path)
    # if not os.path.isdir(save_path):
    #     os.makedirs(save_path)
    # np.savetxt(save_path + '/' + args.log_name, acc, '%.4f')


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