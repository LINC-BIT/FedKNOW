#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: n")
    parser.add_argument('--shard_per_user', type=int, default=5, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.4, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=6, help="the number  of local epochs: E")
    parser.add_argument('--local_test_bs', type=int, default=30, help="the number  of local test epochs: E")
    parser.add_argument('--local_bs', type=int, default=20, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=100, help="test batch size")
    parser.add_argument('--optim', type=str, default='Adam', help="optimizer")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")
    parser.add_argument('--local_updates', type=int, default=1000000, help="maximum number of local updates")
    parser.add_argument('--m_tr', type=int, default=240, help="maximum number of samples/user to use for training")
    parser.add_argument('--m_ft', type=int, default=240, help="maximum number of samples/user to use for fine-tuning")

    # model arguments
    parser.add_argument('--model', type=str, default='6layer_CNN', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_layers_keep', type=int, default=1, help='number layers to keep')
    parser.add_argument('--alg', type=str, default='fedrep', help='FL algorithm to use')
    
    # algorithm-specific hyperparameters
    parser.add_argument('--local_local_ep', type=int, default=2, help="the number of local epochs for the representation for FedRep")
    parser.add_argument('--lr_g', type=float, default=0.1, help="global learning rate for SCAFFOLD")
    parser.add_argument('--mu', type=float, default='0.1', help='FedProx parameter mu')
    parser.add_argument('--gmf', type=float, default='0', help='FedProx parameter gmf')
    parser.add_argument('--alpha_apfl', type=float, default='0.75', help='APFL parameter alpha')
    parser.add_argument('--alpha_l2gd', type=float, default='1', help='L2GD parameter alpha')
    parser.add_argument('--lambda_l2gd', type=float, default='0.5', help='L2GD parameter lambda')
    parser.add_argument('--lr_in', type=float, default='0.001', help='PerFedAvg inner loop step size')
    parser.add_argument('--bs_frac_in', type=float, default='0.8', help='PerFedAvg fraction of batch used for inner update')
    parser.add_argument('--lam_ditto', type=float, default='1', help='Ditto parameter lambda')
    parser.add_argument('--store_rate', type=float, default=0.1,
                        help='the store rate of model in FedKNOW')
    parser.add_argument('--select_grad_num', type=int, default=10,
                        help='the store rate of model in FedKNOW')
    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar100', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=100, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--test_freq', type=int, default=50, help='how often to test on val set')
    parser.add_argument('--load_fed', type=str, default='n', help='define pretrained federated model path')
    parser.add_argument('--results_save', type=str, default='runA', help='define fed results save folder')
    parser.add_argument('--save_every', type=int, default=50, help='how often to save models')
    parser.add_argument('--round', type=int, default=5, help='train number of the per-task')
    parser.add_argument('--task', type=int, default=10, help='train number of task')
    parser.add_argument('--lamb', default=0, type=float, help='(default=%(default)f)')
    parser.add_argument('--n_memories', type=int, default=10,
                        help='number of memories per task')
    parser.add_argument('--memory_strength', default=0.5, type=float,
                        help='memory strength (meaning depends on memory)')
    parser.add_argument('--pFedMelamda', default=15, type=float, help='(default=%(default)f)')
    parser.add_argument('--pFedMelr', default=0.09, type=float, help='(default=%(default)f)')
    parser.add_argument('--pFedMeK', default=5, type=int, help='(default=%(default)f)')
    parser.add_argument('--APFLalpha', default=0.5, type=float, help='(default=%(default)f)')
    parser.add_argument('--AMPalpha', default=0.5, type=float, help='(default=%(default)f)')
    parser.add_argument('--AMPlamb', default=0.5, type=float, help='(default=%(default)f)')
    parser.add_argument('--BCNxstep', type=int, default=5,
                        help='Number of epochs per task')
    parser.add_argument('--BCNthetaBstep', type=int, default=5,
                        help='Number of epochs per task')
    parser.add_argument('--BCNbeta', type=int, default=2,
                        help='Number of epochs per task')
    parser.add_argument('--Co2Lis_train', type=bool, default=False,
                        help='Number of epochs per task')
    parser.add_argument('--client_id', type=int, default=0,
                        help='client ID')

    parser.add_argument('--ip', type=str, default=0,
                        help='ip address')
    args = parser.parse_args()
    return args
