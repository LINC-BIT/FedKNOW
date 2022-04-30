import math
from torch import nn
import torch.nn.functional as F
import collections.abc
import numpy as np
from torch.nn import init

container_abcs = collections.abc
from torch.nn.parameter import Parameter
import torch
# from torch.nn.modules import init
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))
_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
class DecomposedConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super().__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups
        self.padding_mode = padding_mode
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        self.sw = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        self.mask = Parameter(torch.Tensor(
                out_channels))
        self.aw = Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.sw, a=math.sqrt(5))
        init.kaiming_uniform_(self.aw, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.sw)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            init.uniform_(self.mask, -bound, bound)
    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    def set_atten(self,t,dim):
        if t==0:
            self.atten = Parameter(torch.zeros(dim))
        else:
            self.atten = Parameter(torch.rand(dim))
    def set_knlwledge(self,from_kb):
        self.from_kb = from_kb
        self.from_kb.cuda()
    def get_weight(self):
        m = nn.Sigmoid()
        sw = self.sw.transpose(0, -1)
        # newmask = m(self.mask)
        # print(sw*newmask)
        weight = (sw * m(self.mask)).transpose(0, -1) + self.aw + torch.sum(self.atten * (self.from_kb.cuda()), dim=-1)
        weight = weight.type(torch.cuda.FloatTensor)
        return weight
    def forward(self, input):

        # newmask = m(self.mask)
        # print(sw*newmask)
        weight = self.get_weight()
        return self._conv_forward(input, weight)

class DecomposedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,bias: bool = True) -> None:
        super(DecomposedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sw = Parameter(torch.Tensor(out_features, in_features))
        self.mask = Parameter(torch.Tensor(out_features))
        self.aw = Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.sw, a=math.sqrt(5))
        init.kaiming_uniform_(self.aw, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.sw)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            init.uniform_(self.mask, -bound, bound)
    def set_atten(self,t,dim):
        if t==0:
            self.atten = Parameter(torch.zeros(dim))
            self.atten.requires_grad=False
        else:
            self.atten = Parameter(torch.rand(dim))
    def set_knlwledge(self,from_kb):
        self.from_kb = from_kb

    def get_weight(self):
        m = nn.Sigmoid()
        sw = self.sw.transpose(0, -1)
        # newmask = m(self.mask)
        # print(sw*newmask)
        weight = (sw * m(self.mask)).transpose(0, -1) + self.aw + torch.sum(self.atten * self.from_kb.cuda(), dim=-1)
        weight = weight.type(torch.cuda.FloatTensor)
        return weight
    def forward(self, input):
        weight = self.get_weight()
        return F.linear(input, weight, self.bias)