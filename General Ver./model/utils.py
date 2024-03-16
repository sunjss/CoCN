import math
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch_geometric.utils import to_dense_adj, to_networkx, scatter, to_torch_coo_tensor, \
                                  remove_self_loops, add_remaining_self_loops, add_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
import networkx as nx
import matplotlib.pyplot as plt

class Htanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, epsilon=10):
        ctx.save_for_backward(x.data, torch.tensor(epsilon))
        y = torch.where((x > 0), torch.ones_like(x), x)
        y = torch.where((y < 0), -torch.ones_like(x), x)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, epsilon = ctx.saved_tensors
        y = (torch.exp(epsilon * x) - torch.exp(- epsilon * x)) / (torch.exp(epsilon * x) + torch.exp(- epsilon * x))
        dx = dy * epsilon * (1 - y ** 2)
        return dx, None


class Hrelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rx, epsilon=1):
        ctx.save_for_backward((x/ (rx + 1)).data , torch.tensor(epsilon))
        return rx

    @staticmethod
    def backward(ctx, dy):
        x, epsilon = ctx.saved_tensors
        # dx = torch.where((x > 0), dy, torch.zeros_like(x))
        y = 1 / (1 + torch.exp(- epsilon * x))
        dx = dy * epsilon * y * (1 - y)
        return dx, None


class Hsigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, epsilon=4):
        ctx.save_for_backward(x.data, torch.tensor(epsilon))
        y = torch.where((x > 0), torch.ones_like(x), torch.zeros_like(x))
        return y

    @staticmethod
    def backward(ctx, dy):
        x, epsilon = ctx.saved_tensors
        y = 1 / (1 + torch.exp(- epsilon * x))
        dx = dy * epsilon * y * (1 - y)
        # dx = dy * epsilon
        return dx, None


class DiagUnfolder(Module):
    def __init__(self, filter_size):
        super(DiagUnfolder, self).__init__()
        self.register_buffer('idx', torch.triu_indices(filter_size, filter_size, 1))
        self.idx = torch.concat((self.idx, self.idx[[1, 0]]), -1)
        # self.register_buffer('idxx', torch.arange(0, filter_size))
        # self.register_buffer('x', self.idxx.view(1, -1, 1).expand(1, filter_size, filter_size))
        # self.register_buffer('remove_idx', (self.x != self.x.permute(0, 2, 1)).nonzero())
    
    def forward(self, adj, filter_size, stride):
        n = adj.shape[-2]
        idx_adder = torch.arange(0, n - filter_size + 1, stride, device=adj.device)
        idx_adder = idx_adder.view(-1, 1, 1)
        # x = self.x + idx_adder
        # y = x[:, self.remove_idx[:, 2], self.remove_idx[:, 1]]
        # x = x[:, self.remove_idx[:, 1], self.remove_idx[:, 2]]
        idx = self.idx + idx_adder
        out = adj[:, :, :, idx[:, 0, :], idx[:, 1, :]]
        out = out.permute(0, 1, 3, 2, 4).flatten(-2, -1)
        return out


class BiasDiagUnfolder(Module):
    def __init__(self, filter_size, bias=None, bias_step=None):
        super(BiasDiagUnfolder, self).__init__()
        self.register_buffer('idx', torch.triu_indices(filter_size, filter_size, 1))
        if bias is not None:
            if bias_step is None:
                bias_step = filter_size
            self.register_buffer('bias', torch.arange(bias_step - 1, (bias_step - 1) * bias, bias_step - 1))
            y = self.idx[1, :].view(1, 1, -1) + self.bias.view(1, -1, 1)  # (1, nbias, nhk), nhk = n_half_kernel
            x = self.idx[0, :].view(1, 1, -1).expand(-1, self.bias.numel(), -1)
            self.idx = torch.concat((x, y), 0).flatten(-2, -1) # (2, nbias * nhk)
        self.idx = torch.concat((self.idx, self.idx[[1, 0]]), -1)
    
    def forward(self, adj, filter_size, stride, n=None):
        if n is None:
            n = adj.shape[-1]
        idx_adder = torch.arange(0, n - filter_size + 1, stride, device=adj.device)
        idx_adder = idx_adder.view(-1, 1, 1)
        idx = self.idx + idx_adder
        assert idx.max() < adj.shape[-1]
        out = adj[:, :, :, idx[:, 0, :], idx[:, 1, :]]
        out = out.permute(0, 1, 3, 2, 4).flatten(-2, -1)
        return out


def get_normalized_adj(edge_index, 
                       whole_size, 
                       max_num_nodes, 
                       edge_weight=None, 
                       batch=None, 
                       selfloop=False, 
                       ori_edge=False, 
                       norm_type="sym", 
                       dense=True):
    if selfloop and not ori_edge:
        edge_index = add_remaining_self_loops(edge_index, num_nodes=whole_size)[0]        
    elif not selfloop and not ori_edge:
        edge_index = remove_self_loops(edge_index)[0]
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    if edge_weight.dtype == torch.long:
        edge_weight = edge_weight.type(torch.float32)

    # normalization
    row, col = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, row, 0, dim_size=maybe_num_nodes(edge_index), reduce='sum')
    if norm_type == "sym":
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col] 
    elif norm_type == "sym-lp":
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # L = I - A_norm.
        edge_index, tmp = add_self_loops(edge_index, -edge_weight,
                                         fill_value=1., num_nodes=max_num_nodes)
        assert tmp is not None
        edge_weight = tmp
    else:
        deg_inv_sqrt = deg.pow_(-1)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = edge_weight * deg_inv_sqrt[col]    
    
    # batch based re-index
    if batch is None:
        num_nodes = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
        batch = edge_index.new_zeros(num_nodes)

    batch_size = int(batch.max()) + 1 if batch.numel() > 0 else 1

    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='sum')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if ((idx1.numel() > 0 and idx1.max() >= max_num_nodes)
          or (idx2.numel() > 0 and idx2.max() >= max_num_nodes)):
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_weight = edge_weight[mask]
    
    if dense:
        size = [batch_size, max_num_nodes, max_num_nodes]
        size += list(edge_weight.size())[1:]
        flattened_size = batch_size * max_num_nodes * max_num_nodes
        idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2

        adj = scatter(edge_weight, idx, dim=0, dim_size=flattened_size, reduce='sum')
        return adj.view(size)
    else:
        idx = torch.stack((idx1, idx2), 0)
        idx = idx0 * max_num_nodes + idx
        size = max_num_nodes if batch is None else (batch.max() + 1) * max_num_nodes
        return torch.sparse_coo_tensor(idx, edge_weight, size=(size, size))