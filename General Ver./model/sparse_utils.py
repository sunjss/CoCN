import math
import paddle
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch_geometric.utils import scatter
from mmcv.ops.sparse_functional import indice_maxpool
from mmcv.ops.sparse_ops import get_indice_pairs
from mmcv.ops.sparse_structure import SparseConvTensor


class SparseBiasDiagUnfolder(Module):
    def __init__(self, filter_size, bias=None, bias_step=None):
        super(SparseBiasDiagUnfolder, self).__init__()
        self.register_buffer('idx', torch.triu_indices(filter_size, filter_size, 1))
        if bias is not None:
            if bias_step is None:
                bias_step = filter_size
            self.register_buffer('bias', torch.arange(bias_step - 1, (bias_step - 1) * bias, bias_step - 1))
            y = self.idx[1, :].view(1, 1, -1) + self.bias.view(1, -1, 1)  # (1, nbias, nhk), nhk = n_half_kernel
            x = self.idx[0, :].view(1, 1, -1).expand(-1, self.bias.numel(), -1)
            self.idx = torch.concat((x, y), 0).flatten(-2, -1) # (2, nbias * nhk)
        else:
            self.bias = None
        self.idx = torch.concat((self.idx, self.idx[[1, 0]]), -1)
    
    def forward(self, adj, filter_size, stride, n=None):
        # adj (b_s * h, n, n, d_ein)
        if n is None:
            n = adj.shape[-2]
        idx_adder = torch.arange(0, n - filter_size + 1, stride, device=adj.device)
        idx = self.idx + idx_adder.view(-1, 1, 1)
        assert idx.max() < adj.shape[-2]

        batch_size = idx_adder.numel() * self.bias.numel() if self.bias is not None else idx_adder.numel()
        batch_index = torch.arange(batch_size, device=adj.device)
        batch_index = batch_index.unsqueeze(-1).expand(-1, filter_size ** 2 - filter_size)
        mask_edge_index = idx.transpose(0, 1).flatten(1, 2)
        mask_edge_index = mask_edge_index[0] * adj.shape[-2] + mask_edge_index[1]
        mask_index = torch.stack((batch_index.flatten(), mask_edge_index), 0)

        mask_values = idx.unflatten(-1, (-1, filter_size ** 2 - filter_size))
        mask_values[:, 1] -= 1
        mask_values = mask_values.select(-1, 0).unsqueeze(-1)
        mask_values = mask_values.expand(-1, -1, -1, filter_size ** 2 - filter_size)
        mask_values = mask_values.permute(0, 2, 3, 1).reshape(-1, 2)
        sparse_mask = torch.sparse_coo_tensor(mask_index, mask_values, 
                                              (batch_size, adj.shape[-2] ** 2, 2))
        
        adj = adj.coalesce()
        index = adj.indices()
        query_index = index[1] * adj.shape[-2] + index[2]
        sparse_mask = sparse_mask.index_select(1, query_index).coalesce()
        mask = sparse_mask.indices()
        index = index[:, mask[1]]
        edge_index = index[1:] - sparse_mask.values().T
        edge_index = edge_index[0] * filter_size + edge_index[1]
        edge_index = edge_index - 1 - edge_index // (filter_size + 1)
        index = torch.stack((index[0], mask[0], edge_index), 0)
        values = adj.values()[mask[1]]
        if index.shape[-1] == 0:
            index = torch.zeros(3, 1, device=adj.device)
            values = torch.zeros(1, 1, device=adj.device)
        out = torch.sparse_coo_tensor(index, values, 
                                      (adj.shape[0], batch_size, 
                                       filter_size ** 2 - filter_size, adj.shape[-1]))
        out = out.to_dense()
        return out.view(adj.shape[0], idx_adder.numel(), -1)


def sparse_average_pool(mx, filter_size, stride):
    # (b_s, h, w, c)
    assert stride > 1, "Invalid pooling stride, requires stride > 1."
    b_s, h, w, c = mx.shape
    h_new, w_new = (h - filter_size) // stride + 1, (w - filter_size) // stride + 1
    device = mx.device
    mx = mx.coalesce()
    index = mx.indices()
    kernel_bias = torch.arange(filter_size, device=device)
    x_bias = (index[1] - kernel_bias.unsqueeze(1) + (1 + stride)) % stride  # (k, h)
    y_bias = (index[2] - kernel_bias.unsqueeze(1) + (1 + stride)) % stride  # (k, w)
    x_kernel_call = (index[1] - kernel_bias.unsqueeze(1)) // stride  # (k, h)
    y_kernel_call = (index[2] - kernel_bias.unsqueeze(1)) // stride  # (k, w)
    mask = ((x_bias == 1) & (y_bias == 1)).view(-1)

    index = torch.stack((torch.stack((index[0],) * filter_size, 0), x_kernel_call, y_kernel_call), 0)
    index = index.flatten(1, 2)
    mx = torch.sparse_coo_tensor(index[:, mask], mx.values()[mask], (b_s, h_new, w_new, c))
    return mx.coalesce() / (filter_size ** 2)


class SparsePerm():
    def __init__(self, indices, values, size):
        self.i = indices.type(torch.int64)
        self.v = values
        self.shape = size
    
    def indices(self):
        return self.i
    
    def values(self):
        return self.v
    
    def transpose(self, i, j):
        first = self.i[i]
        self.i[i] = self.i[j]
        self.i[j] = first
        return self

        
def sparse_max_pool(mx, filter_size, stride):
    assert stride > 1, "Invalid pooling stride, requires stride > 1."
    b_s, h, w, c = mx.shape
    h_new, w_new = (h - filter_size) // stride + 1, (w - filter_size) // stride + 1
    kernel_size = [filter_size, filter_size]
    stride = [stride, stride]
    padding = [0, 0]
    dilation = [1, 1]
    mx = mx.coalesce()

    outids, \
    indice_pairs, \
    indice_pairs_num = get_indice_pairs(mx.indices().T.contiguous().type(torch.int32), 
                                        b_s, [h, w], 
                                        kernel_size, stride,
                                        padding, dilation)

    out_features = indice_maxpool(mx.values(), indice_pairs.to(mx.device),
                                  indice_pairs_num.to(mx.device),
                                  outids.shape[0])
    mx = SparseConvTensor(out_features, outids, [h_new, w_new], b_s)
    index = mx.indices.T
    return torch.sparse_coo_tensor(index, mx.features, (b_s, h_new, w_new, c))


def sparse_padding_square(mx, padding):
    b, h, w, c = mx.shape
    mx = mx.coalesce()
    return torch.sparse_coo_tensor(mx.indices(), mx.values(),
                                   size=(b, h + padding, w + padding, c))


def to_list_batch_sparse_adj(edge_index, batch=None, edge_attr=None, max_num_nodes=None, batch_size=None):
    if batch is None:
        num_nodes = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
        batch = edge_index.new_zeros(num_nodes)

    if batch_size is None:
        batch_size = int(batch.max()) + 1 if batch.numel() > 0 else 1

    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='sum')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())

    elif ((idx1.numel() > 0 and idx1.max() >= max_num_nodes)
          or (idx2.numel() > 0 and idx2.max() >= max_num_nodes)):
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = torch.ones((idx0.numel(), 1), device=edge_index.device)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.size())[1:]

    idx = torch.stack((idx0, idx1, idx2), 0)
    adj = torch.sparse_coo_tensor(idx, edge_attr, size, dtype=torch.float32)
    return adj