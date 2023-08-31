import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.module import Module


class SparseCompressConvolution(Module):
    '''
    this is the sparse compress convolution
    with side edge conv
    feat conv and edge conv are decoupled
    '''
    def __init__(self, d_ein, d_in, d_model, filter_size, stride, dropout):
        super(SparseCompressConvolution, self).__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.sparseConv = nn.Linear(filter_size * d_in + filter_size * (filter_size - 1) * d_ein, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.d_in = d_in
        self.d_model = d_model


    def forward(self, adj, features, edge_features):
        if edge_features is not None:
            n = edge_features.shape[2]
            if self.stride > 1:
                adj = adj.unfold(3, self.filter_size, self.stride)
                adj = adj.unfold(4, self.filter_size, self.stride)
                adj = adj.permute(0, 1, 2, 3, 4, 5, 6).flatten(-2, -1)
                adj = adj.max(-1).values
            else:
                upper = adj.triu(self.filter_size)[:, :, :, :n, self.filter_size - 1:]
                lower = adj.tril(-self.filter_size)[:, :, :, self.filter_size - 1:, :n]
                adj = upper + lower

        features = features.flatten(-2, -1) # (b_s, h, n', d_model * self.filter_size)
        feat = torch.cat((features, edge_features), -1)
        output = self.sparseConv(feat)
        output = F.relu(output)
        output = self.dropout(output)
        return adj, output


class SparseUncompressConvolution(nn.Module):
    def __init__(self, d_model, filter_size, stride, upper_pad, dropout, res=1):
        super(SparseUncompressConvolution, self).__init__()
        self.sparseTConv = nn.Linear(d_model * filter_size, d_model)
        self.register_buffer('kron_kernel', torch.zeros(stride))
        self.kron_kernel[0] = 1
        self.padding = nn.ConstantPad1d(filter_size - upper_pad - 1, 0)
        self.dropout = nn.Dropout(p=dropout)
        self.filter_size = filter_size
        self.stride = stride
        self.d_model = d_model
        self.res = res


    def forward(self, layer_query, uncompress_in, i):
        b_s, h, n = layer_query.shape[:3]
        transpose_style = torch.einsum('ijac, b->ijcab', uncompress_in, self.kron_kernel)
        transpose_style = transpose_style.flatten(-2, -1)
        transpose_style = self.padding(transpose_style)
        up_sample = transpose_style.unfold(-1, self.filter_size, 1)
        up_sample = up_sample[:, :, :, :n, :].permute(0, 1, 3, 4, 2) # (b_s, h, n, filter_size, d_model)


        res_connect = up_sample.sum(3) / self.filter_size
        up_sample = up_sample.flatten(-2, -1) # (b_s, h, n, filter_size * d_model)
        uncompress_out = self.sparseTConv(up_sample)
        uncompress_out = F.relu(uncompress_out)
        uncompress_out = self.dropout(uncompress_out)

        output = layer_query + uncompress_out + res_connect
        return output
