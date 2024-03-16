import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
from model.sparse_utils import sparse_padding_square, sparse_max_pool


class SparseCompressConvolution(Module):
    '''
    this is the sparse ver. res compress convolution
    with side edge conv
    feat conv and edge conv are coupled
    '''
    def __init__(self, d_ein, d_in, d_model, filter_size, stride, dropout, app_node_feat=False, res=True):
        super(SparseCompressConvolution, self).__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.d_in = d_in
        self.d_model = d_model
        self.d_ein = d_ein
        self.CoConv = nn.Linear(filter_size * d_in + filter_size * (filter_size - 1) * d_ein, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.res = res

    def _padding(self, spatial_size):
        below_pad = spatial_size % self.stride
        below_pad = self.stride if below_pad == 0 else below_pad
        below_pad = self.filter_size - below_pad
        return below_pad

    def _conv(self, features, edge_features):
        if features is None:
            feat = edge_features
        else:
            features = features.flatten(-2, -1)
            feat = torch.concat((features, edge_features), -1)
        output = self.CoConv(feat)
        # output = self.layer_norm(output)
        output = self.dropout(F.relu(output))
        return output

    def forward(self, adj, features, edge_features, diag_unfolder, i=None, mask=None):
        if features is None:
            jump_features = torch.tensor([0], dtype=torch.float32, device=adj.device)
        else:
            below_pad = self._padding(features.shape[-2])
            features = F.pad(features.transpose(-2, -1), (0, below_pad), "constant", 0).transpose(-2, -1)
            if mask is not None:
                mask = F.pad(mask.transpose(-2, -1), (0, below_pad), "constant", 0).transpose(-2, -1)
                mask = mask.unfold(-2, self.filter_size, self.stride)
                mask = mask.max(-1).values
            features = features.unfold(-2, self.filter_size, self.stride) # (b_s * h, n', d_model, self.filter_size)
            jump_features = features.sum(-1) / self.filter_size
        if edge_features is None:
            adj = sparse_padding_square(adj, below_pad)
            edge_features = diag_unfolder(adj, self.filter_size, self.stride)
        if self.stride > 1:
            adj = sparse_max_pool(adj, self.filter_size, self.stride)
        output = self._conv(features, edge_features)
        if self.res:
            res_output = output + jump_features
        else:
            res_output = output
        if mask is not None:
            output = mask * output.unflatten(0, (-1, mask.shape[0]))
            output = output.flatten(0, 1)
            res_output = mask * res_output.unflatten(0, (-1, mask.shape[0]))
            res_output = res_output.flatten(0, 1)
        return adj, res_output, output, mask


class CompressConvolution(Module):
    '''
    this is the dense ver. res compress convolution
    with side edge conv
    feat conv and edge conv are coupled
    '''
    def __init__(self, d_ein, d_in, d_model, filter_size, stride, dropout, app_node_feat=False, res=True):
        super(CompressConvolution, self).__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.d_in = d_in
        self.d_model = d_model
        self.d_ein = d_ein
        self.CoConv = nn.Linear(filter_size * d_in + filter_size * (filter_size - 1) * d_ein, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(d_model)
        self.res = res


    def _padding(self, spatial_size):
        below_pad = spatial_size % self.stride
        below_pad = self.stride if below_pad == 0 else below_pad
        below_pad = self.filter_size - below_pad
        return below_pad
        

    def _conv(self, features, edge_features):
        if features is None:
            feat = edge_features
        else:
            features = features.flatten(-2, -1)
            feat = torch.cat((features, edge_features), -1)
        output = self.CoConv(feat)
        output = self.dropout(F.relu(output))
        return output


    def forward(self, adj, features, edge_features, diag_unfolder, i=None, mask=None):
        if features is None:
            jump_features = torch.tensor([0], dtype=torch.float32, device=adj.device)
        else:
            below_pad = self._padding(features.shape[2])
            features = F.pad(features.transpose(-2, -1), (0, below_pad), "constant", 0).transpose(-2, -1)
            if mask is not None:
                mask = F.pad(mask.transpose(-2, -1), (0, below_pad), "constant", 0).transpose(-2, -1)
                mask = mask.unfold(2, self.filter_size, self.stride)
                mask = mask.max(-1).values
            features = features.unfold(2, self.filter_size, self.stride) # (b_s, h, n', d_model, self.filter_size)
            jump_features = features.sum(-1) / self.filter_size
        if edge_features is None:
            adj = F.pad(adj, (0, below_pad, 0, below_pad), "constant", 0)
            edge_features = diag_unfolder(adj, self.filter_size, self.stride)
        if self.stride > 1:
            adj = adj.unfold(3, self.filter_size, self.stride)
            adj = adj.unfold(4, self.filter_size, self.stride)
            adj = adj.permute(0, 1, 2, 3, 4, 5, 6).flatten(-2, -1)
            adj = adj.max(-1).values
        output = self._conv(features, edge_features)
        if self.res:
            res_output = output + jump_features
        else:
            res_output = output
        if mask is not None:
            jump_features = mask * jump_features
            res_output = mask * res_output
        return adj, res_output, jump_features, mask


class UncompressConvolution(nn.Module):
    def __init__(self, d_ein, d_in, d_model, filter_size, stride, upper_pad, dropout, res=1):
        super(UncompressConvolution, self).__init__()
        self.register_buffer('kron_kernel', torch.zeros(stride))
        self.kron_kernel[0] = 1
        self.padding = nn.ConstantPad1d(filter_size - upper_pad - 1, 0)
        self.dropout = nn.Dropout(p=dropout)
        self.filter_size = filter_size
        self.stride = stride
        self.d_model = d_model
        self.res = res
        self.TCoConv = nn.Linear(d_in * filter_size, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, layer_query, uncompress_in, i):
        b_s, h, n = layer_query.shape[:3]
        # dilation
        transpose_style = torch.einsum('ijac, b->ijcab', uncompress_in, self.kron_kernel)
        transpose_style = transpose_style.flatten(-2, -1)
        transpose_style = self.padding(transpose_style)
        up_sample = transpose_style.unfold(-1, self.filter_size, 1)
        up_sample = up_sample[:, :, :, :n].permute(0, 1, 3, 4, 2) # (b_s, h, n, nf, d_model)
        # conv
        up_sample_in = up_sample.flatten(-2, -1)
        uncompress_out = self.TCoConv(up_sample_in)
        uncompress_out = self.dropout(F.relu(uncompress_out))
        up_sample = up_sample.sum(-2) / (self.filter_size / self.stride)
        if self.res:
            output = up_sample - uncompress_out + layer_query
        else:
            output = uncompress_out
        return output