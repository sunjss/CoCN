import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import get_laplacian, to_dense_adj, add_remaining_self_loops, to_dense_batch

from model.utils import Hsigmoid, diag_unfold
from model.layers import SparseCompressConvolution, SparseUncompressConvolution

class PermGenModule(nn.Module):
    def __init__(self, d_model, temp):
        super(PermGenModule, self).__init__()
        self.hsigmoid = Hsigmoid()
        self.d_model = d_model
        self.temp = temp
        
    def forward(self, ranking, mask=None):
        device = ranking.device
        b_s, h, node_num = ranking.shape[:3]
        
        ranking_bias = torch.randn_like(ranking) * 1e-5
        ranking = ranking + ranking_bias
        relative_rating = ranking - ranking.permute(0, 1, 3, 2)
        relative_rating_bias = torch.where((relative_rating==0), torch.ones_like(relative_rating), torch.zeros_like(relative_rating))
        relative_rating_bias = torch.triu(relative_rating_bias, 1)
        relative_rating = relative_rating + relative_rating_bias

        relative_rating = F.relu(relative_rating)
        relative_rating = self.hsigmoid.apply(relative_rating)
        if mask is not None:
            mask = mask.view(b_s, 1, node_num, 1)
            relative_rating = mask * relative_rating
        ranking = relative_rating.sum(-2, keepdim=True) # (b_s, h, 1, n)

        indicator = torch.arange(0, node_num, device=device, dtype=torch.float32).view(-1, 1) # (n, 1)
        indicator = indicator - ranking
        indicator = torch.where(indicator < 0, indicator + node_num, indicator)
        perm = torch.exp(-indicator * self.temp)
        if mask is not None:
            perm = mask * perm * mask.transpose(-2, -1)
        return perm


class Ranker(nn.Module):
    def __init__(self, d_in, d_model, h, dropout, N_enc):
        super(Ranker, self).__init__()
        print("Using testing perm gen model")
        self.feat_ff = nn.Linear(d_in, d_model * h)
        self.ff_out = nn.Sequential(nn.Linear(d_model, d_model),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(d_model, 16),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(16, 2))
        self.dropout = nn.Dropout(dropout)
        self.h = h
        self.nlayers = N_enc
            
    def forward(self, x, Ahat):
        b_s, n = x.shape[:2]
        node_feat = self.feat_ff(x) # (n, d_model * h)
        node_feat = self.dropout(F.relu(node_feat))

        feat_out = node_feat.view(b_s, n, self.h, -1).permute(0, 2, 1, 3) # (b_s, h, n, d_model)

        out = self.ff_out(feat_out) # (b_s, h, n, 2)
        out = out.sum(-1, keepdim=True) / 2
        for i in range(self.nlayers):
            out = torch.matmul(Ahat.unsqueeze(1), out)
        return out


class MiniBatchCoCN(nn.Module):
    def __init__(self, ranking, perm_gen, conv):
        super(MiniBatchCoCN, self).__init__()
        self.ranking = ranking
        self.perm_gen = perm_gen
        self.conv = conv

    def forward(self, data):
        features = data.x
        if isinstance(data, Batch):
            features, mask = to_dense_batch(features, data.batch)
            adj = to_dense_adj(data.edge_index, data.batch)
        else:
            features = data.x
        if features.dtype == torch.long:
            features = features.type(torch.float32)
        edge_index_self_loop = add_remaining_self_loops(data.edge_index)[0]

        L = get_laplacian(edge_index_self_loop, normalization="sym")
        L = to_dense_adj(L[0], data.batch, L[1])
        I = torch.eye(features.shape[-2], device=features.device)
        Ahat = I - L
        adj = to_dense_adj(data.edge_index, data.batch, data.edge_attr)
        if data.num_edge_features > 1:
            adj = adj.permute(0, 3, 1, 2)
        else:
            adj = adj.unsqueeze(1)  # (b_s, d_ein, n, n)
        
        if features.ndim == 2:
            features = features.unsqueeze(0)
        ranking = self.ranking(features, Ahat)
        perm = self.perm_gen(ranking, mask)  # (b_s, h, n, n)
        adj = 0.99 * adj + 0.01 # {0.1,0.9}
        output = self.conv(perm, adj, features)
        
        return output, None


class CoCN(nn.Module):
    def __init__(self, ranking, perm_gen, conv):
        super(CoCN, self).__init__()
        self.ranking = ranking
        self.perm_gen = perm_gen
        self.conv = conv

    def forward(self, data):
        features = data.x
        if features.dtype == torch.long:
            features = features.type(torch.float32)
        edge_index_self_loop = add_remaining_self_loops(data.edge_index)[0]
        L = get_laplacian(edge_index_self_loop, normalization="sym")
        L = to_dense_adj(L[0], edge_attr=L[1])
        I = torch.eye(features.shape[-2], device=features.device)
        Ahat = I - L
        adj = to_dense_adj(data.edge_index)
        adj = adj.unsqueeze(1)  # (d_ein, n, n)
        
        if data.x.ndim == 2:
            features = features.unsqueeze(0)
        ranking = self.ranking(features, Ahat)
        perm = self.perm_gen(ranking)  # (b_s, h, n, n)
        adj = 0.99 * adj + 0.01 # {0.1,0.9}
        output = self.conv(perm, adj, features)
        
        if data.x.ndim == 2:
            output = output.squeeze(0)
        return output, None


class CoCNModuleG(nn.Module):
    def __init__(self, h, d_in, d_ein, d_model, nclass, filter_size, stride, nblocks, nlayers, dropout):
        super(CoCNModuleG, self).__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.d_model = d_model
        self.d_ein = d_ein
        self.ff_in = nn.Linear(d_in, d_model)
        self.compress_layers = nn.ModuleList([])
        PAD = 0
        self.padding_list = []
        for _ in range(nlayers):
            self.padding_list.append(PAD)
            self.compress_layers.append(SparseCompressConvolution(d_ein, d_model, d_model, filter_size, 1, dropout))
        for _ in range(nblocks):
            padding_size_adder = PAD if self.stride < self.filter_size else 0
            self.padding_list.append(padding_size_adder)
            self.compress_layers.append(SparseCompressConvolution(d_ein, d_model, d_model, filter_size, self.stride, dropout))
        self.padding_list.append(PAD)
        self.compress_layers.append(SparseCompressConvolution(d_ein, d_model, d_model, filter_size, 1, dropout))
        self.nlayers = len(self.compress_layers)
        self.ff_head = nn.Linear(h * d_model, d_model)
        self.ff_out = nn.Linear(d_model, nclass)
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, perm, adj, features):
        node_features = self.ff_in(features)  # (b_s, n, d_model)
        node_features = self.input_norm(node_features)
        node_features = self.dropout(F.relu(node_features))
        node_features = node_features.unsqueeze(1)
        features = torch.matmul(perm, node_features)  # (b_s, h, n, d_model)
        adj_perm = perm.unsqueeze(-3) # (b_s, h, 1, n, n)
        adj = adj.unsqueeze(1)
        adj = torch.matmul(adj_perm, torch.matmul(adj, adj_perm.transpose(-2, -1)))  # (b_s, h, d_ein, n, n)

        # init parameter
        spatial_size = perm.shape[2]
        for i in range(self.nlayers):
            this_stride = self.compress_layers[i].stride

            upper_pad = self.padding_list[i]
            below_pad = (upper_pad + spatial_size) % this_stride
            below_pad = this_stride if below_pad == 0 else below_pad
            below_pad = self.filter_size - below_pad

            # edge feature prepare
            adj = F.pad(adj, (upper_pad, below_pad, upper_pad, below_pad), "constant", 0)
            edge_features = diag_unfold(adj, self.filter_size, this_stride)

            # clique feature prepare
            features = F.pad(features.transpose(-2, -1), (upper_pad, below_pad), "constant", 0).transpose(-2, -1)
            features = features.unfold(2, self.filter_size, this_stride) # (b_s, h, n', d_model, self.filter_size)
            jump_features = features.sum(-1) / self.filter_size

            adj, res_features = self.compress_layers[i](adj, features, edge_features)
            features = res_features + jump_features
            spatial_size = features.shape[2]

        output = features.max(2).values
        output = output.flatten(-2, -1)
        output = self.ff_head(output)
        output = self.dropout(F.relu(output))
        output = self.ff_out(output) # (1, nclass)

        return F.log_softmax(output, dim=-1)


class CoCNModuleN(nn.Module):
    def __init__(self, h, d_in, d_ein, d_model, nclass, filter_size, stride, nlayers, nblocks, dropout):
        super(CoCNModuleN, self).__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.ff_in = nn.Linear(d_in, d_model)
        self.compress_layers = nn.ModuleList([])
        self.uncompress_layers = nn.ModuleList([])
        PAD = 0
        self.padding_list = []
        for _ in range(nlayers):
            self.padding_list.append(PAD)
            self.compress_layers.append(SparseCompressConvolution(d_ein, d_model, d_model, filter_size, 1, dropout))
            self.uncompress_layers.append(SparseUncompressConvolution(d_model, filter_size, 1, PAD, dropout))
        for _ in range(nblocks):
            padding_size_adder = PAD if self.stride < self.filter_size else 0
            self.padding_list.append(padding_size_adder)
            self.compress_layers.append(SparseCompressConvolution(d_ein, d_model, d_model, filter_size, self.stride, dropout))
            self.uncompress_layers.append(SparseUncompressConvolution(d_model, filter_size, self.stride, padding_size_adder, dropout))
        self.padding_list.append(PAD)
        self.compress_layers.append(SparseCompressConvolution(d_ein, d_model, d_model, filter_size, 1, dropout))
        self.uncompress_layers.append(SparseUncompressConvolution(d_model, filter_size, 1, PAD, dropout))

        self.nlayers = len(self.compress_layers)

        self.ff_head = nn.Linear(d_model * h, d_model)
        self.ff_out = nn.Linear(d_model, nclass)
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, perm, adj, features):
        # init parameter
        spatial_size = perm.shape[2]

        features = self.ff_in(features)  # (b_s, n, d_model)
        features = self.input_norm(features)
        features = self.dropout(F.relu(features))  # (b_s, n, d_model)

        features = features.unsqueeze(1)
        features = torch.matmul(perm, features)  # (b_s, h, n, d_model)

        adj_perm = perm.unsqueeze(-3) # (b_s, h, 1, n, n)
        adj = adj.unsqueeze(1)
        adj = torch.matmul(adj_perm, torch.matmul(adj, adj_perm.transpose(-2, -1)))  # (b_s, h, d_ein, n, n)
        output = [features.clone()]
        for i in range(self.nlayers):
            this_stride = self.compress_layers[i].stride
            this_filter_size = self.compress_layers[i].filter_size

            upper_pad = self.padding_list[i]
            below_pad = (upper_pad + spatial_size) % this_stride
            below_pad = this_stride if below_pad == 0 else below_pad
            below_pad = this_filter_size - below_pad
            
            # edge feature prepare
            adj = F.pad(adj, (upper_pad, below_pad, upper_pad, below_pad), "constant", 0)
            edge_features = diag_unfold(adj, this_filter_size, this_stride)
            # clique feature prepare
            features = F.pad(features.transpose(-2, -1), (upper_pad, below_pad), "constant", 0).transpose(-2, -1)
            features = features.unfold(2, this_filter_size, this_stride) # (b_s, h, n', d_model, self.filter_size)
            jump_features = features.sum(-1) / this_filter_size

            adj, res_features = self.compress_layers[i](adj, features, edge_features)
            features = res_features + jump_features

            spatial_size = features.shape[2]
            output.append(res_features)
        for i in range(self.nlayers - 1, -1, -1):
            layer_out = output[i] # (b_s, h, n, d_model)
            features = self.uncompress_layers[i](layer_out, features, i)
        output = torch.matmul(perm.transpose(-2, -1), features) # (b_s, h, n, d_model)
        output = output.transpose(1, 2).flatten(-2, -1) # (b_s, n, h * d_model)
        output = self.ff_head(output) # (b_s, n, d_model)
        output = self.output_norm(output)
        output = self.dropout(F.relu(output))
        # return output
        output = self.ff_out(output)
        return F.log_softmax(output, dim=-1)
