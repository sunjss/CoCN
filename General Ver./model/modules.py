import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch

from model.utils import Hrelu, Htanh, Hsigmoid, get_normalized_adj
from model.utils import BiasDiagUnfolder
from model.sparse_utils import SparseBiasDiagUnfolder, to_list_batch_sparse_adj, sparse_padding_square
from model.layers import CompressConvolution, SparseCompressConvolution
from model.layers import UncompressConvolution


class PermGenModule(nn.Module):
    def __init__(self, d_model, temp, b_s, h):
        super(PermGenModule, self).__init__()
        self.htanh = Htanh()
        self.hsigmoid = Hsigmoid()
        self.hrelu = Hrelu()
        self.d_model = d_model
        self.temp = temp
        

    def forward(self, ranking, mask=None):
        device = ranking.device
        b_s, h, node_num = ranking.shape[:3]
        
        relative_ranking = ranking - ranking.permute(0, 1, 3, 2)  # (b_s, h, n, n)
        relative_ranking_bias = relative_ranking.eq(0)
        relative_ranking_bias = torch.triu(relative_ranking_bias, 1).type(torch.float32)
        relative_ranking = relative_ranking + relative_ranking_bias

        relative_ranking_uni = F.relu(relative_ranking)
        relative_ranking_uni = self.hsigmoid.apply(relative_ranking_uni, 4)
        if mask is not None:
            mask = mask.view(b_s, 1, node_num, 1)
            relative_ranking_uni = mask * relative_ranking_uni
        ranking = relative_ranking_uni.sum(-2, keepdim=True) # (b_s, h, 1, n)

        self.ranking = ranking
        indicator = torch.arange(0, node_num, device=device, dtype=torch.float32).view(-1, 1) # (n, 1)
        indicator = indicator - ranking
        indicator = torch.where(indicator < 0, indicator + node_num, indicator)
        indicator = -indicator * self.temp
        perm = torch.exp(indicator)

        if mask is not None:
            perm = mask * perm * mask.transpose(-2, -1)
        return perm



class TopKPermGenModule(nn.Module):
    def __init__(self, d_model, temp, k):
        super(TopKPermGenModule, self).__init__()
        print("using topk cocn")
        self.hsigmoid = Hsigmoid()
        self.d_model = d_model
        self.temp = temp
        self.sub_size = k
        

    # def forward(self, ranking, node_idx_batch, j, mask=None):
    def forward(self, ranking, i, j, mask=None):
        device = ranking.device
        b_s, h, node_num = ranking.shape[:3]
        ranking = ranking.squeeze(-1)
        # top-k extraction
        ranking_sort, sorted_idx = torch.sort(ranking, dim=2, stable=True)  # (b_s, h, n)
        ranking_sort = torch.concat((ranking_sort, sorted_idx), 0)
        ranking_sort = F.pad(ranking_sort, (0, self.sub_size - 1), 'circular')
        topk_ranking = ranking_sort.unfold(-1, self.sub_size, 1) # (2 * b_s, h, n, k)
        # rerank dim 2(node) as input order
        expanse = list(topk_ranking.unflatten(0, (2, b_s)).shape)
        sorted_idx = sorted_idx.view(1, b_s, h, -1, 1).expand(expanse).flatten(0, 1)
        topk_ranking = topk_ranking.scatter(2, sorted_idx, topk_ranking)
        topk_ranking, sorted_idx = topk_ranking[:b_s], topk_ranking[b_s:].type(torch.int64)

        sorted_idx, k_sorted_idx = torch.sort(sorted_idx, dim=-1, stable=True)
        topk_ranking = topk_ranking.gather(-1, k_sorted_idx)

        topk_ranking = topk_ranking[:, :, i:i+j]
        sorted_idx = sorted_idx[:, :, i:i+j]
        # topk_ranking = topk_ranking[:, :, node_idx_batch]
        # sorted_idx = sorted_idx[:, :, node_idx_batch]
        topk_ranking = topk_ranking.permute(0, 2, 1, 3).view(-1, h, self.sub_size, 1)

        relative_ranking = topk_ranking - topk_ranking.permute(0, 1, 3, 2)  # (b_s, h, k, k)
        relative_ranking_bias = relative_ranking.eq(0)
        relative_ranking_bias = torch.triu(relative_ranking_bias, 1).type(torch.float32)
        relative_ranking = relative_ranking + relative_ranking_bias

        relative_ranking_uni = F.relu(relative_ranking)
        relative_ranking_uni = self.hsigmoid.apply(relative_ranking_uni)
        if mask is not None:
            mask = mask.view(b_s, 1, node_num, 1)
            relative_ranking_uni = mask * relative_ranking_uni
        ranking = relative_ranking_uni.sum(-2, keepdim=True) # (b_s, h, 1, n)

        self.ranking = ranking
        indicator = torch.arange(0, self.sub_size, device=device, dtype=torch.float32).view(-1, 1) # (n, 1)
        indicator = indicator - ranking
        indicator = torch.where(indicator < 0, indicator + self.sub_size, indicator)
        indicator = -indicator * self.temp
        perm = torch.exp(indicator)

        if mask is not None:
            perm = mask * perm * mask.transpose(-2, -1)
        return perm, sorted_idx



sub_size = 1
class SparsePermGenModule(nn.Module):
    def __init__(self, d_model, temp, b_s, h):
        super(SparsePermGenModule, self).__init__()
        print("sparse cocn")
        self.hrelu = Hrelu()
        self.d_model = d_model
        self.temp = temp
        
    def forward(self, ranking, mask=None):
        device = ranking.device
        b_s, h, node_num = ranking.shape[:3]

        ranking = ranking.squeeze(-1)
        sorted_idx = torch.argsort(ranking * 100, dim=2, stable=True, descending=True)  # (b_s, h, n)
        ranking_sort = torch.gather(ranking, -1, sorted_idx)
        # approximate to absolute
        rel_scaler = torch.ones((b_s, h, node_num), device=device, dtype=torch.float32)
        # with torch.no_grad():
        #     neighbor_ranking = ranking_sort[:, :, :-1] - ranking_sort[:, :, 1:]
        # rel_scaler = torch.sign(neighbor_ranking)
        # first_scaler = torch.zeros((b_s, h, 1), device=device, dtype=torch.float32)
        # rel_scaler = torch.concat((first_scaler, rel_scaler), 2)
        if mask is not None:
            mask = mask.view(b_s, 1, node_num).expand(-1, h, -1)
            mask = torch.gather(mask, 2, sorted_idx.type(torch.int64))
            rel_scaler = mask * rel_scaler
            ranking_sort = mask * ranking_sort
        rel_scaler = torch.cumsum(rel_scaler, 2)
        rel_minus = torch.cumsum(ranking_sort, 2)
        relative_ranking = rel_minus - rel_scaler * ranking_sort  # for descending.
        rel_scaler = rel_scaler - 1
        rel_scaler = torch.where(rel_scaler < 0, torch.zeros_like(rel_scaler), rel_scaler)
        ranking = self.hrelu.apply(relative_ranking, rel_scaler) # (b_s, h, n)
        # absolute to perm
        self.ranking = ranking
        indicator = rel_scaler - ranking
        indicator = torch.where(indicator < 0, indicator + node_num, indicator)
        perm_value = torch.exp(-indicator * 10)
        if mask is not None:
            perm_value = mask * perm_value

        b_idx = torch.arange(b_s, device=device).view(-1, 1, 1).expand(-1, h, node_num)
        h_idx = torch.arange(h, device=device).view(1, -1, 1).expand(b_s, -1, node_num)
        idx = torch.stack((b_idx * h + h_idx, rel_scaler, sorted_idx), 0)

        perm = torch.sparse_coo_tensor(idx.view(3, -1), perm_value.view(-1), 
                                       size=(b_s * h, node_num, node_num))
        perm = perm.coalesce()
        return perm



class FeatureBasedPositionGenModule(nn.Module):
    def __init__(self, d_in, d_model, h, dropout, N_enc):
        super(FeatureBasedPositionGenModule, self).__init__()
        print("Using feature based position gen model")
        self.feat_ff = nn.Sequential(nn.Linear(d_in, d_model),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(d_model, h))
        self.h = h
        self.nlayers = N_enc
        self.d_model = d_model
            
    def forward(self, x, node_topo, Ahat):
        b_s, n = x.shape[:2]

        node_feat = self.feat_ff(x)
        node_feat = node_feat.flatten(0, 1)
        for i in range(self.nlayers):
            node_feat = torch.mm(Ahat, node_feat)

        node_feat = node_feat.unflatten(0, (b_s, -1))
        out = node_feat.unsqueeze(-1)
        out = out.permute(0, 2, 1, 3)
        return out



class DistanceBasedPositionGenModule(nn.Module):
    def __init__(self, d_in, d_model, h, dropout, N_enc):
        super(DistanceBasedPositionGenModule, self).__init__()
        print("Using distance based position gen model")
        self.dist_ff = nn.Linear(d_in, h)
        self.h = h
        self.nlayers = N_enc
        self.d_model = d_model


    def forward(self, x, node_topo, Ahat):
        b_s, n = x.shape[:2]

        node_topo = self.dist_ff(node_topo)
        node_topo = node_topo.flatten(0, 1)
        node_topo = node_topo.unflatten(0, (b_s, -1))
        out = node_topo.unsqueeze(-1)
        out = out.permute(0, 2, 1, 3)
        return out


import time
class CoCN(nn.Module):
    def __init__(self, ranking, perm_gen, cocn, pre_encoder=None, task_type="single-class", self_loop=1, app_adj=False):
        super(CoCN, self).__init__()
        self.ranking = ranking
        self.perm_gen = perm_gen
        self.cocn = cocn
        self.pre_encoder = pre_encoder
        self.task_type = task_type
        print(task_type)
        self.self_loop = self_loop
        print("self-loop: " + str(self_loop))
        self.app_adj = app_adj
        print("app_adj: " + str(app_adj))


    def _data_extraction(self, data, batch, sub=False):
        mask = None
        pe = None
        n_id = data.n_id if sub else None
        e_id = data.e_id if sub else None
        # node features
        features = data.x[n_id]
        if 'laplacian_eigenvector_pe' in data.keys:
            pe = data.laplacian_eigenvector_pe[n_id]
        elif 'random_walk_pe' in data.keys:
            pe = data.random_walk_pe[n_id]
        elif 'pos' in data.keys:
            pe = data.pos[n_id]
            # features = torch.concat((features, pe), -1)
        if not sub:
            features = features.squeeze(0)
            pe = pe.squeeze(0) if pe is not None else None
        whole_size = features.shape[-2]
        if data.batch.max() > 0:
            features, mask = to_dense_batch(features, batch)
            pe, _ = to_dense_batch(pe, batch) if pe is not None else (None, None)
        
        # edge features
        n = features.shape[-2]
        if features.ndim == 2:
            features = features.unsqueeze(0)
            pe = pe.unsqueeze(0) if pe is not None else None
        if data.num_edges == 0:
            b_s = features.shape[0]
            edge_index = torch.arange(n * b_s, device=features.device)
            edge_index = torch.stack((edge_index, edge_index), 0)
        else:
            edge_index = data.edge_index
        
        Ahat = get_normalized_adj(edge_index, whole_size, n, batch=batch, selfloop=self.self_loop, dense=False)

        if data.edge_attr is not None:
            edge_attr = data.edge_attr[e_id].squeeze(0)
        else:
            edge_attr = torch.ones(edge_index[0].numel(), 1, device=edge_index.device, dtype=torch.float32)
        adj = to_dense_adj(edge_index, batch, edge_attr, max_num_nodes=n)
        adj = adj.permute(0, 3, 1, 2) if adj.ndim == 4 else adj.unsqueeze(1)

        if self.pre_encoder is not None:
            features = self.pre_encoder(features)
            if features.ndim > 3:
                features = features.squeeze(-2)
        if features.dtype == torch.long:
            features = features.type(torch.float32)
        if adj.dtype == torch.int64:
            adj = adj.type(torch.float32)

        if mask is not None:
            features = mask.unsqueeze(-1) * features
            mask = mask.view(adj.shape[0], 1, -1, 1)
            adj = mask * adj * mask.transpose(-2, -1)
        
        return mask, pe, features, Ahat, adj


    def forward(self, data):
        mask, pe, features, Ahat, adj = self._data_extraction(data, data.batch)
        start_time = time.time()
        ranking = self.ranking(features, pe, Ahat)
        perm = self.perm_gen(ranking, mask)  # (b_s, h, n, n)
        features = features.unsqueeze(1)
        if "no-feat" in self.task_type:
            features = None
        b_s, _, node_num = features.shape[:3]
        if self.app_adj:
            features_appd = torch.matmul(Ahat, features.view(b_s * node_num, -1))
            features_appd = features_appd.view(b_s, 1, node_num, -1)
        else:
            features_appd = None
        output = self.cocn(perm, adj.unsqueeze(1), features, appd=features_appd, mask=mask)
        self.time = time.time() - start_time
        if output.ndim > 2:
            output = output.flatten(0, 1)
            output = output[mask.flatten()] if mask is not None else output
        if "single-class" in self.task_type:
            output = F.log_softmax(output, dim=-1)
        elif self.task_type in ["multi-class",  "reg", "binary-class"]:
            pass
        elif "link" in self.task_type:
            outdim = output.shape[-1] // 2
            node_in = output[data.edge_label_index[0], :outdim]
            node_out = output[data.edge_label_index[1], outdim:]
            output = (node_in * node_out).sum(-1) / outdim
        else:
            raise ValueError("Unsupported task type " + self.task_type)
        return output



class SparseCoCN(nn.Module):
    def __init__(self, ranking, perm_gen, cocn, pre_encoder=None, task_type="single-class", self_loop=1, app_adj=False):
        super(SparseCoCN, self).__init__()
        self.ranking = ranking
        self.perm_gen = perm_gen
        self.cocn = cocn
        self.pre_encoder = pre_encoder
        self.task_type = task_type
        print(task_type)
        self.self_loop = self_loop
        print("self-loop: " + str(self_loop))
        self.app_adj = app_adj
        print("app_adj: " + str(app_adj))


    def forward(self, data):
        mask = None
        features = data.x.type(torch.float32)
        if 'laplacian_eigenvector_pe' in data.keys:
            pe = data.laplacian_eigenvector_pe
        elif 'random_walk_pe' in data.keys:
            pe = data.random_walk_pe
        elif 'pos' in data.keys:
            pe = data.pos
        else:
            pe = None
        whole_size = features.shape[-2]
        if data.batch.max() > 0:
            features, mask = to_dense_batch(features, data.batch)
            pe, _ = to_dense_batch(pe, data.batch) if pe is not None else (None, None)
        
        n = features.shape[-2]
        if features.ndim == 2:
            features = features.unsqueeze(0)
            pe = pe.unsqueeze(0) if pe is not None else None
        if data.num_edges == 0:
            b_s = 1 if data.x.ndim == 2 else features.shape[0]
            edge_index = torch.arange(n * b_s, device=features.device)
            edge_index = torch.stack((edge_index, edge_index), 0)
        else:
            edge_index = data.edge_index
        Ahat = get_normalized_adj(edge_index, whole_size, n, batch=data.batch, selfloop=self.self_loop, dense=False)
        adj = to_list_batch_sparse_adj(edge_index, data.batch, data.edge_attr, max_num_nodes=n)

        if self.pre_encoder is not None:
            features = self.pre_encoder(features)
            if features.ndim > 3:
                features = features.squeeze(-2)

        if mask is not None:
            mask = mask.view(adj.shape[0], 1, -1, 1)
            adj = mask * adj * mask.transpose(1, 2)
        start_time = time.time()
        ranking = self.ranking(features, pe, Ahat)
        perm = self.perm_gen(ranking, mask)  # (b_s * h, n, n)

        if "no-feat" in self.task_type:
            features = None
        adj = sparse_perm_2D(perm, adj)
        b_s, node_num = features.shape[:2]
        if self.app_adj:
            features_appd = torch.matmul(Ahat, features.view(b_s * node_num, -1))
            features_appd = features_appd.view(b_s, node_num, -1)
        else:
            features_appd = None
        # features_appd = None
        if mask is not None:
            mask = mask.squeeze(1)
        output = self.cocn(perm, adj, features, appd=features_appd, mask=mask)
        self.time = time.time() - start_time
        if output.ndim > 2:
            output = output.flatten(0, 1)
            output = output[mask.flatten()] if mask is not None else output

        if self.task_type == "single-class":
            output = F.log_softmax(output, dim=-1)
        elif self.task_type in ["multi-class",  "reg", "binary-class"]:
            pass
        elif "link" in self.task_type:
            outdim = output.shape[-1] // 2
            node_in = output[data.edge_label_index[0], :outdim]
            node_out = output[data.edge_label_index[1], outdim:]
            if "scale-dot" in self.task_type:
                output = torch.matmul(node_in, node_out.T) / outdim
            elif "cosine" in self.task_type:
                norm_in = torch.norm(node_in, dim=-1)
                norm_out = torch.norm(node_out, dim=-1)
                output = (node_in * node_out).sum(-1) / (norm_in * norm_out)
            output = output * 2
        else:
            raise ValueError("Unsupported task type " + self.task_type)
        
        return output



class TopKCoCN(nn.Module):
    def __init__(self, ranking, perm_gen, cocn, task_type="single-class", reg_flag=False, self_loop=1, app_adj=True):
        super(TopKCoCN, self).__init__()
        self.ranking = ranking
        self.perm_gen = perm_gen
        self.cocn = cocn
        self.reg = reg_flag
        self.task_type = task_type
        print(task_type)
        self.self_loop = self_loop
        print("self-loop: " + str(self_loop))
        self.app_adj = app_adj
        print("app_adj: " + str(app_adj))

    def forward(self, data, i, j):
        mask = None
        features = data.x.type(torch.float32)
        device = features.device
        if 'laplacian_eigenvector_pe' in data.keys:
            pe = data.laplacian_eigenvector_pe
        elif 'random_walk_pe' in data.keys:
            pe = data.random_walk_pe
        elif 'pos' in data.keys:
            pe = data.pos
        else:
            pe = None
        whole_size = features.shape[-2]
        if data.batch.max() > 0:
            features, mask = to_dense_batch(features, data.batch)
            pe, _ = to_dense_batch(pe, data.batch) if pe is not None else (None, None)
        if features.ndim == 2:
            features = features.unsqueeze(0)
        b_s, node_num = features.shape[:2]
        if data.num_edges == 0:
            edge_index = torch.arange(node_num * b_s, device=device)
            edge_index = torch.stack((edge_index, edge_index), 0)
        else:
            edge_index = data.edge_index
        if "Ahat" in data.keys:
            Ahat = data.Ahat
        else:
            Ahat = get_normalized_adj(edge_index, whole_size, node_num, batch=data.batch, selfloop=self.self_loop, dense=False)
        start_time = time.time()
        ranking = self.ranking(features, pe, Ahat)
        perm_in, sorted_idx = self.perm_gen(ranking, i, j)  # (b_s * n, h, k, k)
        b_s, h, _, k = sorted_idx.shape
        assert b_s == 1, "Mini batches are not supported"
        # subgraph features
        features_idx = sorted_idx.reshape(b_s, -1, 1).expand(-1, -1, features.shape[-1])
        features = features.gather(1, features_idx)
        features = features.view(b_s, h, j, k, -1).transpose(1, 2) # (b_s * j, h, d_model)
        features_in = features.flatten(0, 1)
        # subgraph adj
        ## subgraph edge
        idx = sorted_idx.transpose(1, 2).reshape(b_s * j * h, -1)  # (b_s * j * h, k)
        batch_node_idx = torch.arange(b_s * j * h, device=device).unsqueeze(1).expand(-1, k).flatten()
        mask = torch.sparse_coo_tensor(torch.stack((batch_node_idx, idx.flatten())), 
                                       torch.ones_like(batch_node_idx), 
                                       (b_s * j * h, node_num))
        edge_mask = mask.index_select(-1, edge_index[0]) * mask.index_select(-1, edge_index[1])
        batch_index, edge_mask_index = edge_mask.coalesce().indices()
        edge_attr = data.edge_attr[edge_mask_index] if data.edge_attr is not None else None
        edge_index = edge_index[:, edge_mask_index].unsqueeze(-1)
        ## edge to adj
        batch_node = idx.index_select(0, batch_index)
        sub_idx = torch.arange(0, k, device=device)
        edge_index = (((edge_index - batch_node) == 0) * sub_idx).max(-1).values
        edge_index = edge_index.view(2, -1)
        edge_index = batch_index * (k * k) + edge_index[0] * k + edge_index[1]
        adj_empty = torch.zeros((b_s, j, h, k * k), device=device)  # TODO:d_ein > 1
        adj_in = adj_empty.view(-1).scatter(-1, edge_index, edge_attr if edge_attr is not None else 1.)
        adj_in = adj_in.view(b_s * j, h, 1, k, k)

        output = self.cocn(perm_in, adj_in, features_in)
        self.time = time.time() - start_time
        if self.task_type == "single-class":
            output = F.log_softmax(output, dim=-1)
        elif self.task_type in ["multi-class",  "reg", "binary-class"]:
            pass
        else:
            raise ValueError("Unsupported task type " + self.task_type)
        output = output.view(b_s, j, -1)
        if output.ndim > data.x.ndim:
            output = output.squeeze(0)
        return output



class SparseTopKCoCN(nn.Module):
    def __init__(self, ranking, perm_gen, cocn, task_type="single-class", reg_flag=False, self_loop=1, app_adj=True):
        super(SparseTopKCoCN, self).__init__()
        self.ranking = ranking
        self.perm_gen = perm_gen
        self.cocn = cocn
        self.reg = reg_flag
        self.task_type = task_type
        print(task_type)
        self.self_loop = self_loop
        print("self-loop: " + str(self_loop))
        self.app_adj = app_adj
        print("app_adj: " + str(app_adj))

    def forward(self, data, i, j):
        mask = None
        features = data.x.type(torch.float32)
        device = features.device
        if 'laplacian_eigenvector_pe' in data.keys:
            pe = data.laplacian_eigenvector_pe
        elif 'random_walk_pe' in data.keys:
            pe = data.random_walk_pe
        elif 'pos' in data.keys:
            pe = data.pos
        else:
            pe = None
        whole_size = features.shape[-2]
        if data.batch.max() > 0:
            features, mask = to_dense_batch(features, data.batch)
            pe, _ = to_dense_batch(pe, data.batch) if pe is not None else (None, None)
        n = features.shape[-2]
        if features.ndim == 2:
            features = features.unsqueeze(0)
            pe = pe.unsqueeze(0) if pe is not None else None
        if data.num_edges == 0:
            b_s = 1 if data.x.ndim == 2 else features.shape[0]
            edge_index = torch.arange(n * b_s, device=device)
            edge_index = torch.stack((edge_index, edge_index), 0)
        else:
            edge_index = data.edge_index
        if "Ahat" in data.keys:
            Ahat = data.Ahat
        else:
            Ahat = get_normalized_adj(edge_index, whole_size, n, batch=data.batch, selfloop=self.self_loop, dense=False)

        ranking = self.ranking(features, pe, Ahat)
        perm_in, sorted_idx = self.perm_gen(ranking, i, j)  # (b_s * h * j, k, k)
        b_s, h, _, k = sorted_idx.shape
        # subgraph features
        if self.app_adj:
            features_appd = torch.matmul(Ahat, features.view(b_s * n, -1))
            features_appd = features_appd.view(b_s, n, -1)
            features = torch.concat((features, features_appd), -1)
        features_idx = sorted_idx.reshape(b_s, -1, 1).expand(-1, -1, features.shape[-1])
        features = features.gather(1, features_idx)
        features = features.view(b_s, h, j, k, -1).permute(1, 0, 2, 3, 4)
        features_in = features.flatten(0, 2)
        # subgraph adj
        ## subgraph edge
        idx = sorted_idx.permute(1, 0, 2, 3).reshape(b_s * h * j, -1)  # (b_s * h * j, k)
        batch_node_idx = torch.arange(b_s * h * j, device=device).unsqueeze(1).expand(-1, k).flatten()
        mask = torch.sparse_coo_tensor(torch.stack((batch_node_idx, idx.flatten())), 
                                       torch.ones_like(batch_node_idx), 
                                       (b_s * h * j, n))
        edge_mask = mask.index_select(-1, data.edge_index[0]) * mask.index_select(-1, data.edge_index[1])
        batch_index, edge_mask_index = edge_mask.coalesce().indices()
        edge_attr = data.edge_attr[edge_mask_index] if data.edge_attr is not None else None
        edge_index = data.edge_index[:, edge_mask_index].unsqueeze(-1)
        ## edge to adj
        batch_node = idx.index_select(0, batch_index)
        sub_idx = torch.arange(0, k, device=device)
        edge_index = (((edge_index - batch_node) == 0) * sub_idx).max(-1).values
        edge_index = edge_index.view(2, -1)
        adj_in = to_list_batch_sparse_adj(edge_index, batch_index, edge_attr, max_num_nodes=k, batch_size=perm_in.shape[0])
        adj_in = sparse_perm_2D(perm_in, adj_in)

        output = self.cocn(perm_in, adj_in, features_in)
        if self.task_type == "single-class":
            output = F.log_softmax(output, dim=-1)
        elif self.task_type in ["multi-class",  "reg", "binary-class"]:
            pass
        else:
            raise ValueError("Unsupported task type " + self.task_type)
        output = output.view(b_s, j, -1)
        if output.ndim > data.x.ndim:
            output = output.squeeze(0)
        return output



def sparse_perm_2D(perm, adj):
    '''
    matmul with sparse @ sparse -> sparse
    '''
    # adj (b_s, n, n, d_ein)
    b_s, n = adj.shape[:2]
    h = int(perm.shape[0] / b_s)
    adj = adj.coalesce()
    edge_index = adj.indices()
    
    mul_val = perm.values().view(b_s, h, -1, sub_size)
    edge_features = adj.values().view(edge_index.shape[-1], 1, -1, 1)
    mul_val_l = mul_val[edge_index[0], :, edge_index[1]].unsqueeze(-2)
    mul_val_r = mul_val[edge_index[0], :, edge_index[2]].unsqueeze(-2)
    values = edge_features * mul_val_l * mul_val_r
    values = values.view(-1, adj.shape[-1])
    
    tgt_idx = perm.indices()[1].view(b_s, h, -1, sub_size)
    node_src = tgt_idx[edge_index[0], :, edge_index[1]]
    node_tgt = tgt_idx[edge_index[0], :, edge_index[2]]
    head_idx = torch.arange(h, device=adj.device).view(1, -1)
    head_idx = head_idx.expand(edge_index.shape[1], -1)
    batch_idx = edge_index[0].view(-1, 1) + b_s * head_idx
    batch_idx = torch.stack((batch_idx,) * sub_size, -1)
    idx = torch.stack((batch_idx, node_src, node_tgt), 0)

    adj = torch.sparse_coo_tensor(idx.view(3, -1), values,
                                  size=(b_s * h, n, n, adj.shape[-1]))
    return adj



def sparse_perm_1D(perm, features):
    '''
    matmul with sparse @ dense -> dense
    '''
    bh, n = perm.shape[:2]
    perm = perm.coalesce()
    mul_val = perm.values()    
    index = perm.indices()
    features = features[index[0], index[2]]
    perm_in = mul_val.unsqueeze(-1) * features
    perm_in = perm_in.view(bh * n, sub_size, -1)
    perm_out = torch.zeros_like(perm_in)
    scatter_idx = index[1] + index[0] * n
    scatter_idx = scatter_idx.view(bh * n, sub_size, 1)
    scatter_idx = scatter_idx.expand(-1, -1, perm_in.shape[-1])
    perm_out = perm_out.scatter_reduce_(0, scatter_idx, perm_in, reduce="sum")
    perm_out = perm_out.sum(1)
    return perm_out.view(bh, n, -1)



class CoCNModuleEsem(nn.Module):
    def __init__(self, filter_size_lst, d_ein, d_model, dropout, res):
        super(CoCNModuleEsem, self).__init__()
        self.filter_size_lst = filter_size_lst
        self.conv_block = nn.ModuleList([])
        self.max_filter_size = 0
        for filter_size in filter_size_lst:
            self.conv_block.append(CompressConvolution(d_ein, d_model, d_model, filter_size, 1, dropout, res=0))
            if filter_size > self.max_filter_size:
                self.max_filter_size = filter_size
        self.block_esem = nn.Linear(len(self.filter_size_lst) * d_model, d_model)
        self.res = res

    def forward(self, adj, features, edge_features_lst, _):
        output = [None] * len(self.filter_size_lst)
        for i in range(len(self.filter_size_lst)):
            edge_features = edge_features_lst[i]
            adj, branch_features, res, _ = self.conv_block[i](adj, features, edge_features, _)
            if self.filter_size_lst[i] == self.max_filter_size:
                res_features = res
            prop = self.max_filter_size - self.filter_size_lst[i] + 1
            branch_features = F.pad(branch_features.transpose(-2, -1), (0, prop - 1), "constant", 0).transpose(-2, -1)
            branch_features = branch_features.unfold(-2, prop, 1).sum(-1) / prop
            output[i] = branch_features  # (b_s, h, n, d_model)
        output = torch.stack(output, -1)
        output = output.sum(-1) / len(self.filter_size_lst)
        if self.res:
            output = output + res_features
        return adj, output, output, _



class ResCoCNModuleG(nn.Module):
    def __init__(self, h, d_in, d_ein, d_model, nclass, filter_size, stride, nlayers, nblocks, dropout, app_adj=False, res=True):
        super(ResCoCNModuleG, self).__init__()
        print("Init CoCN for Graph-level Tasks")
        print("Residual Connection:" + str(res))
        self.filter_size = filter_size
        self.stride = stride
        self.nunitlayers = nlayers
        self.d_model = d_model
        self.d_ein = d_ein
        self.ff_in = nn.Linear(d_in * 2, d_model) if app_adj else nn.Linear(d_in, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model * h)
        self.compress_layers = nn.ModuleList([])
        self.bias_diag_unfolder = nn.ModuleList([])
        self.diag_unfolder = nn.ModuleList([])
        for _ in range(nlayers):
            self.compress_layers.append(CompressConvolution(d_ein, d_model, d_model, filter_size, 1, dropout, res=1))
        for _ in range(nblocks):
            self.compress_layers.append(CompressConvolution(d_ein, d_model, d_model, filter_size, self.stride, dropout, res=1))
        self.compress_layers.append(CompressConvolution(d_ein, d_model, d_model, filter_size, 1, dropout, res=1))
        self.nlayers = len(self.compress_layers)
        self.ff_head = nn.Linear(d_model * h, nclass)

        self.filter_size_lst = [filter_size]
        if nlayers > 0:
            for this_flt in self.filter_size_lst:
                self.bias_diag_unfolder.append(BiasDiagUnfolder(this_flt, nlayers, filter_size))
                self.diag_unfolder.append(BiasDiagUnfolder(this_flt))
        else:
            self.diag_unfolder.append(BiasDiagUnfolder(filter_size))
        self.dropout = nn.Dropout(dropout)
    
    def _unit_features_prepare(self, adj, features, filter_size, below_pad, i):
        b_s, h, node_num = features.shape[:3]
        unit_edge = self.bias_diag_unfolder[i](adj, filter_size, 1, node_num + filter_size - 1)
        unit_edge = unit_edge.view(b_s, h, node_num, self.nunitlayers - 1, -1)  # (b_s, h, n', unitlayers - 1, d_ein * k)
        unit_features = unit_edge
        return unit_features
    
    def _features_prepare(self, adj, features):
        node_num = features.shape[2]
        below_pad = self.nunitlayers * (self.filter_size - 1)
        adj = F.pad(adj, (0, below_pad, 0, below_pad), "constant", 0)
        unit_features_lst = [None] * len(self.filter_size_lst)
        edge_features_lst = [None] * len(self.filter_size_lst)
        for i in range(len(self.filter_size_lst)):
            filter_size = self.filter_size_lst[i]
            if self.nunitlayers > 1:
                unit_features_lst[i] = self._unit_features_prepare(adj, features, filter_size, below_pad, i)
            edge_features = self.diag_unfolder[i](adj, filter_size, 1, node_num + filter_size - 1)
            edge_features_lst[i] = edge_features
        upper = adj.triu(below_pad + 1)[:, :, :, :node_num, below_pad:]
        lower = adj.tril(below_pad + 1)[:, :, :, below_pad:, :node_num]
        adj = upper + lower
        return unit_features_lst, edge_features_lst, adj

    def forward(self, perm, adj, features, appd=None, mask=None):
        features = self.ff_in(features)  # (b_s, 1, n, d_model)
        features = self.input_norm(features)
        features = self.dropout(F.relu(features))
        features = torch.matmul(perm, features)  # (b_s, h, n, d_model)
        adj_perm = perm.unsqueeze(-3) # (b_s, h, 1, n, n)
        adj = torch.matmul(adj_perm, torch.matmul(adj, adj_perm.transpose(-2, -1)))  # (b_s, h, d_ein, n, n)
        edge_features = None
        if self.nunitlayers > 0:
            unit_features, edge_features, adj = self._features_prepare(adj, features)
            edge_features = edge_features[0]  # TODO: test incep ver.
        # conv
        for i in range(self.nlayers):
            adj, features, _, mask = self.compress_layers[i](adj, features, 
                                                          edge_features, 
                                                          self.diag_unfolder[-1], mask)
            edge_features = None if i >= self.nunitlayers - 1 else unit_features[0][:, :, :, i-1]
        # perm out
        output = features.max(2).values.flatten(-2, -1)
        output = self.ff_head(output)
        return output



class IncepCoCNModuleG(nn.Module):
    def __init__(self, h, d_in, d_ein, d_model, nclass, filter_size, stride, nlayers, nblocks, dropout, app_adj=True, res=True):
        super(IncepCoCNModuleG, self).__init__()
        print("Init CoCN for Graph-level Tasks")
        print("Residual Connection:" + str(res))
        self.filter_size = filter_size
        self.stride = stride
        self.ff_in = nn.Linear(d_in * 2, d_model) if app_adj else nn.Linear(d_in, d_model)
        self.compress_layers = nn.ModuleList([])
        self.bias_diag_unfolder = nn.ModuleList([])
        self.diag_unfolder = nn.ModuleList([])
        self.filter_size_lst = [2, filter_size]
        print("Kernel Sizes: " + str(self.filter_size_lst))
        for _ in range(nlayers):
            self.compress_layers.append(CoCNModuleEsem(self.filter_size_lst, d_ein, d_model, dropout, res))
        for _ in range(nblocks):
            self.compress_layers.append(CompressConvolution(d_ein, d_model, d_model, filter_size, self.stride, dropout, res))
        self.compress_layers.append(CompressConvolution(d_ein, d_model, d_model, filter_size, 1, dropout, res))

        self.ff_head = nn.Linear(d_model * h, nclass)
        self.unit_conv = nn.ModuleList([])
        self.input_edge_conv = nn.ModuleList([])
        if nlayers > 0:
            for this_flt in self.filter_size_lst:
                self.bias_diag_unfolder.append(BiasDiagUnfolder(this_flt, nlayers, filter_size))
                self.diag_unfolder.append(BiasDiagUnfolder(this_flt))
        else:
            self.diag_unfolder.append(BiasDiagUnfolder(filter_size))
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.nunitlayers = nlayers
        self.nlayers = len(self.compress_layers)
        self.d_model = d_model
        self.d_ein = d_ein
    

    def _unit_features_prepare(self, adj, features, filter_size, below_pad, i):
        b_s, h, node_num = features.shape[:3]
        unit_edge = self.bias_diag_unfolder[i](adj, filter_size, 1, node_num + filter_size - 1)
        unit_edge = unit_edge.view(b_s, h, node_num, self.nunitlayers - 1, -1)  # (b_s, h, n', unitlayers - 1, d_ein * k)
        unit_features = unit_edge
        return unit_features
    

    def _features_prepare(self, adj, features):
        b_s, h, node_num = features.shape[:3]
        below_pad = self.nunitlayers * (self.filter_size - 1)

        adj = F.pad(adj, (0, below_pad, 0, below_pad), "constant", 0)
        unit_features_lst = [None] * len(self.filter_size_lst)
        edge_features_lst = [None] * len(self.filter_size_lst)
        for i in range(len(self.filter_size_lst)):
            filter_size = self.filter_size_lst[i]
            if self.nunitlayers > 1:
                unit_features_lst[i] = self._unit_features_prepare(adj, features, filter_size, below_pad, i)
            edge_features = self.diag_unfolder[i](adj, filter_size, 1, node_num + filter_size - 1)
            edge_features_lst[i] = edge_features
        upper = adj.triu(below_pad + 1)[:, :, :, :node_num, below_pad:]
        lower = adj.tril(below_pad + 1)[:, :, :, below_pad:, :node_num]
        adj = upper + lower
        return unit_features_lst, edge_features_lst, adj


    def forward(self, perm, adj, features, appd=None, mask=None):
        features = self.ff_in(features)  # (b_s, 1, n, d_model)
        features = self.input_norm(features)
        features = self.dropout(F.relu(features))
        features = torch.matmul(perm, features)  # (b_s, h, n, d_model)
        adj_perm = perm.unsqueeze(-3) # (b_s, h, 1, n, n)
        adj = torch.matmul(adj_perm, torch.matmul(adj, adj_perm.transpose(-2, -1)))  # (b_s, h, d_ein, n, n)
        # data prepare
        edge_features = None
        if self.nunitlayers > 0:
            unit_features, edge_features, adj = self._features_prepare(adj, features)
        # conv
        for i in range(self.nlayers):
            adj, features, _, mask = self.compress_layers[i](adj, features, 
                                                                    edge_features, 
                                                                    self.diag_unfolder[-1])
            edge_features = None if i >= self.nunitlayers - 1 else [unit_features[k][:, :, :, i-1] for k in range(len(self.filter_size_lst))]
        output = features.max(2).values.flatten(-2, -1)
        output = self.ff_head(output)  # (b_s, n, d_model)
        return output



class ResCoCNModuleN(nn.Module):
    def __init__(self, h, d_in, d_ein, d_model, nclass, filter_size, stride, nlayers, nblocks, dropout, app_adj=True, res=True):
        super(ResCoCNModuleN, self).__init__()
        print("Init CoCN for Node-level Tasks")
        print("Residual Connection:" + str(res))
        self.filter_size = filter_size
        self.stride = stride
        self.ff_in = (nn.Linear(d_in * 2, d_model) if app_adj else nn.Linear(d_in, d_model)) if d_in > 0 else None
        self.compress_layers = nn.ModuleList([])
        self.uncompress_layers = nn.ModuleList([])
        self.bias_diag_unfolder = nn.ModuleList([])
        self.diag_unfolder = nn.ModuleList([])
        PAD = 0
        self.filter_size_lst = [filter_size]
        d_fin = d_model if d_in > 0 else 0
        for _ in range(nlayers):
            self.compress_layers.append(CompressConvolution(d_ein, d_fin, d_model, filter_size, 1, dropout, res))
            self.uncompress_layers.append(UncompressConvolution(d_ein, d_model, d_model, filter_size, 1, PAD, dropout, res))
            d_fin = d_model
        for _ in range(nblocks):
            self.compress_layers.append(CompressConvolution(d_ein, d_fin, d_model, filter_size, self.stride, dropout, res))
            self.uncompress_layers.append(UncompressConvolution(d_ein, d_model, d_model, filter_size, self.stride, PAD, dropout, res))
            d_fin = d_model
        self.ff_head = nn.Linear(d_model * h, nclass)
        self.unit_conv = nn.ModuleList([])
        self.input_edge_conv = nn.ModuleList([])
        if nlayers > 0:
            for this_flt in self.filter_size_lst:
                self.bias_diag_unfolder.append(BiasDiagUnfolder(this_flt, nlayers, filter_size))
                self.diag_unfolder.append(BiasDiagUnfolder(this_flt))
        else:
            self.diag_unfolder.append(BiasDiagUnfolder(filter_size))
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model * h)
        self.dropout = nn.Dropout(dropout)
        
        self.nunitlayers = nlayers
        self.nlayers = len(self.compress_layers)
        self.ntlayers = len(self.uncompress_layers)
        self.d_model = d_model
        self.d_ein = d_ein
    

    def _unit_features_prepare(self, adj, filter_size, size_tuple, i):
        b_s, h, node_num = size_tuple
        unit_edge = self.bias_diag_unfolder[i](adj, filter_size, 1, node_num + filter_size - 1)
        unit_edge = unit_edge.view(b_s, h, node_num, self.nunitlayers - 1, -1)  # (b_s, h, n', unitlayers - 1, d_ein * k)
        unit_features = unit_edge
        return unit_features
    

    def _features_prepare(self, adj):
        b_s, h, _, node_num = adj.shape[:4]
        below_pad = self.nunitlayers * (self.filter_size - 1)
        adj = F.pad(adj, (0, below_pad, 0, below_pad), "constant", 0)
        unit_features_lst = [None] * len(self.filter_size_lst)
        edge_features_lst = [None] * len(self.filter_size_lst)
        for i in range(len(self.filter_size_lst)):
            filter_size = self.filter_size_lst[i]
            if self.nunitlayers > 1:
                unit_features_lst[i] = self._unit_features_prepare(adj, filter_size, (b_s, h, node_num), i)
            edge_features = self.diag_unfolder[i](adj, filter_size, 1, node_num + filter_size - 1)
            edge_features_lst[i] = edge_features
        upper = adj.triu(below_pad + 1)[:, :, :, :node_num, below_pad:]
        lower = adj.tril(below_pad + 1)[:, :, :, below_pad:, :node_num]
        adj = upper + lower
        return unit_features_lst, edge_features_lst, adj


    def forward(self, perm, adj, features, appd=None, mask=None):
        adj_perm = perm.unsqueeze(-3) # (b_s, h, 1, n, n)
        adj = torch.matmul(adj_perm, torch.matmul(adj, adj_perm.transpose(-2, -1)))  # (b_s, h, d_ein, n, n)
        b_s, h, n = perm.shape[:3]
        if features is not None:
            if appd is not None:
                features = torch.concat((features, appd), -1)
            features = self.ff_in(features)  # (b_s, 1, n, d_model)
            features = self.input_norm(features)
            features = self.dropout(F.relu(features))
            set_features = torch.matmul(perm, features)  # (b_s, h, n, d_model)
        else:
            set_features = None
        # data prepare
        output = [None] * self.nlayers
        edge_features = None
        if self.nunitlayers > 0:
            unit_features, edge_features, adj = self._features_prepare(adj)
            edge_features = edge_features[0]
        # conv
        for i in range(self.nlayers):
            output[i] = set_features
            adj, set_features, _, mask = self.compress_layers[i](adj, set_features, 
                                                                 edge_features, 
                                                                 self.diag_unfolder[-1], i, mask)
            edge_features = None if i >= self.nunitlayers - 1 else unit_features[0][:, :, :, i-1]
        # tconv
        features = set_features
        for i in range(self.nlayers - 1, - 1, -1):
            features = self.uncompress_layers[i](output[i], features, i)
        # perm out
        output = torch.matmul(perm.transpose(-2, -1), features)  # (b_s, h, n, d_model)
        output = output.transpose(1, 2).flatten(-2, -1)  # (b_s, n, h * d_model)
        output = self.output_norm(output)
        output = self.ff_head(output)
        return output



class IncepCoCNModuleN(nn.Module):
    def __init__(self, h, d_in, d_ein, d_model, nclass, filter_size, stride, nlayers, nblocks, nilayers, nolayers, dropout, app_adj=True, res=True):
        super(IncepCoCNModuleN, self).__init__()
        print("Using Inception Block for CoCN")
        print("Residual Connection:" + str(res))
        self.filter_size = filter_size
        self.stride = stride
        self.ff_in = nn.Linear(d_in * 2, d_model) if app_adj else nn.Linear(d_in, d_model)
        self.compress_layers = nn.ModuleList([])
        self.uncompress_layers = nn.ModuleList([])
        self.bias_diag_unfolder = nn.ModuleList([])
        self.diag_unfolder = nn.ModuleList([])
        PAD = 0
        self.filter_size_lst = [2, filter_size]
        print("Kernel Sizes: " + str(self.filter_size_lst))
        for _ in range(nlayers):
            self.compress_layers.append(CoCNModuleEsem(self.filter_size_lst, d_ein, d_model, dropout, res))
            self.uncompress_layers.append(UncompressConvolution(d_ein, d_model * 2, d_model, filter_size, 1, PAD, dropout, res))
        for _ in range(nblocks):
            self.compress_layers.append(CompressConvolution(d_ein, d_model, d_model, filter_size, self.stride, dropout, res))
            self.uncompress_layers.append(UncompressConvolution(d_ein, d_model * 2, d_model, filter_size, self.stride, PAD, dropout, res))
        self.compress_layers.append(CompressConvolution(d_ein, d_model, d_model, filter_size, 1, dropout, res))
        self.uncompress_layers.append(UncompressConvolution(d_ein, d_model, d_model, filter_size, 1, PAD, dropout, res))

        self.ff_head = nn.Linear(d_model * h, nclass)
        self.unit_conv = nn.ModuleList([])
        self.input_edge_conv = nn.ModuleList([])
        if nlayers > 0:
            for this_flt in self.filter_size_lst:
                self.bias_diag_unfolder.append(BiasDiagUnfolder(this_flt, nlayers, filter_size))
                self.diag_unfolder.append(BiasDiagUnfolder(this_flt))
        else:
            self.diag_unfolder.append(BiasDiagUnfolder(filter_size))
        self.output_norm = nn.LayerNorm(d_model * h, elementwise_affine=False)
        self.dropout = nn.Dropout(dropout)
        
        self.nunitlayers = nlayers
        self.nlayers = len(self.compress_layers)
        self.d_model = d_model
        self.d_ein = d_ein
    

    def _unit_features_prepare(self, adj, features, filter_size, below_pad, i):
        b_s, h, node_num = features.shape[:3]
        unit_edge = self.bias_diag_unfolder[i](adj, filter_size, 1, node_num + filter_size - 1)
        unit_edge = unit_edge.view(b_s, h, node_num, self.nunitlayers - 1, -1)  # (b_s, h, n', unitlayers - 1, d_ein * k)
        unit_features = unit_edge
        return unit_features

    
    def _features_prepare(self, adj, features):
        b_s, h, node_num = features.shape[:3]
        below_pad = self.nunitlayers * (self.filter_size - 1)

        adj = F.pad(adj, (0, below_pad, 0, below_pad), "constant", 0)
        unit_features_lst = [None] * len(self.filter_size_lst)
        edge_features_lst = [None] * len(self.filter_size_lst)
        for i in range(len(self.filter_size_lst)):
            filter_size = self.filter_size_lst[i]
            if self.nunitlayers > 1:
                unit_features_lst[i] = self._unit_features_prepare(adj, features, filter_size, below_pad, i)
            edge_features = self.diag_unfolder[i](adj, filter_size, 1, node_num + filter_size - 1)
            edge_features_lst[i] = edge_features
        upper = adj.triu(below_pad + 1)[:, :, :, :node_num, below_pad:]
        lower = adj.tril(below_pad + 1)[:, :, :, below_pad:, :node_num]
        adj = upper + lower
        return unit_features_lst, edge_features_lst, adj


    def forward(self, perm, adj, features):
        features = self.ff_in(features).unsqueeze(1)  # (b_s, 1, n, d_model)
        # features = self.input_norm(features)
        features = self.dropout(F.relu(features))  # (b_s, n, d_model)
        # perm in
        features = torch.matmul(perm, features)  # (b_s, h, n, d_model)
        adj_perm = perm.unsqueeze(-3) # (b_s, h, 1, n, n)
        adj = torch.matmul(adj_perm, torch.matmul(adj.unsqueeze(1), adj_perm.transpose(-2, -1)))  # (b_s, h, d_ein, n, n)
        # data prepare
        output = [None] * self.nlayers
        edge_features = None
        if self.nunitlayers > 0:
            unit_features, edge_features, adj = self._features_prepare(adj, features)
        # conv
        for i in range(self.nlayers):
            output[i] = features
            adj, features, _, mask = self.compress_layers[i](adj, features, 
                                                             edge_features, 
                                                             self.diag_unfolder[-1])
            edge_features = None if i >= self.nunitlayers - 1 else [unit_features[k][:, :, :, i-1] for k in range(len(self.filter_size_lst))]
        # tconv
        for i in range(self.nlayers - 1, -1, -1):
            features = self.uncompress_layers[i](output[i], features, i)
        features = features[:, :, :, :self.d_model]
        # perm out
        output = torch.matmul(perm.transpose(-2, -1), features)  # (b_s, h, n, d_model)
        output = output.transpose(1, 2).flatten(-2, -1)  # (b_s, n, h * d_model)
        output = self.output_norm(output)
        output = self.ff_head(output)  # (b_s, n, d_model)
        return output



class SparseCoCNModuleEsem(nn.Module):
    def __init__(self, filter_size_lst, d_ein, d_model, dropout, app):
        super(SparseCoCNModuleEsem, self).__init__()
        self.filter_size_lst = filter_size_lst
        self.conv_block = nn.ModuleList([])
        self.max_filter_size = 0
        for filter_size in filter_size_lst:
            self.conv_block.append(SparseCompressConvolution(d_ein, d_model, d_model, filter_size, 1, dropout, app_node_feat=0))
            if filter_size > self.max_filter_size:
                self.max_filter_size = filter_size
        self.res = app

    def forward(self, adj, features, edge_features_lst, _, idx, mask):
        output = [None] * len(self.filter_size_lst)
        for i in range(len(self.filter_size_lst)):
            edge_features = edge_features_lst[i]
            adj, branch_features, res, __, mask = self.conv_block[i](adj, features, edge_features, _, idx, mask)
            if self.filter_size_lst[i] == self.max_filter_size:
                res_features = res
            prop = self.max_filter_size - self.filter_size_lst[i] + 1
            branch_features = F.pad(branch_features.transpose(-2, -1), (0, prop - 1), "constant", 0).transpose(-2, -1)
            branch_features = branch_features.unfold(-2, prop, 1).sum(-1) / prop
            output[i] = branch_features  # (b_s, h, n, d_model)
        features = torch.stack(output, -1)
        output = features.sum(-1) / len(self.filter_size_lst)
        if self.res:
            output = output + res_features
        return adj, output, output, [0, 0], mask



class SparseResCoCNModuleN(nn.Module):
    def __init__(self, h, d_in, d_ein, d_model, nclass, filter_size, stride, nlayers, nblocks, dropout, app_adj=True, res=True):
        super(SparseResCoCNModuleN, self).__init__()
        print("Using Sparse Cross Residual Block for CoCN")
        print("Residual Connection:" + str(res))
        self.filter_size = filter_size
        self.stride = stride
        self.ff_in = (nn.Linear(d_in * 2, d_model) if app_adj else nn.Linear(d_in, d_model)) if d_in > 0 else None
        self.compress_layers = nn.ModuleList([])
        self.uncompress_layers = nn.ModuleList([])
        self.bias_diag_unfolder = nn.ModuleList([])
        self.diag_unfolder = nn.ModuleList([])
        PAD = 0
        self.filter_size_lst = [filter_size]
        d_fin = d_model if d_in > 0 else 0
        for _ in range(nlayers):
            self.compress_layers.append(SparseCompressConvolution(d_ein, d_fin, d_model, filter_size, 1, dropout, res))
            self.uncompress_layers.append(UncompressConvolution(d_ein, d_model, d_model, filter_size, 1, PAD, dropout, res))
            d_fin = d_model
        for _ in range(nblocks):
            self.compress_layers.append(SparseCompressConvolution(d_ein, d_fin, d_model, filter_size, self.stride, dropout, res))
            self.uncompress_layers.append(UncompressConvolution(d_ein, d_model, d_model, filter_size, self.stride, PAD, dropout, res))
            d_fin = d_model
        self.ff_head = nn.Linear(d_model * h, nclass)
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model * h)

        self.unit_conv = nn.ModuleList([])
        self.input_edge_conv = nn.ModuleList([])
        if nlayers > 0:
            for this_flt in self.filter_size_lst:
                self.bias_diag_unfolder.append(SparseBiasDiagUnfolder(this_flt, nlayers, filter_size))
                self.diag_unfolder.append(SparseBiasDiagUnfolder(this_flt))
        else:
            self.diag_unfolder.append(SparseBiasDiagUnfolder(filter_size))
        self.dropout = nn.Dropout(dropout)
        
        self.nunitlayers = nlayers
        self.nlayers = len(self.compress_layers)
        self.ntlayers = len(self.uncompress_layers)
        self.d_model = d_model
        self.d_ein = d_ein
        self.h = h
    

    def _unit_features_prepare(self, adj, filter_size, size_tuple, i):
        bh, node_num = size_tuple
        unit_edge = self.bias_diag_unfolder[i](adj, filter_size, 1, node_num + filter_size - 1)
        unit_edge = unit_edge.view(bh, node_num, self.nunitlayers - 1, -1)  # (b_s, h, n', unitlayers - 1, d_ein * k)
        return unit_edge


    def _anti_diagonal_compression(self, adj, bias, size):
        adj = adj.coalesce()
        edge_index = adj.indices()
        mask_triu = (edge_index[2] - edge_index[1] > bias)
        index_triu = edge_index[:, mask_triu]
        index_triu[2] -= bias
        mask_tril = (edge_index[1] - edge_index[2] > bias)
        index_tril = edge_index[:, mask_tril]
        index_tril[1] -= bias
        values = adj.values()
        return torch.sparse_coo_tensor(torch.concat((index_triu, index_tril), -1), 
                                       torch.concat((values[mask_triu], values[mask_tril]), 0),
                                       size)


    def _features_prepare(self, adj):
        '''extract unit step edge features'''
        bh, node_num = adj.shape[:2]
        # adj padding
        below_pad = self.nunitlayers * (self.filter_size - 1)
        adj = sparse_padding_square(adj, below_pad)
        # feature extracting
        unit_features_lst = [None] * len(self.filter_size_lst)
        edge_features_lst = [None] * len(self.filter_size_lst)
        for i in range(len(self.filter_size_lst)):
            filter_size = self.filter_size_lst[i]
            if self.nunitlayers > 1:
                unit_features_lst[i] = self._unit_features_prepare(adj, filter_size, (bh, node_num), i)
            edge_features = self.diag_unfolder[i](adj, filter_size, 1, node_num + filter_size - 1)
            edge_features_lst[i] = edge_features
        # compress adj
        adj = self._anti_diagonal_compression(adj, below_pad, (bh, node_num, node_num, adj.shape[-1]))
        return unit_features_lst, edge_features_lst, adj


    def forward(self, perm, adj, features, appd=None, mask=None):
        bh, n = perm.shape[:2]
        b_s = int(bh / self.h)
        set_features = None
        if features is not None:
            if appd is not None:
                features = torch.concat((features, appd), -1)
            features = torch.stack((features,) * self.h, 1).flatten(0, 1)
            features = sparse_perm_1D(perm, features)
            features = self.ff_in(features)  # (b_s * h, n, d_model)
            # features = self.input_norm(features)
            set_features = self.dropout(F.relu(features))
        # data prepare
        output = [None] * self.nlayers
        edge_features = None
        if self.nunitlayers > 0:
            unit_features, edge_features, adj = self._features_prepare(adj)
            edge_features = edge_features[0]
        # conv
        for i in range(self.nlayers):
            output[i] = set_features.unflatten(0, (b_s, self.h))
            adj, set_features, _, mask = self.compress_layers[i](adj, set_features, 
                                                                 edge_features, 
                                                                 self.diag_unfolder[-1], i, mask)
            edge_features = None if i >= self.nunitlayers - 1 else unit_features[0][:, :, i-1]
        # tconv
        features = set_features.unflatten(0, (b_s, self.h))
        for i in range(self.nlayers - 1, - 1, -1):
            features = self.uncompress_layers[i](output[i], features, i)
        # perm out
        features = features.flatten(0, 1)
        output = sparse_perm_1D(perm.transpose(-2, -1), features)
        output = output.unflatten(0, (b_s, self.h))
        output = output.permute(0, 2, 1, 3).flatten(-2, -1)  # (b_s, n, h * d_model)
        output = self.output_norm(output)
        output = self.ff_head(output)
        return output


class SparseIncepCoCNModuleN(nn.Module):
    def __init__(self, h, d_in, d_ein, d_model, nclass, filter_size, stride, nlayers, nblocks, dropout, app_adj=True, res=True):
        super(SparseIncepCoCNModuleN, self).__init__()
        print("Using Sparse Inception Block for CoCN")
        print("Residual Connection:" + str(res))
        self.filter_size = filter_size
        self.stride = stride
        self.ff_in = (nn.Linear(d_in * 2, d_model) if app_adj else nn.Linear(d_in, d_model)) if d_in > 0 else None
        self.compress_layers = nn.ModuleList([])
        self.uncompress_layers = nn.ModuleList([])
        self.bias_diag_unfolder = nn.ModuleList([])
        self.diag_unfolder = nn.ModuleList([])
        PAD = 0
        self.filter_size_lst = [2, filter_size]
        d_fin = d_model if d_in > 0 else 0
        for _ in range(nlayers):
            self.compress_layers.append(SparseCoCNModuleEsem(self.filter_size_lst, d_ein, d_model, dropout, res))
            self.uncompress_layers.append(UncompressConvolution(d_ein, d_model, d_model, filter_size, 1, PAD, dropout, res))
            d_fin = d_model
        for _ in range(nblocks):
            self.compress_layers.append(SparseCompressConvolution(d_ein, d_fin, d_model, filter_size, self.stride, dropout, res))
            self.uncompress_layers.append(UncompressConvolution(d_ein, d_model, d_model, filter_size, self.stride, PAD, dropout, res))
            d_fin = d_model
        self.ff_head = nn.Linear(d_model * h, nclass)
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model * h)

        self.unit_conv = nn.ModuleList([])
        self.input_edge_conv = nn.ModuleList([])
        if nlayers > 0:
            for this_flt in self.filter_size_lst:
                self.bias_diag_unfolder.append(SparseBiasDiagUnfolder(this_flt, nlayers, filter_size))
                self.diag_unfolder.append(SparseBiasDiagUnfolder(this_flt))
        else:
            self.diag_unfolder.append(SparseBiasDiagUnfolder(filter_size))
        self.dropout = nn.Dropout(dropout)
        
        self.nunitlayers = nlayers
        self.nlayers = len(self.compress_layers)
        self.ntlayers = len(self.uncompress_layers)
        self.d_model = d_model
        self.d_ein = d_ein
        self.h = h
    

    def _unit_features_prepare(self, adj, filter_size, size_tuple, i):
        bh, node_num = size_tuple
        unit_edge = self.bias_diag_unfolder[i](adj, filter_size, 1, node_num + filter_size - 1)
        unit_edge = unit_edge.view(bh, node_num, self.nunitlayers - 1, -1)  # (b_s, h, n', unitlayers - 1, d_ein * k)
        return unit_edge


    def _anti_diagonal_compression(self, adj, bias, size):
        adj = adj.coalesce()
        edge_index = adj.indices()
        mask_triu = (edge_index[2] - edge_index[1] > bias)
        index_triu = edge_index[:, mask_triu]
        index_triu[2] -= bias
        mask_tril = (edge_index[1] - edge_index[2] > bias)
        index_tril = edge_index[:, mask_tril]
        index_tril[1] -= bias
        values = adj.values()
        return torch.sparse_coo_tensor(torch.concat((index_triu, index_tril), -1), 
                                       torch.concat((values[mask_triu], values[mask_tril]), 0),
                                       size)


    def _features_prepare(self, adj):
        '''extract unit step edge features'''
        bh, node_num = adj.shape[:2]
        # adj padding
        below_pad = self.nunitlayers * (self.filter_size - 1)
        adj = sparse_padding_square(adj, below_pad)
        # feature extracting
        unit_features_lst = [None] * len(self.filter_size_lst)
        edge_features_lst = [None] * len(self.filter_size_lst)
        for i in range(len(self.filter_size_lst)):
            filter_size = self.filter_size_lst[i]
            if self.nunitlayers > 1:
                unit_features_lst[i] = self._unit_features_prepare(adj, filter_size, (bh, node_num), i)
            edge_features = self.diag_unfolder[i](adj, filter_size, 1, node_num + filter_size - 1)
            edge_features_lst[i] = edge_features
        # compress adj
        adj = self._anti_diagonal_compression(adj, below_pad, (bh, node_num, node_num, adj.shape[-1]))
        return unit_features_lst, edge_features_lst, adj


    def forward(self, perm, adj, features, appd=None, mask=None):
        bh, n = perm.shape[:2]
        b_s = int(bh / self.h)
        set_features = None
        if features is not None:
            if appd is not None:
                features = torch.concat((features, appd), -1)
            features = torch.stack((features,) * self.h, 1).flatten(0, 1)
            features = sparse_perm_1D(perm, features)
            features = self.ff_in(features)  # (b_s * h, n, d_model)
            features = self.input_norm(features)
            set_features = self.dropout(F.relu(features))
        # data prepare
        output = [None] * self.nlayers
        edge_features = None
        if self.nunitlayers > 0:
            unit_features, edge_features, adj = self._features_prepare(adj)
        # conv
        for i in range(self.nlayers):
            output[i] = set_features.unflatten(0, (b_s, self.h))
            adj, set_features, _, mask = self.compress_layers[i](adj, set_features, 
                                                                 edge_features, 
                                                                 self.diag_unfolder[-1], i, mask)
            edge_features = None if i >= self.nunitlayers - 1 else [unit_features[k][:, :, i-1] for k in range(len(self.filter_size_lst))]
        # tconv
        features = set_features.unflatten(0, (b_s, self.h))
        for i in range(self.nlayers - 1, - 1, -1):
            features = self.uncompress_layers[i](output[i], features, i)
        # perm out
        features = features.flatten(0, 1)
        output = sparse_perm_1D(perm.transpose(-2, -1), features)
        output = output.unflatten(0, (b_s, self.h))
        output = output.permute(0, 2, 1, 3).flatten(-2, -1)  # (b_s, n, h * d_model)
        output = self.output_norm(output)
        output = self.ff_head(output)
        return output