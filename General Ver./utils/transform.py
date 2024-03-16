import math
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import homophily, one_hot, to_networkx, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.transforms import BaseTransform
import torch_geometric.transforms as T
from tqdm import tqdm


def dijkstras_matrix_gpu(W_d):
    n = np.shape(W_d)[0]
    
    # Create a columnar dataframe describing edges
    df_g = cd.DataFrame({
        'source':      cp.array([x // n for x in range(n * n)]),
        'destination': cp.array([x % n for x in range(n * n)]),
        'weight':      cp.minimum(W_d, cp.transpose(W_d)).ravel()})
    
    graph_d = cg.Graph()
    graph_d.from_cudf_edgelist(df_g, source='source', destination='destination', edge_attr='weight', renumber=False)
    dist_d = cp.empty_like(W_d)
    for i in range(n):
        dist_d[i, :] = cp.asarray(cg.traversal.shortest_path(graph_d, i)['distance'])
    
    return dist_d


def communicability_based_dist_mx(G, n):
    ## connectivity
    print("creating communicability based distance matrix")
    nodelist = list(G)  # ordering of nodes in matrix
    A = nx.to_numpy_array(G, nodelist)
    # convert to 0-1 matrix
    A[A != 0.0] = 1
    w, vec = np.linalg.eigh(A)
    expw = np.exp(w)
    mapping = dict(zip(nodelist, range(len(nodelist))))
    dist = np.zeros((n, n))
    # computing communicabilities
    with tqdm(unit='it', total=n) as tqdm_t:
        for u in G:
            for v in G:
                s = 0
                p = mapping[u]
                q = mapping[v]
                for j in range(len(nodelist)):
                    s += vec[:, j][p] * vec[:, j][q] * expw[j]
                dist[u][v] = float(s)
            tqdm_t.update(1)
    dist = torch.from_numpy(dist).type(torch.float32)
    return dist

import itertools


def connectivity_based_dist_mx(G, n):
    ## connectivity cpu ver.
    # dist = nx.all_pairs_node_connectivity(G)
    print("creating connectivity based distance matrix")
    nbunch = G
    flow_func = None

    directed = G.is_directed()
    if directed:
        iter_func = itertools.permutations
        len_iter = math.perm(n, 2)
    else:
        iter_func = itertools.combinations
        len_iter = math.comb(n, 2)
    all_pairs = np.zeros((n, n))
    # Reuse auxiliary digraph and residual network
    H = nx.connectivity.build_auxiliary_node_connectivity(G)
    R = nx.flow.build_residual_network(H, "capacity")
    kwargs = {"flow_func": flow_func, "auxiliary": H, "residual": R}
    with tqdm(unit='it', total=len_iter) as tqdm_t:
        for u, v in iter_func(nbunch, 2):
            K = nx.connectivity.local_node_connectivity(G, u, v, **kwargs)
            all_pairs[u, v] = K
            all_pairs[v, u] = K
            tqdm_t.update(1)
    dist = torch.from_numpy(all_pairs).type(torch.float32)
    return dist

def rd_based_dist_mx(G, n):
    ## resistance distance cpu ver.
    print("creating rd based distance matrix")
    g_components_list = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    g_resistance_matrix = np.zeros((n, n)) - 1.0
    g_index = 0
    with tqdm(unit='it', total=len(g_components_list)) as tqdm_t:
        for item in g_components_list:
            cur_adj = nx.to_numpy_array(item)
            cur_num_nodes = cur_adj.shape[0]
            cur_res_dis = np.linalg.pinv(
                np.diag(cur_adj.sum(axis=-1)) - cur_adj + 
                np.ones((cur_num_nodes, cur_num_nodes), dtype=np.float32) / cur_num_nodes
            ).astype(np.float32)
            A = np.diag(cur_res_dis)[:, None]
            B = np.diag(cur_res_dis)[None, :]
            cur_res_dis = A + B - 2 * cur_res_dis
            g_resistance_matrix[g_index:g_index + cur_num_nodes, g_index:g_index + cur_num_nodes] = cur_res_dis
            g_index += cur_num_nodes
            tqdm_t.update(1)
    g_cur_index = []
    for item in g_components_list:
        g_cur_index.extend(list(item.nodes))
    # g_resistance_matrix = g_resistance_matrix[g_cur_index, :]
    # g_resistance_matrix = g_resistance_matrix[:, g_cur_index]
    idx = np.arange(n)
    g_resistance_matrix[g_cur_index, :] = g_resistance_matrix[idx, :]
    g_resistance_matrix[:, g_cur_index] = g_resistance_matrix[:, idx]

    if g_resistance_matrix.max() > n - 1:
        print(f'error: {g_resistance_matrix}')
    g_resistance_matrix[g_resistance_matrix == -1.0] = g_resistance_matrix.max() * 2
    dist = torch.from_numpy(g_resistance_matrix).type(torch.float32)
    return dist

def spd_based_dist_mx(G, n):
    ## shortest distance cpu ver.
    # path_len = dict(nx.all_pairs_shortest_path_length(G))
    # print("creating spd based distance matrix")
    dist = nx.floyd_warshall_numpy(G)
    # dist = np.zeros((n, n))
    # cutoff = float("inf")
    # with tqdm(unit='it', total=n) as tqdm_t:
    #     for s in G:
    #         if s not in G:
    #             raise nx.NodeNotFound(f"Source {s} is not in G")
    #         seen = set([s])
    #         nextlevel = [s]
    #         level = 0
    #         for v in nextlevel:
    #             dist[s, v] = level
    #         while nextlevel and cutoff > level:
    #             level += 1
    #             thislevel = nextlevel
    #             nextlevel = []
    #             for v in thislevel:
    #                 for w in G._adj[v]:
    #                     if w not in seen:
    #                         seen.add(w)
    #                         nextlevel.append(w)
    #                         dist[s, w] = level
    #         tqdm_t.update(1)

    dist = torch.from_numpy(dist).type(torch.float32)
    unreachable_dist = dist.max()
    dist = dist * 0.1
    dist = torch.where(dist != 0, dist, unreachable_dist) * (1 - torch.eye(n))
    return dist


class PathPre(BaseTransform):
    def __init__(self) -> None:
        # print("---getting distance---")
        # print("This might take a while")
        self.tqdm_t = None

    def __call__(self, data):
        if self.tqdm_t is None:
            self.tqdm_t = tqdm(unit='it')
        n = data.num_nodes
        G = to_networkx(data).to_undirected()
        tau = 1
        dist = spd_based_dist_mx(G, n) / tau
        self.tqdm_t.update(1)
        # data.pos = dist
        dist_pow_col_sum = dist.pow(2).sum(1)
        fdist_b = dist_pow_col_sum.unsqueeze(-1) * 2 / n  # (n, 1)
        fdist = 2 * dist / n - 1

        fdist_pow_col_sum = fdist.pow(2).sum(1) - 1  # -2 + 1 = -1, 直接覆盖对角线，所以fdist-1无效
        fdist_pow_col_sum = torch.where(fdist_pow_col_sum == 0, 1e-5, fdist_pow_col_sum)
        # fdist_csum_as_diag = torch.diagonal_scatter(torch.zeros_like(fdist), fdist_pow_col_sum, 0, -2, -1) # 近似
        fdist_csum_as_diag = torch.diagonal_scatter(fdist * 2, fdist_pow_col_sum, 0, -2, -1)

        fdist_inv = torch.linalg.inv(fdist_csum_as_diag)  # (n, n)
        data.pos = torch.matmul(fdist_inv, fdist_b) * tau # (n, 1)
        return data


class FeatIdx2OneHotPre(BaseTransform):
    def __init__(self, num_classes=None) -> None:
        self.num_classes = num_classes
    
    def __call__(self, data):
        if data.x is None:
            features = data.node_species
        else:
            features = data.x
        if self.num_classes is None:
            self.num_classes = len(features.unique())
        data.x = one_hot(features.view(-1), num_classes=self.num_classes)
        data.num_features = self.num_classes
        data.num_node_features = self.num_classes
        return data


class IrrgularFeatIdx2OneHotPre(BaseTransform):
    def __init__(self, num_classes=None) -> None:
        self.num_classes = num_classes
    
    def __call__(self, data):
        if data.x is None:
            features = data.node_species
        else:
            features = data.x
        ids = features.unique()
        if self.num_classes is None:
            self.num_classes = len(ids)
        id_dict = dict(list(zip(ids.numpy(), np.arange(self.num_classes))))
        one_hot_encoding = torch.zeros((features.shape[0], self.num_classes))
        for i, u in enumerate(features):
            if id_dict[u.item()] == 4:
                pass
            else:
                one_hot_encoding[i][id_dict[u.item()]] = 1
        data.x = one_hot_encoding
        data.num_features = self.num_classes
        data.num_node_features = self.num_classes
        return data


class EdgeIdx2OneHotPre(BaseTransform):
    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes
    
    def __call__(self, data):
        data.edge_attr = one_hot(data.edge_attr, num_classes=self.num_classes)
        data.num_edge_features = self.num_classes
        return data


class ConcatCoordAndPixel(BaseTransform):
    def __call__(self, data):
        data.x = torch.concat((data.x, data.pos), -1)
        return data


class AddDeg(BaseTransform):
    def __call__(self, data):
        edge_weight = torch.ones(data.edge_index.size(1), device=data.edge_index.device)
        row, col = data.edge_index[0], data.edge_index[1]
        deg = scatter(edge_weight, row, 0, dim_size=maybe_num_nodes(data.edge_index), reduce='sum')
        deg = deg / deg.max()
        data.x = torch.concat((data.x, deg.unsqueeze(1)), -1)
        return data


class AddCoord(BaseTransform):
    def __call__(self, data):
        data.pos = data.x[:, -2:]
        return data


class AddNormCoord(BaseTransform):
    def __call__(self, data):
        data.pos = data.x[:, -2:]
        return data


class NormalizeCoord(BaseTransform):
    def __call__(self, data):
        data.x[:, -2] = data.x[:, -2] / data.x[:, -2].max()
        data.x[:, -1] = data.x[:, -1] / data.x[:, -1].max()
        return data


class LabelRename(BaseTransform):
    def __call__(self, data):
        data.y = data.edge_label.type(torch.float32)
        return data


class AddVirtualNode(BaseTransform):
    def __init__(self) -> None:
        self.add_virtual = T.VirtualNode()

    def __call__(self, data):
        data = self.add_virtual(data)
        data.x[-1] = data.x[:-1].sum(0) / (data.x.shape[0] - 1)
        return data