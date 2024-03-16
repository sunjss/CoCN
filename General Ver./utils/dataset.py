import os
import pickle
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.utils import to_undirected

class Grapg8cDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Grapg8cDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["graph8c.g6"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for _, datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(), 1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1, 0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PlanarSATPairsDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["GRAPHSAT.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass  

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, "raw/GRAPHSAT.pkl"), "rb"))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(Data.from_dict(data)) for data in data_list]
        else:
            data_list = [Data.from_dict(data) for data in data_list]


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class SRDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["sr251256.g6"]  #sr251256  sr351668

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list. 
        dataset = nx.read_graph6(self.raw_paths[0])
        data_list = []
        for i, datum in enumerate(dataset):
            x = torch.ones(datum.number_of_nodes(), 1)
            edge_index = to_undirected(torch.tensor(list(datum.edges())).transpose(1, 0))            
            data_list.append(Data(edge_index=edge_index, x=x, y=0))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SpectralDesign(object):   
    def __init__(self, nmax=0, recfield=1, dv=5, nfreq=5, adddegree=False, laplacien=True, addadj=False, vmax=None):
        # receptive field. 0: adj, 1; adj+I, n: n-hop area 
        self.recfield = recfield  
        # b parameter
        self.dv = dv
        # number of sampled point of spectrum
        self.nfreq = nfreq
        # if degree is added to node feature
        self.adddegree = adddegree
        # use laplacian or adjacency for spectrum
        self.laplacien = laplacien
        # add adjacecny as edge feature
        self.addadj = addadj
        # use given max eigenvalue
        self.vmax = vmax

        # max node for PPGN algorithm, set 0 if you do not use PPGN
        self.nmax = nmax    

    def __call__(self, data):

        n = data.x.shape[0]
        data.x = data.x.type(torch.float32)
               
        nsup = self.nfreq + 1
        if self.addadj:
            nsup += 1
            
        A = np.zeros((n, n), dtype = np.float32)
        # SP = np.zeros((nsup,n,n),dtype=np.float32) 
        A[data.edge_index[0], data.edge_index[1]] = 1
        
        if self.adddegree:
            data.x = torch.cat([data.x, torch.tensor(A.sum(0)).unsqueeze(-1)], 1)

        # calculate receptive field. 0: adj, 1; adj+I, n: n-hop area
        # if self.recfield == 0:
        #     M = A
        # else:
        #     M = (A + np.eye(n))
        #     for i in range(1, self.recfield):
        #         M = M.dot(M) 
        # M = (M > 0)
        # d = A.sum(axis=0) 

        # # normalized Laplacian matrix.
        # dis = 1 / np.sqrt(d)
        # dis[np.isinf(dis)] = 0
        # dis[np.isnan(dis)] = 0
        # D = np.diag(dis)
        # nL = np.eye(D.shape[0]) - (A.dot(D)).T.dot(D)
        # V, U = np.linalg.eigh(nL) 
        # V[V < 0] = 0

        # if not self.laplacien:        
        #     V, U = np.linalg.eigh(A)

        # # design convolution supports
        # vmax = self.vmax
        # if vmax is None:
        #     vmax = V.max()

        # freqcenter = np.linspace(V.min(), vmax, self.nfreq)
        
        # # design convolution supports (aka edge features)         
        # for i in range(0, len(freqcenter)): 
        #     SP[i, :, :] = M * (U.dot(np.diag(np.exp(-(self.dv * (V - freqcenter[i])**2))).dot(U.T))) 
        # # add identity
        # SP[len(freqcenter), :, :] = np.eye(n)
        # # add adjacency if it is desired
        # if self.addadj:
        #     SP[len(freqcenter)+1, :, :] = A
           
        # # set convolution support weigths as an edge feature
        # E = np.where(M > 0)
        # data.edge_index2 = torch.Tensor(np.vstack((E[0], E[1]))).type(torch.int64)
        # data.edge_attr2 = torch.Tensor(SP[:, E[0], E[1]].T).type(torch.float32)        

        return data


class YandexDataset:
    def __init__(self, name, fold_idx, root_path, device='cpu', transform=None):

        print('Preparing data...')
        data = np.load(root_path + f'{name.replace("-", "_")}.npz')
        node_features = torch.tensor(data['node_features'])
        labels = torch.tensor(data['node_labels'])
        edges = torch.tensor(data['edges']).T
        if name not in ['squirrel-directed', 'squirrel-filtered-directed', 'chameleon-directed', 'chameleon-filtered-directed']:
            edges = to_undirected(edges)

        num_classes = len(labels.unique())
        num_targets = num_classes

        train_masks = torch.tensor(data['train_masks'])
        val_masks = torch.tensor(data['val_masks'])
        test_masks = torch.tensor(data['test_masks'])

        self.graph = Data(node_features, edges, y=labels)
        if transform is not None:
            self.graph = transform(self.graph)
        self.graph.train_mask = train_masks[fold_idx]
        self.graph.val_mask = val_masks[fold_idx]
        self.graph.test_mask = test_masks[fold_idx]

        self.name = name
        self.device = device

        self.num_targets = num_targets


    # def compute_metrics(self, logits):
    #     if self.num_targets == 1:
    #         train_metric = roc_auc_score(y_true=self.labels[self.train_idx].cpu().numpy(),
    #                                      y_score=logits[self.train_idx].cpu().numpy()).item()

    #         val_metric = roc_auc_score(y_true=self.labels[self.val_idx].cpu().numpy(),
    #                                    y_score=logits[self.val_idx].cpu().numpy()).item()

    #         test_metric = roc_auc_score(y_true=self.labels[self.test_idx].cpu().numpy(),
    #                                     y_score=logits[self.test_idx].cpu().numpy()).item()

    #     else:
    #         preds = logits.argmax(axis=1)
    #         train_metric = (preds[self.train_idx] == self.labels[self.train_idx]).float().mean().item()
    #         val_metric = (preds[self.val_idx] == self.labels[self.val_idx]).float().mean().item()
    #         test_metric = (preds[self.test_idx] == self.labels[self.test_idx]).float().mean().item()

    #     metrics = {
    #         f'train {self.metric}': train_metric,
    #         f'val {self.metric}': val_metric,
    #         f'test {self.metric}': test_metric
    #     }

    #     return metrics
