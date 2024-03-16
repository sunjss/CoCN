import csv
import os
import gdown
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.datasets as GeoData
import torch_geometric.transforms as T
from torch_geometric.data.data import Data
from torch_geometric.loader import (DataListLoader, DataLoader, 
                                    ClusterData, ClusterLoader, GraphSAINTNodeSampler,
                                    NeighborLoader, RandomNodeSampler)
from torch_geometric.utils import index_to_mask, degree
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from model.utils import get_normalized_adj
from utils.transform import (PathPre, LabelRename)
from utils.dataset import YandexDataset
from utils.ncd_dataset import load_nc_dataset
from utils.evaluators import (accuracy, accuracy_SBM, macro_f1_score, eval_mrr_and_hits,
                              eval_hits, eval_mrr, eval_rocauc, eval_average_precision)
from NeuroGraph.datasets import NeuroGraphDataset

# root = os.path.dirname(os.path.abspath(__file__))[:-6] + '/data/'
root = "/data/junshusun/COCONetwork/data/"
splits_drive_url = {
    'snap-patents' : '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N', 
    'pokec' : '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_', 
}

dataset_list = ["COLLAB",  "IMDB-BINARY", "IMDB-MULTI", 'HCPGender', 'HCPActivity', 'HCPAge',
                "MUTAG", "PROTEINS", "NCI1", "ACTOR", "genius", "PCQM-Contact",
                'squirrel-directed', 'squirrel-filtered-directed', "squirrel-filtered", "squirrel", 
                'chameleon-directed', 'chameleon-filtered-directed', "chameleon", "chameleon-filtered", 
                'amazon-ratings', 'minesweeper', 'tolokers', 'questions',
                "AmazonComputers", "AmazonPhoto", "CoauthorCS", "CoauthorPhysics",
                "WikiCS", "CORNELL", "TEXAS", "WISCONSIN", "SQUIRREL", "CHAMELEON"]

class CoCNDataLoader:
    def __init__(self, sampler='random'):
        self.sampler = sampler
        self.saint_batch_size = 1000
        self.ndists = 1

    def load_data(self, dataset='COLLAB', spilit_type="public", nbatch=1, fold_idx=0):
        if dataset in ["COLLAB", "REDDIT-BINARY", "REDDIT-MULTI-12K", "IMDB-BINARY", "IMDB-MULTI"]:
            self.load_tu(dataset, nbatch, fold_idx, True)
        elif dataset in ["MUTAG", "PROTEINS", "NCI1"]:
            self.load_tu(dataset, nbatch, fold_idx, False)
        elif dataset in ["CORNELL", "TEXAS", "WISCONSIN"]:
            self.load_WebKB(dataset, nbatch)
        elif dataset in ["SQUIRREL", "CHAMELEON"]:
            self.load_wikiNet(dataset, nbatch)
        elif dataset in ["AmazonComputers", "AmazonPhoto"]:
            self.load_Amazon(dataset[6:], nbatch)
        elif dataset in ["CoauthorCS", "CoauthorPhysics"]:
            self.load_Coauthor(dataset[8:], nbatch)
        elif dataset == "ACTOR":
            self.load_actor(nbatch)
        elif dataset in ['HCPGender', 'HCPActivity', 'HCPAge']:
            self.load_neurograph(dataset, nbatch)
        elif dataset in ["genius"]:
            self.load_nc(dataset, nbatch, fold_idx)
        elif dataset in ["PCQM-Contact"]:
            self.load_LRGB(dataset, nbatch)
        elif dataset in ['squirrel-directed', 'squirrel-filtered-directed', "squirrel-filtered", "squirrel", 
                         'chameleon-directed', 'chameleon-filtered-directed', "chameleon", "chameleon-filtered", 
                         'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
            self.load_yandex(dataset, nbatch, fold_idx)
        else:
            sup_dataset = "\nSupported datasets include: "
            for i in dataset_list:
                sup_dataset += i
                sup_dataset += ", "
            sup_dataset = sup_dataset[:-2]
            raise ValueError("Unsupported type: " + dataset + sup_dataset)
    

    def load_LRGB(self, dataset, batch_size):
        data_path = root + "LRGB/"
        if dataset in ['PCQM-Contact']:
            pre_transform = T.Compose([LabelRename()])
        else:
            pre_transform = T.Compose([])
        train_dataset = GeoData.LRGBDataset(data_path, name=dataset, split="train", pre_transform=pre_transform)
        val_dataset = GeoData.LRGBDataset(data_path, name=dataset, split="val", pre_transform=pre_transform)
        test_dataset = GeoData.LRGBDataset(data_path, name=dataset, split="test", pre_transform=pre_transform)
        if batch_size > 1:
            self.train_data = DataListLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            self.val_data = DataListLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
            self.test_data = DataListLoader(dataset=test_dataset, batch_size=1)
        else:
            self.train_data = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            self.val_data = DataLoader(dataset=val_dataset, batch_size=batch_size)
            self.test_data = DataLoader(dataset=test_dataset, batch_size=batch_size)
        self.nclass = train_dataset.num_classes
        self.nfeats = train_dataset.num_node_features
        self.nedgefeats = train_dataset.num_edge_features if train_dataset.num_edge_features else 1
        self.task_type = "single-class"
        if dataset in ['PCQM-Contact']:
            self.metric = eval_mrr_and_hits
            self.task_type = "link-scale-dot"
        

    def load_neurograph(self, name, batch_size):
        dataset = NeuroGraphDataset(root=root, name=name)
        labels = [d.y.item() for d in dataset]
        file_dir = root + name + "/split"
        flag = os.path.exists(file_dir)
        if flag:
            train_tmp = np.load(file_dir + "/tmp.npy")
            test_indices = np.load(file_dir + "/test_ind.npy")
        else:
            train_tmp, test_indices = train_test_split(list(range(len(labels))),
                                                    test_size=0.2, stratify=labels, shuffle= True)
            os.makedirs(file_dir)
            np.save(file_dir + "/tmp.npy", train_tmp)
            np.save(file_dir + "/test_ind.npy", test_indices)
        tmp = dataset[train_tmp]
        if flag:
            train_indices = np.load(file_dir + "/train_ind.npy")
            val_indices = np.load(file_dir + "/val_ind.npy")
        else:
            train_labels = [d.y.item() for d in tmp]
            train_indices, val_indices = train_test_split(list(range(len(train_labels))),
                                                        test_size=0.125, stratify=train_labels, shuffle = True)
            np.save(file_dir + "/train_ind", train_indices)
            np.save(file_dir + "/val_ind", val_indices)
        train_dataset = tmp[train_indices]
        val_dataset = tmp[val_indices]
        test_dataset = dataset[test_indices]
        if batch_size > 1:
            self.train_data = DataListLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            self.val_data = DataListLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
            self.test_data = DataListLoader(dataset=test_dataset, batch_size=1)
        else:
            self.train_data = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            self.val_data = DataLoader(dataset=val_dataset, batch_size=batch_size)
            self.test_data = DataLoader(dataset=test_dataset, batch_size=batch_size)
        self.nclass = dataset.num_classes
        self.nfeats = dataset.num_node_features
        self.nedgefeats = dataset.num_edge_features if dataset.num_edge_features else 1
        self.task_type = "single-class"
        self.metric = accuracy

    
    def generate_splits(self, data, g_split):
        n_nodes = len(data.x)
        train_mask = torch.zeros(n_nodes, dtype=bool)
        valid_mask = torch.zeros(n_nodes, dtype=bool)
        test_mask = torch.zeros(n_nodes, dtype=bool)
        idx = torch.randperm(n_nodes)
        val_num = test_num = int(n_nodes * (1 - g_split) / 2)
        train_mask[idx[val_num + test_num:]] = True
        valid_mask[idx[:val_num]] = True
        test_mask[idx[val_num:val_num + test_num]] = True
        data.train_mask = train_mask
        data.val_mask = valid_mask
        data.test_mask = test_mask
        return data


    def load_Amazon(self, dataset, batch_size):
        data_path = root + "amazon/"
        pre_transform = T.Compose([])
        dataset = GeoData.Amazon(data_path, name=dataset, pre_transform=pre_transform)
        data = dataset[0]
        data = self.generate_splits(data, 0.6)
        if batch_size > 1:
            self.train_data = DataListLoader(dataset=[data], batch_size=batch_size, shuffle=True)
            self.val_data = DataListLoader(dataset=[data], batch_size=batch_size, shuffle=True)
            self.test_data = DataListLoader(dataset=[data], batch_size=1)
        else:
            self.train_data = DataLoader(dataset=[data], batch_size=batch_size, shuffle=True)
            self.val_data = DataLoader(dataset=[data], batch_size=batch_size)
            self.test_data = DataLoader(dataset=[data], batch_size=batch_size)
        self.nclass = dataset.num_classes
        self.nfeats = dataset.num_node_features
        self.nedgefeats = dataset.num_edge_features if dataset.num_edge_features else 1
        self.task_type = "single-class"
        self.metric = accuracy 


    def load_Coauthor(self, dataset, batch_size):
        data_path = root + "coauthor/"
        pre_transform = T.Compose([])
        dataset = GeoData.Coauthor(data_path, name=dataset, pre_transform=pre_transform)
        data = dataset[0]
        data = self.generate_splits(data, 0.6)
        if batch_size > 1:
            self.train_data = DataListLoader(dataset=[data], batch_size=batch_size, shuffle=True)
            self.val_data = DataListLoader(dataset=[data], batch_size=batch_size, shuffle=True)
            self.test_data = DataListLoader(dataset=[data], batch_size=1)
        else:
            self.train_data = DataLoader(dataset=[data], batch_size=batch_size, shuffle=True)
            self.val_data = DataLoader(dataset=[data], batch_size=batch_size)
            self.test_data = DataLoader(dataset=[data], batch_size=batch_size)
        self.nclass = dataset.num_classes
        self.nfeats = dataset.num_node_features
        self.nedgefeats = dataset.num_edge_features if dataset.num_edge_features else 1
        self.task_type = "single-class"
        self.metric = accuracy 


    def load_yandex(self, dataset_type, batch_size, fold_idx):
        data_path = root + 'yandex/'
        if dataset_type in ["squirrel", "chameleon"]:
            print(dataset_type.title() + " from Yandex")
        dataset = YandexDataset(dataset_type, fold_idx, data_path)
        data = dataset.graph
        if batch_size > 1:
            if self.sampler == "random":
                self.train_data = RandomNodeSampler(data, num_parts=batch_size, shuffle=True, num_workers=0)
                graph_loader = RandomNodeSampler(data, num_parts=batch_size, shuffle=True, num_workers=0)
            elif self.sampler == "graphsaint":
                self.train_data = GraphSAINTNodeSampler(data, batch_size=self.saint_batch_size, num_steps=batch_size, shuffle=True, num_workers=0)
                graph_loader = RandomNodeSampler(data, num_parts=batch_size, shuffle=True, num_workers=0)
            elif self.sampler == "cluster":
                cluster_data = ClusterData(data, num_parts=batch_size, save_dir=data_path+'cluster/'+dataset_type)
                self.train_data = ClusterLoader(cluster_data, batch_size=1, shuffle=True, num_workers=0)
                graph_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True, num_workers=0)
            else:
                raise ValueError("Unsupported type: " + self.sampler)
            self.val_data = graph_loader
            self.test_data = graph_loader
        else:
            self.train_data = DataLoader([data], batch_size=batch_size, shuffle=True, num_workers=0)
            self.val_data = DataLoader([data], batch_size=batch_size, shuffle=False)
            self.test_data = DataLoader([data], batch_size=batch_size, shuffle=False)
        self.nclass = dataset.num_targets
        self.nfeats = data.num_features
        self.nedgefeats = 1
        self.task_type = "single-class"
        self.ndits = 6
        if dataset_type in ['minesweeper', 'tolokers', 'questions']:
            self.metric = eval_rocauc
        else:
            self.metric = accuracy


    def load_fixed_splits(self, dataset, sub_dataset):
        """ loads saved fixed splits for dataset
        """
        name = dataset
        if sub_dataset and sub_dataset != 'None':
            name += f'-{sub_dataset}'

        if not os.path.exists(root + f'nc/splits/{name}-splits.npy'):
            assert dataset in splits_drive_url.keys()
            gdown.download(id=splits_drive_url[dataset], \
                           output=root + f'nc/splits/{name}-splits.npy', quiet=False) 
        
        splits_lst = np.load(root + f'nc/splits/{name}-splits.npy', allow_pickle=True)
        for i in range(len(splits_lst)):
            for key in splits_lst[i]:
                if not torch.is_tensor(splits_lst[i][key]):
                    splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
        return splits_lst


    def load_nc(self, dataset_type, batch_size, fold_idx):
        data, sub_dataname = load_nc_dataset(dataset_type)
        data = Data(data.graph["node_feat"], data.graph["edge_index"], data.graph["edge_feat"], data.label)
        if batch_size == 1:
            if data.num_edges == 0:
                edge_index = torch.arange(n)
                edge_index = torch.stack((edge_index, edge_index), 0)
            else:
                edge_index = data.edge_index
            n = data.x.shape[-2]
            data.Ahat = get_normalized_adj(edge_index, n, n, selfloop=1, dense=False)
        split_idx_lst = self.load_fixed_splits(dataset_type, sub_dataname)
        data.train_mask = index_to_mask(split_idx_lst[fold_idx]['train'], data.num_nodes)
        data.val_mask = index_to_mask(split_idx_lst[fold_idx]['valid'], data.num_nodes)
        data.test_mask = index_to_mask(split_idx_lst[fold_idx]['test'], data.num_nodes)
        if data.y.min() == -1:
            data.mask = torch.ne(data.y, -1)
        else:
            data.mask = None

        if dataset_type in ['yelp-chi', 'twitch-e', 'ogbn-proteins']:
            if data.y.dim() == 1 or data.y.shape[1] == 1:
                # change -1 instances to 0 for one-hot transform
                # dataset.label[dataset.label==-1] = 0
                data.y = F.one_hot(data.y, data.y.max() + 1).squeeze(1)
        if batch_size > 1:
            if self.sampler == "random":
                self.train_data = RandomNodeSampler(data, num_parts=batch_size, shuffle=True, num_workers=0)
                graph_loader = RandomNodeSampler(data, num_parts=batch_size, shuffle=True, num_workers=0)
            elif self.sampler == "graphsaint":
                self.train_data = GraphSAINTNodeSampler(data, batch_size=self.saint_batch_size, num_steps=batch_size, shuffle=True, num_workers=0)
                graph_loader = RandomNodeSampler(data, num_parts=batch_size, shuffle=True, num_workers=0)
            elif self.sampler == "cluster":
                cluster_data = ClusterData(data, num_parts=batch_size, save_dir=root+'nc/cluster/'+dataset_type)
                self.train_data = ClusterLoader(cluster_data, batch_size=1, shuffle=True, num_workers=0)
                graph_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True, num_workers=0)
            else:
                raise ValueError("Unsupported type: " + self.sampler)
            self.val_data = graph_loader
            self.test_data = graph_loader
        else:
            self.train_data = DataLoader([data], batch_size=batch_size, shuffle=True)
            self.val_data = DataLoader([data], batch_size=batch_size, shuffle=False)
            self.test_data = DataLoader([data], batch_size=batch_size, shuffle=False)
        self.nclass = data.y.max().item() + 1
        self.nfeats = data.num_features
        self.nedgefeats = data.num_edge_features if data.num_edge_features else 1
        self.task_type = "single-class"
        if dataset_type in ['yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius']:
            self.metric = eval_rocauc
        else:
            self.metric = accuracy


    def random_disassortative_splits(self, labels):
        # * 0.6 labels for training
        # * 0.2 labels for validation
        # * 0.2 labels for testing
        num_classes = labels.max() + 1
        num_classes = num_classes.numpy()
        indices = []
        for i in range(num_classes):
            index = torch.nonzero((labels == i)).view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
        percls_trn = int(round(0.6 * (labels.size()[0] / num_classes)))
        val_lb = int(round(0.2 * labels.size()[0]))
        train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        train_mask = index_to_mask(train_index, size=labels.size()[0])
        val_mask = index_to_mask(rest_index[:val_lb], size=labels.size()[0])
        test_mask = index_to_mask(rest_index[val_lb:], size=labels.size()[0])

        return train_mask, val_mask, test_mask


    def load_WebKB(self, dataset_type, batch_size):
        data_path = root + 'webkb/'
        pre_transform = PathPre()
        dataset = GeoData.WebKB(data_path, name=dataset_type, pre_transform=pre_transform)
        if batch_size > 1:
            self.train_data = NeighborLoader(dataset[0], num_neighbors=[50] * 2, batch_size=batch_size, shuffle=True)
            self.val_data = NeighborLoader(dataset[0], num_neighbors=[50] * 2, batch_size=batch_size)
            self.test_data = NeighborLoader(dataset[0], num_neighbors=[50] * 2, batch_size=batch_size)
        else:
            self.train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self.val_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            self.test_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)
 
        self.nclass = dataset.num_classes
        self.nfeats = dataset.num_node_features
        self.nedgefeats = dataset.num_edge_features if dataset.num_edge_features else 1
        self.metric = accuracy
        self.task_type = "single-class"


    def load_wikiNet(self, dataset_type, batch_size):
        data_path = root + 'wikinet/'
        pre_transform = PathPre()
        dataset = GeoData.WikipediaNetwork(data_path, name=dataset_type, geom_gcn_preprocess=True, pre_transform=pre_transform)
        if batch_size > 1:
            data = dataset[0]
            self.train_data = NeighborLoader(dataset[0], num_neighbors=[50] * 2, batch_size=batch_size, shuffle=True, input_nodes=data.train_mask)
            self.val_data = NeighborLoader(dataset[0], num_neighbors=[50] * 2, batch_size=batch_size, input_nodes=data.val_mask)
            self.test_data = NeighborLoader(dataset[0], num_neighbors=[50] * 2, batch_size=batch_size, input_nodes=data.test_mask)
        else:
            self.train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self.val_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            self.test_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.nclass = dataset.num_classes
        self.nfeats = dataset.num_node_features
        self.nedgefeats = dataset.num_edge_features if dataset.num_edge_features else 1
        self.metric = accuracy
        self.task_type = "single-class"


    def load_actor(self, batch_size):
        data_path = root + 'actor/'
        dataset = GeoData.Actor(data_path)
        if batch_size > 1:
            data = dataset[0]
            self.train_data = NeighborLoader(dataset[0], num_neighbors=[50] * 2, batch_size=batch_size, shuffle=True, input_nodes=data.train_mask)
            self.val_data = NeighborLoader(dataset[0], num_neighbors=[50] * 2, batch_size=batch_size, input_nodes=data.val_mask)
            self.test_data = NeighborLoader(dataset[0], num_neighbors=[50] * 2, batch_size=batch_size, input_nodes=data.test_mask)
        else:
            self.train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            self.val_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            self.test_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.nclass = dataset.num_classes
        self.nfeats = dataset.num_node_features
        self.nedgefeats = dataset.num_edge_features if dataset.num_edge_features else 1
        self.metric = accuracy
        self.task_type = "single-class"


    def load_wikiCS(self, batch_size):
        data_path = root + 'wikics/'
        dataset = GeoData.WikiCS(root=data_path)
        self.train_data = DataLoader(dataset=dataset, batch_size=batch_size)
        self.val_data = DataLoader(dataset=dataset, batch_size=batch_size)
        self.test_data = DataLoader(dataset=dataset, batch_size=batch_size)
        self.nclass = dataset.num_classes
        self.nfeats = dataset.num_node_features
        self.nedgefeats = dataset.num_edge_features if dataset.num_edge_features else 1
        self.metric = accuracy
        self.task_type = "single-class"


    def csv_reader(self, file, fold_idx):
        with open(file, "rt") as f:
            data = pd.read_csv(f, delimiter='\t', header=None)
            idx = data.values[fold_idx][0]
        idx = idx.split(',')
        idx = torch.tensor(list(map(int, idx)))
        return idx


    def csv_writer(self, labels, path, seed=42):
        skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
        train_idx_list = []
        val_idx_list = []
        for idx in skf.split(np.zeros(len(labels)), labels):
            train_idx_list.append(idx[0])
            val_idx_list.append(idx[1])

        with open(path + "/train_idx.csv", "wt") as f:
            cw = csv.writer(f)
            cw.writerows(train_idx_list)
        with open(path + "/val_idx.csv", "wt") as f:
            cw = csv.writer(f)
            cw.writerows(val_idx_list)


    def load_tu(self, dataset_type, batch_size, fold_idx, attr=False):
        data_path = root + 'tu/'
        dataset = GeoData.TUDataset(root=data_path, name=dataset_type)
        if attr:
            max_degree = torch.max(degree(dataset.edge_index[0]))
            transform = T.Compose([T.OneHotDegree(int(max_degree.item()))])
            dataset.transform = transform
        flag = os.path.exists(data_path + dataset_type + "/train_idx.csv") and os.path.exists(data_path + dataset_type + "/val_idx.csv")
        if not flag:
            self.csv_writer(dataset.data.y.numpy(), data_path + dataset_type)
        
        train_idx = self.csv_reader(data_path + dataset_type + "/train_idx.csv", fold_idx)
        val_idx = self.csv_reader(data_path + dataset_type + "/val_idx.csv", fold_idx)
        train_dataset = dataset.index_select(train_idx)
        val_dataset = dataset.index_select(val_idx)
        if batch_size > 1:
            self.train_data = DataListLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            self.val_data = DataListLoader(dataset=val_dataset, batch_size=batch_size)
            self.test_data = DataListLoader(dataset=dataset, batch_size=1)
        else:
            self.train_data = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            self.val_data = DataLoader(dataset=val_dataset, batch_size=batch_size)
            self.test_data = DataLoader(dataset=dataset, batch_size=batch_size)
        self.nclass = train_dataset.num_classes
        self.nfeats = train_dataset.num_node_features
        self.nedgefeats = train_dataset.num_edge_features if dataset.num_edge_features else 1
        self.metric = accuracy
        self.task_type = "single-class"


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        if not p.requires_grad:
            continue
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp