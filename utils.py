import os
import csv
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold

import torch_geometric.datasets as GeoData
from torch_geometric.loader import DataLoader, DataListLoader
import torch_geometric.transforms as T
from torch_geometric.utils import index_to_mask

root = './data/'

class CoCNDataLoader:
    def __init__(self):
        self.adj = None
        self.features = None
        self.labels = None
        self.train_mask, self.val_mask, self.test_mask = None, None, None
        self.nclass = None
        self.idx = None

    def load_data(self, dataset='CORNELL', nbatch=1, fold_idx=0):
        if dataset in ["COLLAB", "IMDB-BINARY", "IMDB-MULTI"]:
            self.load_tu(dataset, nbatch, fold_idx, True)
        elif dataset in ["MUTAG", "PROTEINS", "NCI1"]:
            self.load_tu(dataset, nbatch, fold_idx, False)
        elif dataset in ["CORNELL", "TEXAS", "WISCONSIN"]:
            self.load_WebKB(dataset, nbatch)
        elif dataset in ["SQUIRREL", "CHAMELEON"]:
            self.load_wikiNet(dataset, nbatch)
        elif dataset == "ACTOR":
            self.load_actor(nbatch)
        else:
            raise ValueError("Unsupported type: " + dataset)

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
        dataset = GeoData.WebKB(data_path, name=dataset_type)

        self.train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.val_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.test_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.nclass = dataset.num_classes
        self.nfeats = dataset.num_node_features
        self.nedgefeats = dataset.num_edge_features if dataset.num_edge_features else 1
        self.accuracy = accuracy

    def load_wikiNet(self, dataset_type, batch_size):
        data_path = root + 'wikinet/'
        dataset = GeoData.WikipediaNetwork(data_path, name=dataset_type, geom_gcn_preprocess=True)

        self.train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.val_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.test_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.nclass = dataset.num_classes
        self.nfeats = dataset.num_node_features
        self.nedgefeats = dataset.num_edge_features if dataset.num_edge_features else 1
        self.accuracy = accuracy

    def load_actor(self, batch_size):
        data_path = root + 'actor/'
        dataset = GeoData.Actor(data_path)

        self.train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.val_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.test_data = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.nclass = dataset.num_classes
        self.nfeats = dataset.num_node_features
        self.nedgefeats = dataset.num_edge_features if dataset.num_edge_features else 1
        self.accuracy = accuracy

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
            transform = T.Compose([T.OneHotDegree(dataset.max_num_nodes)])
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
            self.test_data = DataListLoader(dataset=dataset, batch_size=batch_size)
        else:
            self.train_data = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            self.val_data = DataLoader(dataset=val_dataset, batch_size=batch_size)
            self.test_data = DataLoader(dataset=dataset, batch_size=batch_size)
        self.nclass = train_dataset.num_classes
        self.nfeats = train_dataset.num_node_features
        self.nedgefeats = train_dataset.num_edge_features if dataset.num_edge_features else 1
        self.accuracy = accuracy


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
