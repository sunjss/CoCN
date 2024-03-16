import torch
import torch.nn as nn
import torch.nn.functional as F

def float_bcelogits_loss(y_pred, labels):
    return F.binary_cross_entropy_with_logits(y_pred, labels.type(torch.float32))


def bcelogits_loss(y_pred, labels, label_index, num_nodes):
    # y_pred = y_pred[label_index[0], label_index[1]].flatten()
    return F.binary_cross_entropy_with_logits(y_pred, labels)


def auc_loss(y_pred, labels, label_index, num_nodes):
    pos_edge_index = label_index[:, labels == 1]
    pos_out = y_pred[pos_edge_index[0], pos_edge_index[1]].view(-1, 1)

    num_pos_edges = pos_edge_index.shape[1]
    neg_edge_index = label_index[:, num_pos_edges:]
    neg_out = y_pred[neg_edge_index[0], neg_edge_index[1]].view(num_pos_edges, -1)

    return torch.square(1 - (pos_out - neg_out)).sum() / num_pos_edges