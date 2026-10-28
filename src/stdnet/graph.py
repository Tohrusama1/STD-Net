"""One graph per user; nodes represent aggregated timestamps."""
import numpy as np
import torch
from torch_geometric.data import Data


def build_graphs(base, features, label_rule='any', anomaly_ratio=None):
    if label_rule not in {'any', 'ratio'}:
        raise ValueError('label_rule must be any or ratio')
    if label_rule == 'ratio' and anomaly_ratio is None:
        raise ValueError('ratio labels require the training anomaly ratio')
    graphs = []
    for user, group in base.groupby('_user', sort=True):
        nodes, labels = [], []
        for _, rows in group.groupby('_time', sort=True):
            nodes.append(features[rows.index.to_numpy()].mean(axis=0))
            attack_rate = rows['_label'].mean()
            labels.append(int(attack_rate > (0.0 if label_rule == 'any' else anomaly_ratio)))
        n = len(nodes)
        edges = [(i, i) for i in range(n)]
        for hop in (1, 2):
            for i in range(n - hop):
                edges.extend([(i, i + hop), (i + hop, i)])
        graphs.append(Data(x=torch.from_numpy(np.stack(nodes)),
                           y=torch.tensor(labels, dtype=torch.float32).view(-1, 1),
                           edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
                           user_id=str(user)))
    if not graphs:
        raise ValueError('No user graphs were constructed')
    return graphs
