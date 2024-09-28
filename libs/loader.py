import numpy as np
import pandas as pd

from typing import Optional, Callable

from libs import graph_utils

import torch
from torch_geometric.data import (InMemoryDataset, HeteroData, Data)


class DNS(InMemoryDataset):
    def __init__(self, nodes, edges, extras,  num_test=0.3, num_val=0.2, balance_gt=False, fill_missing=None,
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        super().__init__("", transform, pre_transform)
        processed, self.extras = _process(nodes, edges, extras, num_test, num_val, balance_gt, fill_missing)
        self.data, self.slices = self.collate([processed])
    
    @staticmethod
    def from_saved(root, num_test=0.3, num_val=0.2, balance_gt=False, fill_missing=None,
                 transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        nodes, edges, _, _, _, extras = graph_utils.load_graph(root)
        return DNS(nodes, edges, extras, num_test, num_val, balance_gt, fill_missing, transform, pre_transform)
        
    @property
    def num_classes(self) -> int:
        return 2

    def to_homogeneous(self, transform=None, method='padded') -> Data:
        return to_homogeneous(self.__getitem__(0), transform=transform)


def _process(graph_nodes, edges, extras, num_test=0.3, num_val=0.2, balance_gt=False, fill_missing=None):
    data = HeteroData()

    node_ilocs = {}
    for node_type, node_features in graph_nodes.items():
        node_type = node_type + "_node"
        for x, idx in enumerate(node_features.index):
            node_ilocs[idx] = (x, node_type)
            
        if fill_missing is not None and node_features.shape[1] == 0:
            if fill_missing == 'unit':
                node_features['filled'] = 1
            else:
                raise Exception("Not Implemented")

        data[node_type].num_nodes = node_features.shape[0]
        data[node_type].x = torch.from_numpy(node_features.values).float()
        if node_type == 'domain_node':
            extras = pd.DataFrame(extras[extras.node_id.isin(node_features.index)])
            extras['node_iloc'] = extras['node_id'].apply(lambda i: node_ilocs[i][0] if i in node_ilocs else None)
            extras = extras.dropna()

            labels = extras.sort_values('node_iloc')['type'] \
                .apply(lambda i: 1 if i == 'malicious' else (0 if i == 'benign' else 2))

            data[node_type].y = torch.from_numpy(labels.values)
            labeled = labels.values < 2
            labeled_indices = labeled.nonzero()[0]

            # balance benign and mal nodes
            if balance_gt:
                mal_nodes = (data[node_type].y == 1).nonzero().t()[0]
                ben_nodes = (data[node_type].y == 0).nonzero().t()[0]

                min_count = min(len(mal_nodes), len(ben_nodes))

                mal_nodes = mal_nodes[torch.randperm(len(mal_nodes))[:min_count]]
                ben_nodes = ben_nodes[torch.randperm(len(ben_nodes))[:min_count]]
                labeled_indices = torch.concat([mal_nodes, ben_nodes], axis=-1)

                new_labels = torch.full((data[node_type].y.size(0),), 2, dtype=torch.long)
                new_labels[labeled_indices] = data[node_type].y[labeled_indices]
                data[node_type].y = new_labels

            n_nodes = len(labeled_indices)
            perm = torch.randperm(n_nodes)

            test_idx = labeled_indices[perm[:int(n_nodes * num_test)]]
            val_idx = labeled_indices[perm[int(n_nodes * num_test):int(n_nodes * (num_test + num_val))]]
            train_idx = labeled_indices[perm[int(n_nodes * (num_test + num_val)):]]

            for v, idx in [('train', train_idx), ('test', test_idx), ('val', val_idx)]:
                mask = torch.zeros(data[node_type].num_nodes, dtype=torch.bool)
                mask[idx] = True
                data[node_type][f'{v}_mask'] = mask

    for edge_type, edge_data in edges.groupby('type'):
        from_type = edge_data['source'].apply(lambda i: node_ilocs[i][1]).drop_duplicates().values[0]
        to_type = edge_data['target'].apply(lambda i: node_ilocs[i][1]).drop_duplicates().values[0]

        edge_data['source'] = edge_data['source'].apply(lambda i: node_ilocs[i][0])
        edge_data['target'] = edge_data['target'].apply(lambda i: node_ilocs[i][0])
        edge_data = torch.from_numpy(edge_data.loc[:, ['source', 'target']].values.T)

        data[from_type, edge_type, to_type].edge_index = edge_data

    return data, extras

        
def count_labels(labels):
    label_map = {0: 'Malicious', 1: 'Benign', 2: 'N/A'}
    return pd.DataFrame(labels.cpu())[0].apply(lambda x: label_map[x]).value_counts()


def to_homogeneous(data, transform=None, method='padded'):
    if method == 'padded':
        return _to_padded_homogeneous(data, transform)
    elif method == 'metapath':
        return _to_metapath_homogeneous(data)
    raise NotImplementedError("Implement metapath, linear and padded homogeneous conversion.")
    
    
def _to_metapath_homogeneous(data):
    metapaths = [[('domain_node', 'ip_node'), ('ip_node', 'domain_node')]]
    data = T.AddMetaPaths(metapaths=metapaths)(data)

    del data['ip_node']
    del data[('domain_node', 'resolves', 'ip_node')]
    del data[('ip_node', 'rev_resolves', 'domain_node')]

    data = data.to_homogeneous()
    return data
    
    
def _to_padded_homogeneous(data, transform=None, with_mapping=True, concat_features=True):
    data = data.clone()
    device = 'cuda' if data.is_cuda else 'cpu'

    features_shape = {nt: node_features.shape[1] for nt, node_features in data.x_dict.items()}
    if concat_features:
        if all([list(features_shape.values())[0] == v for v in features_shape.values()]):
            print("Feature dimensions are equal! Not concatenating")
            concat_features = False
        else:
            features_shape = sum([v for v in features_shape.values()])
        
    mask_types = [k for k in list(data['domain_node'].keys()) if 'mask' in k]
    masks = {k: [] for k in mask_types}
    y = []

    edge_map = {}
    if with_mapping:
        node_map = {node_type: i for i, node_type in enumerate(data.node_types)}
        edge_map = {i: (node_map[edge_type[0]], node_map[edge_type[2]]) for i, edge_type in enumerate(data.edge_types)}

    l_padding, feat_list = 0, []
    for node_type, node_features in data.x_dict.items():
        if 'y' in data[node_type]:
            y.append(data[node_type].y)
            for mask_type in mask_types:
                masks[mask_type].append(data[node_type][mask_type])
        else:
            y.append(torch.full((node_features.shape[0],), 2, dtype=torch.long).to(device))
            for mask_type in mask_types:
                masks[mask_type].append(torch.zeros(node_features.shape[0], dtype=torch.bool).to(device))

        if concat_features:
            node_features = node_features.cpu().numpy()
            r_padding = features_shape - node_features.shape[1] - l_padding
            features = []
            for node_feature in node_features:
                resized = np.pad(node_feature, (l_padding, r_padding), 'constant', constant_values=(0, 0))
                features.append(resized)

            l_padding += node_features.shape[1]
            data[node_type].x = torch.from_numpy(np.array(features)).float()
        else:
            feat_list.append(node_features)

    data = data.to_homogeneous(add_edge_type=True, add_node_type=True)

    for mask_type, mask in masks.items():
        data[mask_type] = torch.cat(mask)
    data.y = torch.cat(y)
    num_nodes = data.num_nodes

    if transform is not None:
        transform(data)
        
    if with_mapping:
        data.edge_map = edge_map
        data.num_nodes = num_nodes
        
    if (not concat_features) and 'x' not in data:
        data.x = feat_list

    return data
