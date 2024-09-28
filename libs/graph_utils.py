import os

import numpy as np
import pandas as pd

import json


def load_graph(path, not_multigraph=True):
    """
    Load stored graph in the given directory 

    :param path: Directory of the stored graph
    :returns: tuple containing nodes, edges, has_public, has_isolates, pruning information and extras
    """
    with open(os.path.join(path, 'summary.json'), 'r') as json_data:
        summary = json.load(json_data)

    has_public = summary['has_public']
    has_isolates = summary['has_isolates']
    pruning = summary['pruning']

    directory = [os.path.join(path, f) for f in os.listdir(path)]
    files = [f for f in directory if os.path.isfile(f)]

    edges = [f for f in files if 'edges' in f]
    if len(edges) > 0:
        edges = pd.read_csv(edges[0])
    else:
        raise Exception("No 'edges.csv' file found in the path")

    if not_multigraph:
        edges_sorted = pd.DataFrame(np.sort(edges.loc[:, ['source', 'target']].values, axis=1), columns=['source', 'target'])
        edges_sorted['type'] = edges.type

        edges_sorted = edges_sorted.sort_values(['source', 'target', 'type'])
        print(f"Remove parallel edges: {edges_sorted[edges_sorted.duplicated(['source', 'target'])].value_counts('type')}")
        edges = edges_sorted.drop_duplicates(['source', 'target'])

    graph_nodes = {}
    nodes = [f for f in files if 'nodes' in f]
    if len(nodes) > 0:
        for n_type_file in nodes:
            n_type = n_type_file.split('.')[-2]
            nodes_df = pd.read_csv(n_type_file, index_col=0)
            graph_nodes[n_type] = nodes_df
    else:
        raise Exception("No 'nodes.<node_type>.csv' files found in the path")

    extras = [f for f in files if 'extras' in f]
    if len(extras) > 0:
        extras = pd.read_csv(extras[0])#, index_col=0)
    else:
        extras = None

    return graph_nodes, edges, has_public, has_isolates, pruning, extras