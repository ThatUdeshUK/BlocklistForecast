import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import json

from stellargraph import StellarGraph, globalvar
from stellargraph.core import convert
from stellargraph.core.element_data import NodeData, EdgeData

from collections import defaultdict
from itertools import chain


class KnowledgeGraph(StellarGraph):

    """
    Wrapper around StellarGraph 
    
    :param nodes: Nodes of the graph
    :param edges: Edges of the graph
    :param is_directed: If True, the data represents a directed multigraph, otherwise an undirected multigraph.
    :param edge_type_column: The name of the column in the edges to use as the edge type.
    :param dtype: The numpy data-type to use for the features extracted from each of the nodes.
    :param has_public: Whether the nodes contain pubilc domains or not
    :param has_isolates: Whether the nodes contain isolated nodes or not
    :param pruning: dict mapping node types and degree threshold values
    :param extras: DataFrame containing extra data for each node in the graph (i.e. real value of the node, label of the node)
    """
    
    def __init__(
        self,
        nodes=None,
        edges=None,
        *,
        is_directed=False,
        source_column=globalvar.SOURCE,
        target_column=globalvar.TARGET,
        edge_weight_column=globalvar.WEIGHT,
        edge_type_column=None,
        node_type_default=globalvar.NODE_TYPE_DEFAULT,
        edge_type_default=globalvar.EDGE_TYPE_DEFAULT,
        dtype="float32",
        has_public=True,
        has_isolates=True,
        pruning = {},
        extras=None
    ):
        super().__init__(
            nodes=nodes, 
            edges=edges, 
            is_directed=is_directed, 
            source_column=source_column,
            target_column=target_column,
            edge_weight_column=edge_weight_column,
            edge_type_column=edge_type_column,
            node_type_default=node_type_default,
            edge_type_default=edge_type_default,
            dtype=dtype
        )
        
        self.has_public = has_public
        self.has_isolates = has_isolates
        self.pruning = pruning
        
        if extras is not None:
            if type(extras) is pd.DataFrame:
                # TODO - Validate extras
                self.extras = extras
            else:
                raise Exception(f"'extras' should be a {pd.DataFrame} containing real value and label mapping for node ids")
        else:
            self.extras = None

        
    def __subgraph_node_and_edges(self, nodes):
        """
        Compute the nodes and edges of node-induced subgraph implied by ``nodes``.
        
        :param nodes: Nodes in the subgraph
        :returns: Tuple containing nodes and edges of the subgraph for the given nodes.
        """
        node_ilocs = self._nodes.ids.to_iloc(nodes, strict=True)
        node_types = self._nodes.type_of_iloc(node_ilocs)
        node_type_to_ilocs = pd.Series(node_ilocs, index=node_types).groupby(level=0)

        node_frames = {
            type_name: pd.DataFrame(
                self._nodes.features(type_name, ilocs),
                index=self._nodes.ids.from_iloc(ilocs),
            )
            for type_name, ilocs in node_type_to_ilocs
        }

        # FIXME(#985): this is O(edges in graph) but could potentially be optimised to O(edges in
        # graph incident to `nodes`), which could be much fewer if `nodes` is small
        edge_ilocs = np.where(
            np.isin(self._edges.sources, node_ilocs)
            & np.isin(self._edges.targets, node_ilocs)
        )
        edge_frame = pd.DataFrame(
            {
                "id": self._edges.ids.from_iloc(edge_ilocs),
                globalvar.SOURCE: self._nodes.ids.from_iloc(
                    self._edges.sources[edge_ilocs]
                ),
                globalvar.TARGET: self._nodes.ids.from_iloc(
                    self._edges.targets[edge_ilocs]
                ),
                globalvar.WEIGHT: self._edges.weights[edge_ilocs],
            },
            index=self._edges.type_of_iloc(edge_ilocs),
        )
        edge_frames = {
            type_name: df.set_index("id")
            for type_name, df in edge_frame.groupby(level=0)
        }
        
        return node_frames, edge_frames
        
        
    def prune_by_degree(self, node_type, threshold):
        """
        Remove nodes of the given node type that have degree more than the given threshold from the KnowledgeGraph.
        
        :param node_type: A type of nodes that exist in the graph
        :param threshold: Degree threshold
        :returns: KnowledgeGraph without the pruned nodes
        """
        degrees = self._edges.degrees()

        type_node_ilocs = self._nodes.type_range(node_type)
        type_nodes_degree = {node:degrees[node] for node in type_node_ilocs}
        
        all_ilocs = range(self.number_of_nodes()) 
        outliers = dict(filter(lambda x: x[1] >= threshold, type_nodes_degree.items()))
        
        remain = set(all_ilocs).difference(set(outliers.keys()))
        remain_nodes = self.node_ilocs_to_ids(list(remain))
        
        new_nodes, new_edges = self.__subgraph_node_and_edges(remain_nodes)
                
        self.pruning[node_type] = threshold

        return KnowledgeGraph(
            new_nodes, 
            new_edges,
            has_public=self.has_public,
            has_isolates=self.has_isolates,
            pruning=self.pruning,
            extras=self.extras
        )
    
    
    def prune_isolates(self):
        """
        Remove isolated nodes and connected components that have less than two domains from the KnowledgeGraph.

        :returns: KnowledgeGraph without the pruned nodes
        """
        def has_two_or_more_domains(component):
            if len(component) <= 1:
                return []

            domain_count = 0
            for node in component:
                if self.node_type(node) == 'domain':
                    domain_count += 1
                if domain_count > 1:
                    break

            if domain_count > 1:
                return component

            return []

        connected_components = map(has_two_or_more_domains, self.connected_components())#tqdm(, desc='Analysing connected components')
        connected_nodes = list(set(chain.from_iterable(connected_components)))

        new_nodes, new_edges = self.__subgraph_node_and_edges(connected_nodes)
        
        self.has_isolates = False
        
        return KnowledgeGraph(
            new_nodes, 
            new_edges,
            has_public=self.has_public,
            has_isolates=self.has_isolates,
            pruning=self.pruning,
            extras=self.extras
        )
    
    
    def name(self, special=''):
        """
        Get the of the knowledge graph 

        :param special: Prefix to the graph to identify special graph
        :returns: Name of the graph
        """
        name = 'graph' if not special else special

        for node_type in sorted(self.node_types):
            name = f'{name}_{node_type}'
            if node_type in self.pruning.keys():
                name = f'{name}-{self.pruning[node_type]}'

        for edge_type in sorted(self.edge_types):
            name = f'{name}_{edge_type}'

        if self.has_public:
            name = f'{name}_with_public'

        if self.has_isolates:
            name = f'{name}_with_isolated'

        return name

    
    def save(self, data_dir, special=''):
        """
        Save the knowledge graph in a new specific directory in the given data directory 

        :param path: Directory to store the graph
        :param special: Prefix to the graph to identify special graph
        :returns: Path of the stored graph
        """
        name = self.name(special)
        graph_path = os.path.join(data_dir, name)
        os.makedirs(graph_path, exist_ok=True)
        
        for node_type in self.node_types:
            node_ids = self.nodes(node_type=node_type)
            node_features = self.node_features(node_type=node_type)
            
            nodes_df = pd.DataFrame(node_ids, columns=['node_id'])
            features_df = pd.DataFrame(node_features)
            
            for column in features_df.columns:
                nodes_df.insert(len(nodes_df.columns), column, features_df[column])
                
            nodes_df.columns = ['node_id', *range(len(features_df.columns))]

            nodes_df.to_csv(os.path.join(graph_path, f'nodes.{node_type}.csv'), index=None)

        edges_df = pd.DataFrame(self.edges(include_edge_type=True), columns=['source', 'target', 'type'])
        edges_df.to_csv(os.path.join(graph_path, f'edges.csv'), index=None)
        
        self.extras.to_csv(os.path.join(graph_path, f'extras.csv'))
        
        summary = {
            'has_public': self.has_public,
            'has_isolates': self.has_isolates,
            'pruning': self.pruning,
        }
        with open(os.path.join(graph_path, f'summary.json'), 'w') as f:
            j = json.dumps(summary, indent=4)
            print(j, file=f)
        
        return graph_path
        
        
    @staticmethod
    def load(path, split_extras=False):   
        """
        Load KnowledgeGraph instance from a stored graph in the given directory 

        :param path: Directory of the stored graph
        :param split_extras: Whether to load graph extras as a seperate dataset instead of within the graph
        :returns: KnowledgeGraph instance (tuple of KnowledgeGraph instance and DataFrame of extras if split_extras=True)
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
            extras = pd.read_csv(extras[0]).loc[:, ['node_id', 'value', 'type']]
        else:
            extras = None
                        
        kg = KnowledgeGraph(graph_nodes, edges, edge_type_column="type", has_public=has_public, has_isolates=has_isolates, pruning=pruning)
        if split_extras:
            return kg, extras
        
        kg.extras = extras
        return kg
    
    
    @staticmethod
    def load_predicted_extras(path):
        extras = pd.read_csv(os.path.join(path, 'predicted_extras.csv')).loc[:, ['node_id', 'value', 'type']]
        return extras
    
        
    def get_extras(self, node_type=None):
        """
        Get extras of current nodes in the graph 

        :param node_type: A type of nodes that exist in the graph
        :returns: DataFrame with extras
        """
        return self.extras.loc[self.nodes(node_type=node_type), :]
    
    def export(self):
        def get_type_node_features(graph, node_type):
            kg_nodes = pd.DataFrame(graph.node_features(node_type=node_type))
            kg_nodes[f'{node_type}_node'] = graph.nodes(node_type=node_type)
            kg_nodes = kg_nodes.set_index(f'{node_type}_node')
            return kg_nodes

        nodes = {nt: get_type_node_features(self, nt) for nt in self.node_types}
        edges = pd.DataFrame(self.edges(include_edge_type=True), columns=['source', 'target', 'type'])
        
        return nodes, edges, self.extras

    def info(self):
        """
        Return an information string summarizing information on the current graph.
        This includes node and edge type information, availability of public and isolated domains, pruning thresholds and extras.
        
        :returns: Information string
        """
        _info = super().info()
        
        _info += f'\n\n Pruning:'
        
        for n_type, threshold in self.pruning.items():
            _info += f'\n  {n_type} threshold: {threshold}'
        
        _info += f'\n  Has public: {self.has_public}'
        _info += f'\n  Has isolates: {self.has_isolates}'
        
        if self.extras is not None:
            _info += f'\n\n Extras: columns - {list(self.extras.columns)}'

        return _info
        