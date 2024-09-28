import pandas as pd
import numpy as np

import psycopg2
import sqlalchemy as sa

from VT_Forecast.libs import firstseen, graph_gen as gen
from VT_Forecast.libs.graph import KnowledgeGraph


from stellargraph.utils import plot_history
from stellargraph.mapper import CorruptedGenerator, HinSAGENodeGenerator
from stellargraph.layer import DeepGraphInfomax, HinSAGE

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model

from multiprocessing import cpu_count

from tqdm import tqdm

layer_sizes = [48, 48]
batch_size = 1000
num_samples = [10, 5]


def load_fqdn(year, month, from_date, to_date, test_size, verbose=True):
    engine   = sa.create_engine('postgresql+psycopg2://postgres:qcri2020%40@0.0.0.0:5433/VT_Forecast', pool_recycle=3600);
    conn = engine.connect()

    table = firstseen.get_table_name(year, month, from_date, to_date)

    train_query = f"select * from {table} where first_seen::date between '{year}-{month:02}-{from_date:02}' and '{year}-{month:02}-{to_date-test_size:02}';"
    train_expansion = pd.read_sql(train_query, conn).drop_duplicates(subset=['fqdn', 'ip']).dropna().reset_index(drop=True)

    test_query = f"select * from {table} where first_seen::date between '{year}-{month:02}-{to_date-test_size+1:02}' and '{year}-{month:02}-{to_date:02}';"
    test_expansion = pd.read_sql(test_query, conn).drop_duplicates(subset=['fqdn', 'ip']).dropna().reset_index(drop=True)
    
    if verbose:
        print("Raw train resolutions:", len(train_expansion))
        print("Raw Integrated Gradients (HinSAGE).ipynb resolutions:", len(test_expansion))

    conn.close()
    
    return train_expansion, test_expansion


def preprocess_expansions(train_expansion, test_expansion, public_path, verbose=True):
    train_dataset = train_expansion.loc[:, ['fqdn', 'ip']]
    train_dataset.columns = ['domain', 'ip']
    train_dataset = gen.preprocess_resolutions(train_dataset, remove_public=public_path)

    test_dataset = test_expansion.loc[:, ['fqdn', 'ip']]
    test_dataset.columns = ['domain', 'ip']
    test_dataset = gen.preprocess_resolutions(test_dataset, remove_public=public_path)

    if verbose:
        print("Train resolutions after filtering public apexes: {}".format(len(train_dataset)))
        print("Test resolutions after filtering public apexes: {}".format(len(test_dataset)))
    
    return train_dataset, test_dataset


def get_full_dataset(train_dataset, test_dataset):
    full_dataset = pd.concat([train_dataset, test_dataset]).drop_duplicates()
    return gen.get_domain_ip_indexes(full_dataset)


def get_domain_labels(full_dataset, attacker_owned_path, alexa_path):
    return gen.label_domains(full_dataset.loc[:, ['domain', 'domain_node', 'apex']], attacker_owned_path, alexa_path)


def gen_full_graph(full_dataset, train_dataset, test_dataset, labelled_domains, asn_path, similarity, pruning=True, verbose=True):
    # Domain extraction
    domains = full_dataset.loc[:, ['domain', 'domain_node']].drop_duplicates()

    if verbose:
        print("No. of domain nodes: {}\n".format(len(labelled_domains)))
        print(labelled_domains.groupby('type').size())
        
    domain_features = gen.get_domain_features(domains, feature_type='lexical')

    # Remove Domains without features
    labelled_domains = labelled_domains[labelled_domains.domain.isin(domain_features['domain'].values)]
    full_dataset = full_dataset[full_dataset.domain.isin(domain_features['domain'].values)]
    
    # IP extraction
    ips = full_dataset.loc[:, ['ip_node', 'ip']].drop_duplicates()
    
    if verbose:
        print("No. of IP nodes: {}\n".format(len(ips)))
        
    ip_features = gen.get_ip_features(ips, asn_path=asn_path)

    # Remove IPs without features
    ips = ips[ips.ip.isin(ip_features['ip'].values)]
    full_dataset = full_dataset[full_dataset.ip.isin(ip_features['ip'].values)]
    
    graph_nodes, extras = gen.get_nodes_and_extras(
        domains=labelled_domains, 
        ips=ips,
        domain_features=domain_features, 
        ip_features=ip_features
    )
    
    
    # Edge processing
    similar_domains = similarity.join(domains.set_index('domain'), on='domain')
    similar_domains = similar_domains.join(domains.set_index('domain'), on='matched', rsuffix='2').dropna()
    similar_domains = similar_domains.loc[:, ['domain_node', 'domain_node2']]
    
    edges = gen.get_edges(resolutions=full_dataset, similar_domains=similar_domains, apex_edge=True)
    
    G = KnowledgeGraph(graph_nodes, edges, edge_type_column="type", has_public=False, extras=extras)
    
    if pruning:
        G_ip_pruned = G.prune_by_degree('ip', threshold=1500)
        G = G_ip_pruned.prune_isolates()

    if verbose:
        print(G.info())
        
    return G


def train_test_graph_split(full_graph, full_dataset, train_dataset, test_dataset, verbose=True):
    full_domain_nodes = list(full_graph.nodes(node_type='domain'))
    full_ip_nodes = list(full_graph.nodes(node_type='ip'))

    domain_nodes = full_dataset.loc[:, ['domain_node', 'domain']]
    domain_nodes = domain_nodes[domain_nodes.domain_node.isin(full_domain_nodes)]
    
    train_domain_nodes = domain_nodes[domain_nodes.domain.isin(train_dataset['domain'].values)]
    test_domain_nodes = domain_nodes[domain_nodes.domain.isin(test_dataset['domain'].values)]

    train_ip_nodes = full_dataset.loc[:, ['ip_node', 'ip']]
    train_ip_nodes = train_ip_nodes[train_ip_nodes.ip_node.isin(full_ip_nodes)]
    train_ip_nodes = train_ip_nodes[train_ip_nodes.ip.isin(train_dataset['ip'].values)]

    train_domains = list(train_domain_nodes['domain_node'].drop_duplicates().values)
    train_ips = list(train_ip_nodes['ip_node'].drop_duplicates().values)

    train_graph = full_graph.subgraph(train_domains + train_ips)
    
    if verbose:
        print(train_graph.info())
    
    # Extract Integrated Gradients (HinSAGE).ipynb nodes
    train_node_ids = list(train_domain_nodes['domain_node'].drop_duplicates().values)
    test_node_ids = list(test_domain_nodes['domain_node'].drop_duplicates().values)
        
    return train_graph, train_node_ids, test_node_ids


def train_hinsage(train_graph, verbose=True):
    epochs = 10
    
    # Init HinSAGE
    generator = HinSAGENodeGenerator(train_graph, batch_size=batch_size, num_samples=num_samples, head_node_type="domain")

    base_model = HinSAGE(
        layer_sizes=layer_sizes, activations=["relu"] * len(layer_sizes), generator=generator
    )

    corrupted_generator = CorruptedGenerator(generator)
    node_gen = corrupted_generator.flow(train_graph.nodes(node_type='domain'))
    
    # Train model
    infomax = DeepGraphInfomax(base_model, corrupted_generator)
    x_in, x_out = infomax.in_out_tensors()

    model = Model(inputs=x_in, outputs=x_out)
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=Adam(lr=1e-3))

    es = EarlyStopping(monitor="loss", min_delta=0, patience=20)
    history = model.fit(node_gen, epochs=epochs, verbose=1 if verbose else 0, callbacks=[es], workers=cpu_count(), use_multiprocessing=True)
    plot_history(history)
    
    # Build embedding model
    x_emb_in, x_emb_out = base_model.in_out_tensors()

    if generator.num_batch_dims() == 2:
        x_emb_out = tf.squeeze(x_emb_out, axis=0)

    emb_model = Model(inputs=x_emb_in, outputs=x_emb_out)
    
    return emb_model


def gen_embeddings(emb_model, full_graph, test_node_ids, labelled_domains, verbose=True):
    generator = HinSAGENodeGenerator(full_graph, batch_size=batch_size, num_samples=num_samples, head_node_type="domain")
    node_embeddings = emb_model.predict(generator.flow(test_node_ids), verbose=1 if verbose else 0)
        
    labelled_test = labelled_domains[labelled_domains.domain_node.isin(test_node_ids)].loc[:, ['domain_node', 'type']]
    labelled_test.columns = ['node_id', 'type']
        
    embeddings = pd.DataFrame(node_embeddings)
    embeddings.insert(0, 'node_id', test_node_ids)
    embeddings = embeddings.join(labelled_test.set_index('node_id'), on='node_id').fillna('n_a')
    
    labelled_embeddings = embeddings[(embeddings.type == 'malicious') | (embeddings.type == 'benign')]
            
    if verbose:
        print("Shape of the embeddings: {}".format(node_embeddings.shape))
        print("No. of Integrated Gradients (HinSAGE).ipynb labels", len(labelled_test))
        print("Labelled Embedding Population :", labelled_embeddings.shape)
        print("Labelled Embedding Dist:\n", labelled_embeddings.groupby('type').size())
        
    return labelled_embeddings


def save_embeddings(path, train_embeddings, test_embeddings, table, test_size, verbose=True):
    embeddings_path = f'{path}/node_embeddings.hinsage.{table}_{test_size}.csv'

    train_embeddings['is_train'] = [True] * len(train_embeddings)
    test_embeddings['is_train'] = [False] * len(test_embeddings)
    
    embeddings = pd.concat([train_embeddings, test_embeddings]).reset_index(drop=True)

    embeddings.to_csv(embeddings_path, index=None)
    
    if verbose:
        print("Wrote to:", embeddings_path)
        
        
def extract_continous_embeddings(year, month, from_date, to_date, test_size, paths, verbose = False):
    public_path = paths['public_path']
    asn_path = paths['asn_path']
    alexa_path = paths['alexa_path']
    attacker_owned_path = paths['attacker_owned_path']
    embedding_path = paths['embedding_path']
    similarity_path = paths['similarity_path']

    train_expansion, test_expansion = load_fqdn(year, month, from_date, to_date, test_size, verbose)
    train_dataset, test_dataset = preprocess_expansions(train_expansion, test_expansion, public_path, verbose)
    full_dataset = get_full_dataset(train_dataset, test_dataset)
    labelled_domains = get_domain_labels(full_dataset, attacker_owned_path, alexa_path )

#     similar_domains = None
    similar_domains = pd.read_csv(similarity_path)
    graph = gen_full_graph(full_dataset, train_dataset, test_dataset, labelled_domains, asn_path, similar_domains, True, verbose)
    train_graph, train_node_ids, test_node_ids = train_test_graph_split(graph, full_dataset, train_dataset, test_dataset, verbose=False)
    
    emb_model = train_hinsage(train_graph, verbose)

    train_embeddings = gen_embeddings(emb_model, graph, train_node_ids, labelled_domains, verbose)
    test_embeddings = gen_embeddings(emb_model, graph, test_node_ids, labelled_domains, verbose)

    table = firstseen.get_table_name(year, month, from_date, to_date)
    save_embeddings(embedding_path, train_embeddings, test_embeddings, table, test_size, verbose)