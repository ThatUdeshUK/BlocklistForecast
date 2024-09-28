import os
import sys
import copy

import datetime
import calendar

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
# cuda_device=7
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# device = tf.config.list_physical_devices('GPU')[0]
# tf.config.experimental.set_memory_growth(device, True)
# print(device)
    
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from VT_Forecast.libs import graph_gen as gen, firstseen, graph
from sklearn.model_selection import TimeSeriesSplit
from stellargraph import utils, mapper, layer
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tqdm.notebook import tqdm 
tqdm.pandas()

from multiprocessing import cpu_count


expansion_dir = '../data/continous/expansions'
public_path = '../data/extra/pub_domains_20201029'
#asn_path = '../data/extra/geo_20210103_GeoIPASNum.dat'
asn_path = '../data/extra/geo_20201018_GeoIPASNum.dat'

alexa_paths = [
     '../data/alexa/2020-10-01_2020-10-31-alexa_domains.csv',
#     '../data/alexa/2020-11-01_2020-11-30-alexa_domains.csv',
#     '../data/alexa/2020-12-01_2020-12-31-alexa_domains.csv',
    #'../data/alexa/2021-01-01_2021-01-31-alexa_domains.csv',
    # '../data/alexa/2021-02-01_2021-02-28-alexa_domains.csv',
    # '../data/alexa/2021-03-01_2021-03-31-alexa_domains.csv',
    # '../data/alexa/2021-04-01_2021-04-30-alexa_domains.csv',
]
attacker_owned_paths = [
     '../data/longitudinal/2020-10-01_2020-10-31-filtered_attacker_apexes.csv',
#     '../data/longitudinal/2020-11-01_2020-11-30-filtered_attacker_apexes.csv',
#     '../data/longitudinal/2020-12-01_2020-12-31-filtered_attacker_apexes.csv',
    #'../data/longitudinal/2021-01-01_2021-01-31-filtered_attacker_apexes.csv',
    # '../data/longitudinal/2021-02-01_2021-02-28-filtered_attacker_apexes.csv',
    # '../data/longitudinal/2021-03-01_2021-03-31-filtered_attacker_apexes.csv',
    # '../data/longitudinal/2021-04-01_2021-04-30-filtered_attacker_apexes.csv',
]
cos_sim_paths = [
     '../data/cos_sim_domains_0_8.20_oct.csv',
#     '../data/cos_sim_domains_0_8.20_nov.csv',
#     '../data/cos_sim_domains_0_8.20_dec.csv',
    #'../data/cos_sim_domains_0_8.21_jan.csv',
    # '../data/cos_sim_domains_0_8.21_feb.csv',
    # '../data/cos_sim_domains_0_8.21_mar.csv',
    # '../data/cos_sim_domains_0_8.21_apr.csv',
]
layer_sizes = [48, 48]
batch_size = 1000
num_samples = [10, 5]
epochs = 10
no_workers = int(cpu_count() * 0.9)

k = 5
max_train = 10
test_size = 4

from_date = (2020, 10, 1)
to_date = (2020, 10, 30)


def get_files(from_date, to_date):
    year, month, day = from_date
    to_year, to_month, to_day = to_date
    
    if year != to_year or month != to_month:
        raise NotImplemented("Year and month should match!")
        
    return firstseen.list_expansion_files(expansion_dir, day, to_day, f'{year}-{month:02}')


def get_new_domain_ip_stats(resolutions, existing_domain_map=None, existing_ip_map=None):
    def get_new_column_stats(column, existing):
        new_values = []
        for value in np.unique(column):
            if value not in existing:
                new_values.append(value)

        return len(np.unique(column)), len(new_values)
    
    domain_stats = get_new_column_stats(resolutions['domain'].values, existing_domain_map)
    ip_stats = get_new_column_stats(resolutions['ip'].values, existing_ip_map)
    return *domain_stats, *ip_stats

t_files = get_files(from_date, to_date)
print(t_files)
tscv = TimeSeriesSplit(gap=0, max_train_size=max_train, n_splits=k, test_size=test_size)


splits = []
for train_index, test_index in tscv.split(t_files):
    print(train_index + from_date[2], test_index + from_date[2])
    splits.append(([t_files[i] for i in train_index], [t_files[i] for i in test_index]))
    
    
split_expansions = []
for i, (train_files, test_files) in tqdm(enumerate(splits)):
    train_expansion = pd.concat([pd.read_csv(file).loc[:, ['fqdn', 'ip']] for file in train_files]).drop_duplicates()
    test_expansion = pd.concat([pd.read_csv(file).loc[:, ['fqdn', 'ip']] for file in test_files]).drop_duplicates()
    
    all_resolutions = pd.concat([train_expansion, test_expansion])
    duplicates = all_resolutions[all_resolutions.duplicated()]
    unique_test_expansion = pd.concat([test_expansion, duplicates]).drop_duplicates(keep=False)
    
    print(i, " - Train Size:", len(train_expansion), ", Test Size:", len(unique_test_expansion))
    split_expansions.append((train_expansion, unique_test_expansion))


split_datasets = []
domain_ip_stats = []
for i, (train_expansion, test_expansion) in tqdm(enumerate(split_expansions)):
    train_dataset = train_expansion.loc[:, ['fqdn', 'ip']]
    train_dataset.columns = ['domain', 'ip']
    train_dataset = gen.preprocess_resolutions(train_dataset, remove_public=public_path)

    test_dataset = test_expansion.loc[:, ['fqdn', 'ip']]
    test_dataset.columns = ['domain', 'ip']
    test_dataset = gen.preprocess_resolutions(test_dataset, remove_public=public_path)

    print("Filtering public apexes: Train: {0}, Test: {1}".format(len(train_dataset), len(test_dataset)))
    
    train_dataset, train_domain_map, train_ip_map = gen.get_dynamic_domain_ip_indexes(train_dataset)
    domain_ip_stats.append(get_new_domain_ip_stats(test_dataset, train_domain_map, train_ip_map))
    test_dataset = gen.get_dynamic_domain_ip_indexes(test_dataset, train_domain_map, train_ip_map)[0] 
    
    split_datasets.append((train_dataset, test_dataset))
    
    
attacker_data = []
for attacker_owned_path in tqdm(attacker_owned_paths):
    attacker_data.append(pd.read_csv(attacker_owned_path, names=['domain']))
attacker_df = pd.concat(attacker_data).drop_duplicates().reset_index(drop=True)


alexa_data = []
for alexa_path in tqdm(alexa_paths):
    alexa_data.append(pd.read_csv(alexa_path))
alexa_df = pd.concat(alexa_data).drop_duplicates().reset_index(drop=True)


cos_sim = []
for cos_sim_path in tqdm(cos_sim_paths):
    cos_sim.append(pd.read_csv(cos_sim_path))
similar_domains = pd.concat(cos_sim).drop_duplicates().reset_index(drop=True)


split_embeddings = []

for i, (train_dataset, test_dataset) in tqdm(enumerate(split_datasets)):
    print("Split:", i)
    print("Build train graph")

#     if i in [1, 2, 3, 4]:
#         print("Skip split:", i)
#         continue
    
    """Build DNS graph"""
    train_domains = train_dataset.loc[:, ['domain', 'domain_node']].drop_duplicates()
    labelled_train_domains = gen.label_domains(train_dataset.loc[:, ['domain', 'domain_node', 'apex']], attacker_df, alexa_df)

    train_domain_features = gen.get_domain_features(train_domains, feature_type='lexical')
    labelled_domains = labelled_train_domains[labelled_train_domains.domain.isin(train_domain_features['domain'].values)]
    train_dataset = train_dataset[train_dataset.domain.isin(train_domain_features['domain'].values)]

    train_ips = train_dataset.loc[:, ['ip_node', 'ip']].drop_duplicates()
    train_ip_features = gen.get_ip_features(train_ips, asn_path=asn_path)
    train_ips = train_ips[train_ips.ip.isin(train_ip_features['ip'].values)]
    train_dataset = train_dataset[train_dataset.ip.isin(train_ip_features['ip'].values)]

    graph_nodes, extras = gen.get_nodes_and_extras(
        domains=labelled_train_domains, 
        ips=train_ips,
        domain_features=train_domain_features, 
        ip_features=train_ip_features
    )
    
    train_similar_domains = similar_domains.join(labelled_domains.set_index('domain'), on='domain')
    train_similar_domains = train_similar_domains.join(
        labelled_domains.set_index('domain'), on='matched', rsuffix='2'
    ).dropna()
    train_similar_domains = train_similar_domains.loc[:, ['domain_node', 'domain_node2']]

    edges = gen.get_edges(resolutions=train_dataset, similar_domains=train_similar_domains, apex_edge=True)
    
    train_graph = graph.KnowledgeGraph(graph_nodes, edges, edge_type_column="type", has_public=False, extras=extras)
    train_graph = train_graph.prune_by_degree('ip', threshold=1500)
    train_graph = train_graph.prune_isolates()

    # print(train_graph.info())
    print("Train emb. model")
    """Train emb. model"""
    generator = mapper.HinSAGENodeGenerator(train_graph, batch_size=batch_size, 
                                            num_samples=num_samples, head_node_type="domain")

    base_model = layer.HinSAGE(
        layer_sizes=layer_sizes, activations=["relu"] * len(layer_sizes), generator=generator
    )

    corrupted_generator = mapper.CorruptedGenerator(generator)
    corrupted_gen = corrupted_generator.flow(train_graph.nodes(node_type='domain'))

    infomax = layer.DeepGraphInfomax(base_model, corrupted_generator)
    x_in, x_out = infomax.in_out_tensors()

    model = tf.keras.Model(inputs=x_in, outputs=x_out)
    model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=tf.keras.optimizers.Adam(lr=1e-3))

    es = tf.keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=20)
    history = model.fit(corrupted_gen, epochs=epochs, verbose=1, callbacks=[es], workers=no_workers, use_multiprocessing=True)
    utils.plot_history(history)
    
    print("Gen emb. model")
    """Generate emb. model"""
    x_emb_in, x_emb_out = base_model.in_out_tensors()
    if generator.num_batch_dims() == 2:
        x_emb_out = tf.squeeze(x_emb_out, axis=0)
    emb_model = tf.keras.Model(inputs=x_emb_in, outputs=x_emb_out)

    train_node_ids = train_graph.nodes(node_type='domain')

    print("Train emb.")
    """Generate emb. for train"""
    labelled_train = labelled_train_domains[labelled_train_domains.domain_node.isin(train_node_ids)].loc[:, ['domain_node', 'type']]
    labelled_train.columns = ['node_id', 'type']
    # print(len(labelled_train))

    train_node_embeddings = emb_model.predict(generator.flow(train_node_ids), verbose=1, workers=8, 
                                              use_multiprocessing=False)

    embeddings = pd.DataFrame(train_node_embeddings)
    embeddings.insert(0, 'node_id', train_node_ids)
    embeddings = embeddings.join(labelled_train.set_index('node_id'), on='node_id').fillna('n_a')

    train_embeddings = embeddings[(embeddings.type == 'malicious') | (embeddings.type == 'benign')]
    print("Population :", train_embeddings.shape)
    print(train_embeddings.groupby('type').size())
    
    print("Build Integrated Gradients (HinSAGE).ipynb graph")
    """Build Integrated Gradients (HinSAGE).ipynb DNS graph"""
    test_domains = test_dataset.loc[:, ['domain', 'domain_node']].drop_duplicates()
    labelled_test_domains = gen.label_domains(test_dataset.loc[:, ['domain', 'domain_node', 'apex']], attacker_df, alexa_df)

    test_domain_features = gen.get_domain_features(test_domains, feature_type='lexical')
    labelled_test_domains = labelled_test_domains[labelled_test_domains.domain.isin(test_domain_features['domain'].values)]
    test_dataset = test_dataset[test_dataset.domain.isin(test_domain_features['domain'].values)]

    test_ips = test_dataset.loc[:, ['ip_node', 'ip']].drop_duplicates()
    test_ip_features = gen.get_ip_features(test_ips, asn_path=asn_path)
    test_ips = test_ips[test_ips.ip.isin(test_ip_features['ip'].values)]
    test_dataset = test_dataset[test_dataset.ip.isin(test_ip_features['ip'].values)]

    test_graph_nodes, test_extras = gen.get_nodes_and_extras(
        domains=labelled_test_domains, 
        ips=test_ips,
        domain_features=test_domain_features, 
        ip_features=test_ip_features
    )
    
    test_similar_domains = similar_domains.join(labelled_test_domains.set_index('domain'), on='domain')
    test_similar_domains = test_similar_domains.join(labelled_test_domains.set_index('domain'), on='matched', rsuffix='2').dropna()
    test_similar_domains = test_similar_domains.loc[:, ['domain_node', 'domain_node2']]

    test_edges = gen.get_edges(resolutions=test_dataset, similar_domains=test_similar_domains, apex_edge=True)
    
    eval_graph_nodes = {}

    for key, value in  test_graph_nodes.items():
        type_nodes = train_graph.nodes(node_type=key)

        new_nodes = pd.concat([graph_nodes[key].loc[type_nodes, :], value])
        new_nodes = new_nodes[~(new_nodes.index.duplicated())]

        eval_graph_nodes[key] = new_nodes

    eval_extras = pd.concat([extras, test_extras])
    existing_edges = pd.DataFrame(train_graph.edges(include_edge_type=True), columns=['source', 'target', 'type'])
    eval_edges = pd.concat([existing_edges, test_edges]).reset_index(drop=True)

    test_graph = graph.KnowledgeGraph(eval_graph_nodes, eval_edges, edge_type_column="type", has_public=False, extras=eval_extras)
    test_graph = test_graph.prune_by_degree('ip', threshold=1500)
    test_graph = test_graph.prune_isolates()
    
    print("Test emb.")
    """Generate emb. for Integrated Gradients (HinSAGE).ipynb"""
    generator = mapper.HinSAGENodeGenerator(test_graph, batch_size=batch_size, num_samples=num_samples, 
                                            head_node_type="domain")

    test_domain_nodes = test_dataset[test_dataset.domain_node.isin(test_graph.nodes(node_type='domain'))]

    node_ids = list(test_domain_nodes['domain_node'].drop_duplicates().values)
    print(len(node_ids))

    node_embeddings = emb_model.predict(generator.flow(node_ids), verbose=1, workers=8, use_multiprocessing=False)
    print("Shape of the embeddings: {}".format(node_embeddings.shape))

    labelled_test = test_graph.extras.loc[:, ['node_id', 'type']]
    print(len(labelled_test))

    embeddings = pd.DataFrame(node_embeddings)
    embeddings.insert(0, 'node_id', node_ids)
    embeddings = embeddings.join(labelled_test.set_index('node_id'), on='node_id').fillna('n_a')
    
    test_embeddings = embeddings[(embeddings.type == 'malicious') | (embeddings.type == 'benign')]

    print("Population :", test_embeddings.shape)
    print(test_embeddings.groupby('type').size())
    
#     split_embeddings.append((train_embeddings, test_embeddings))
    train_path = f'../data/val/{"_".join(map(str, from_date))}-{"_".join(map(str, to_date))}_size_{test_size}_split_{i}_train.csv'
    test_path = f'../data/val/{"_".join(map(str, from_date))}-{"_".join(map(str, to_date))}_size_{test_size}_split_{i}_test.csv'
    
    train_embeddings.to_csv(train_path, index=None)
    test_embeddings.to_csv(test_path, index=None)
    print("====================\n")