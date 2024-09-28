import pandas as pd
import numpy as np

from pandarallel import pandarallel
pandarallel.initialize()

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import category_encoders as ce

from libs.domain_utils import get_2ld
from libs import maxmind_utils
from libs.features.lexical import lexical_features as lf
from libs.features.ringer import ringer_features as rf
from libs.features.ringer.tlds import get_tlds
from libs.data_utils import load_path_or_df


def log(msg, verbose=True):
    if verbose:
        print(msg)

        
# def __get_node_indexes(node_type, column, start=0):
#     """ Extract ids for each unique node """
#     column_map = {l: i for i, l in enumerate(np.unique(column), start=start)}
#     return [node_type + str(column_map[target]) for target in column]


def __get_node_indexes(node_type, column, existing=None, time=None):
    """ Extract ids for each unique node """
    if existing != None:
        new_values = []
        for value in np.unique(column):
            if value not in existing:
                new_values.append(value)

        if time is None:
            column_map = {l: i for i, l in enumerate(np.unique(new_values), start=max(existing.values()) + 1)}
        else:
            column_map = {l: (i, time) for i, l in enumerate(np.unique(new_values), start=max([e[0] for e in existing.values()]) + 1)}
        column_map = {**existing, **column_map}
    else:
        if time is None:
            column_map = {l: i for i, l in enumerate(np.unique(column))}
        else:
            column_map = {l: (i, time) for i, l in enumerate(np.unique(column))}
    
    if time is None:
        return [node_type + str(column_map[target]) for target in column], column_map
    else:
        return [node_type + str(column_map[target][0]) for target in column], column_map


def preprocess_resolutions(resolutions, remove_public=None, only_domain_ip=True):
    columns = ['domain', 'ip'] if only_domain_ip else list(resolutions.columns)
    resolutions = resolutions.loc[:, columns].drop_duplicates(subset=['domain', 'ip'])
    resolutions = resolutions.dropna(subset=['domain', 'ip']).reset_index(drop=True)
    resolutions['apex'] = [get_2ld(domain) for domain in resolutions['domain'].values]
    
    if remove_public is not None:
        public_df = load_path_or_df(remove_public, header=None)     
        public_domains = public_df.iloc[:, 0].values

        resolutions = resolutions[~(resolutions.apex.isin(public_domains))]
    return resolutions


def get_domain_ip_indexes(resolutions):
    """
    Extract apexes and allocate indexs for domain and ip for the graph

    :param resolutions: Dataset (DataFrame) of domain-ip resolutions
    :returns: DataFrame with extracted apexes and allocated node indexes
    """
    resolutions['domain_node'], _ = __get_node_indexes('d', resolutions['domain'].values)
    resolutions['ip_node'], _ = __get_node_indexes('i', resolutions['ip'].values)
    return resolutions


def get_dynamic_domain_ip_indexes(resolutions, existing_domain_map=None, existing_ip_map=None, time=None):
    """
    Extract apexes and allocate indexs for domain and ip for the graph

    :param resolutions: Dataset (DataFrame) of domain-ip resolutions
    :returns: DataFrame with extracted apexes and allocated node indexes
    """
    resolutions['domain_node'], domain_map = __get_node_indexes('d', resolutions['domain'].values, existing_domain_map, time=time)
    resolutions['ip_node'], ip_map = __get_node_indexes('i', resolutions['ip'].values, existing_ip_map, time=time)
    return resolutions, domain_map, ip_map


# TODO - Move stats here


def get_subnet_indexes(resolutions):
    subnets = resolutions.loc[:, ['ip']].drop_duplicates()

    def extract_subnet(ip):
        return ".".join(ip.split('.')[:3])

    subnets['subnet'] = subnets['ip'].apply(extract_subnet)
    subnets['subnet_node'], _ = __get_node_indexes('s', subnets['subnet'].values)
    return subnets.drop_duplicates()


def get_asn_indexes(resolutions, asn_path):
    asn = maxmind_utils.MyASN(asn_path)

    asns = resolutions.loc[:, ['ip']].drop_duplicates()
    asns['asn'] = asns['ip'].apply(asn.get_asn)
    asns['asn_node'] = asns['asn'].str.lower()
    return asns.drop_duplicates()


def get_apex_indexes(resolutions):
    domains = resolutions.loc[:, ['domain', 'apex']].drop_duplicates()
    
    apexes = pd.DataFrame({'subdomains' : domains.groupby('apex').size()}).reset_index()
    apexes = apexes[(apexes.apex != '') & (apexes.subdomains > 1)]
    apexes['apex_node'] = __get_node_indexes('a', apexes['apex'].values)
    return apexes.drop_duplicates()
    
    
def label_domains(resolutions, attacker_owned, alexa):
    """
    Set labels for domains in resolutions

    :param resolutions: Dataset (DataFrame) of domain-ip resolutions
    :param attacker_owned: Dataset (DataFrame) or path to the dataset (str) with attacker-owned domains 
    :param alexa: Dataset (DataFrame) or path to the dataset (str) with alexa domains 
    :returns: DataFrame with labelled domains
    """
    attacker_owned = load_path_or_df(attacker_owned)
    alexa = load_path_or_df(alexa)

    def label_attacker_owned_domains(x):
        domain_type = 'malicious' if x['apex'] in attacker_owned.iloc[:, 0].values else 'n_a'
        return (x['apex'], x['domain'], x['domain_node'], domain_type)
        
    def label_benign_domains(x):
        return (x['apex'], x['domain'], x['domain_node'], 'benign' if x['apex'] in alexa.iloc[:, 0].values else x['type'])

    domains = resolutions.loc[:, ['domain', 'apex', 'domain_node']].drop_duplicates()
    attacker_owned_labelled = domains.parallel_apply(label_attacker_owned_domains, axis=1, result_type='expand')
    attacker_owned_labelled.columns = ['apex', 'domain', 'domain_node', 'type']

    all_labelled_domains = attacker_owned_labelled.parallel_apply(label_benign_domains, axis=1, result_type='expand')
    all_labelled_domains.columns = ['apex', 'domain', 'domain_node', 'type']
    
    return all_labelled_domains


def get_domain_features(domains, feature_type='lexical', tlds_path=None):
    """
    Set labels for domains in resolutions

    :param resolutions: Dataset (DataFrame) of domain-ip resolutions
    :param attacker_owned: Dataset (DataFrame) or path to the dataset (str) with attacker-owned domains 
    :param alexa: Dataset (DataFrame) or path to the dataset (str) with alexa domains 
    :returns: DataFrame with labelled domains
    """
    if feature_type not in ['lexical', 'ringer']:
        raise Exception("feature_type should be either `lexical` or `ringer`")
    elif feature_type == 'ringer' and not tlds_path:
        raise Exception("tlds_path should be passed for `ringer` feature extraction")
        
    if feature_type == 'lexical':
        domain_features = domains.parallel_apply(lambda row: {'domain':row['domain'], **lf.get_features(row['domain'])}, axis=1, result_type='expand')
        
        domain_features['length'] = MinMaxScaler().fit_transform(np.array(domain_features['length']).reshape(-1,1))
        domain_features['entropy'] = MinMaxScaler().fit_transform(np.array(domain_features['entropy']).reshape(-1,1))
    elif feature_type == 'ringer':
        tlds = get_tlds(tlds_path)
        domain_features = domains.parallel_apply(lambda row: {'domain':row['domain'], **rf.get_features(row['domain'], tlds)}, axis=1, result_type='expand')

    return domain_features


def get_ip_features(ips, norm=True, ngram=True):
    def extract_subnet(ip, n=3):
        return ".".join(ip.split('.')[:n])

    # le = LabelEncoder()
    # mm = MinMaxScaler()

    ip_features = ips.loc[:, ['ip']]
    if ngram:
        ip_features['subnet_a'] = ip_features['ip'].parallel_apply(extract_subnet, n=1)
        ip_features['subnet_b'] = ip_features['ip'].parallel_apply(extract_subnet, n=2)
        ip_features['subnet_c'] = ip_features['ip'].parallel_apply(extract_subnet, n=3)
    else:
        ip_features['subnet'] = ip_features['ip'].parallel_apply(extract_subnet)
    ip_features = ip_features.dropna()
    
    if norm:
        cols = list(ip_features.columns)
        for col in cols:
            if 'subnet' in col:
                encoder=ce.HashingEncoder(cols=[col], n_components=4)
                ip_features = encoder.fit_transform(ip_features)
                ip_features.columns = [col+c[-2:] if 'col' in c else c for c in ip_features.columns]

        # ip_features['subnet'] = mm.fit_transform(np.array(le.fit_transform(ip_features['subnet'])).reshape(-1, 1))

    return ip_features


def get_asn_features(asns):
    asn_features = pd.DataFrame(asns.loc[:, ['asn_node']]).drop_duplicates()
    
    encoder=ce.HashingEncoder(cols=['asn_node'], n_components=10)
    asn_features = encoder.fit_transform(asn_features)
    asn_features['asn_node'] = asns.asn_node
    
    return asn_features


def get_nodes_and_extras(domains=None, ips=None, subnets=None, asns=None, apexes=None, domain_features=None, 
                         ip_features=None, subnet_features=None, asn_features=None):
    graph_nodes = {}
    extras = pd.DataFrame([], columns=['node_id', 'value', 'type']) 

    if domains is not None:
        labelled_domains = domains.loc[:, ['domain_node', 'domain']]
        
        if domain_features is not None:
            labelled_domains = labelled_domains.join(domain_features.set_index('domain'), on='domain').drop(columns=['domain']).dropna()

        domain_extras = domains.loc[:, ['domain_node', 'domain', 'type']]
        domain_extras.columns = ['node_id', 'value', 'type']

        domain_nodes = labelled_domains.set_index('domain_node').rename(columns={'domain':'value'})
        graph_nodes["domain"] = domain_nodes
        extras = extras.append(domain_extras)
    
    if ips is not None:
        labelled_ips = ips.loc[:, ['ip_node', 'ip']].drop_duplicates()

        if ip_features is not None:
            labelled_ips = labelled_ips.join(ip_features.set_index('ip'), on='ip').drop(columns=['ip']).dropna()

        ip_extras = ips.loc[:, ['ip_node', 'ip']]
        ip_extras['type'] = ['n_a'] * len(ip_extras) 
        ip_extras.columns = ['node_id', 'value', 'type']

        ip_nodes = labelled_ips.set_index('ip_node').rename(columns={'ip':'value'})
        graph_nodes["ip"] = ip_nodes
        extras = extras.append(ip_extras)

    if subnets is not None:    
        labelled_subnets = subnets.loc[:, ['subnet_node']].drop_duplicates()
        
        if subnet_features is not None:
            labelled_subnets = labelled_subnets.join(subnet_features.set_index('subnet_node'), on='subnet_node').dropna()

        subnet_extras = subnets.loc[:, ['subnet_node', 'subnet']]
        subnet_extras['type'] = ['n_a'] * len(subnet_extras) 
        subnet_extras.columns = ['node_id', 'value', 'type']

        subnet_nodes = labelled_subnets.set_index('subnet_node').rename(columns={'subnet':'value'})
        graph_nodes["subnet"] = subnet_nodes
        extras = extras.append(subnet_extras)
    
    if asns is not None:
        labelled_asns = asns.loc[:, ['asn_node']].drop_duplicates()
        
        if asn_features is not None:
            labelled_asns = labelled_asns.join(asn_features.set_index('asn_node'), on='asn_node').dropna()

        asns_extras = asns.loc[:, ['asn_node']]
        asns_extras['asn'] = asns_extras['asn_node']
        asns_extras['type'] = ['n_a'] * len(asns_extras) 
        asns_extras.columns = ['node_id', 'value', 'type']

        asn_nodes = labelled_asns.set_index('asn_node').rename(columns={'asn':'value'})
        graph_nodes["asn"] = asn_nodes
        extras = extras.append(asns_extras)
    
    if apexes is not None:
        labelled_apexes = apexes.loc[:, ['apex_node']].drop_duplicates()

        apex_extras = apexes.loc[:, ['apex_node', 'apex']]
        apex_extras['type'] = ['n_a'] * len(apex_extras) 
        apex_extras.columns = ['node_id', 'value', 'type']

        edges_list.append(domain_to_apex_edges)
        graph_nodes["apex"] = apex_nodes
        extras = extras.append(apex_extras)
    
    return graph_nodes, extras


def get_edges(resolutions=None, subnets=None, asns=None, apexes=None, similar_domains=None, apex_edge=False):
    edges_list = []

    if resolutions is not None:
        resolve_edges = resolutions.loc[:, ['domain_node', 'ip_node']].drop_duplicates().dropna()
        resolve_edges['type'] = ['resolves'] * len(resolve_edges)
        resolve_edges.columns = ['source', 'target', 'type']
        
        edges_list.append(resolve_edges)
        log("domain-ip edges added")
        
        if subnets is not None:
            ip_to_subnet = resolutions.loc[:, ['ip_node', 'ip']].join(subnets.set_index('ip'), on='ip')
            ip_to_subnet_edges = ip_to_subnet.loc[:, ['ip_node', 'subnet_node']].drop_duplicates().dropna()
            ip_to_subnet_edges['type'] = ['in_subnet'] * len(ip_to_subnet_edges)
            ip_to_subnet_edges.columns = ['source', 'target', 'type']

            edges_list.append(ip_to_subnet_edges)
            log("ip-subnet edges added")

            if asns is not None:
                subnet_to_asn = ip_to_subnet.loc[:, ['subnet_node', 'ip']].join(asns.set_index('ip'), on='ip')
                subnet_to_asn_edges = subnet_to_asn.loc[:, ['subnet_node', 'asn_node']].drop_duplicates().dropna()
                subnet_to_asn_edges['type'] = ['in_asn'] * len(subnet_to_asn_edges)
                subnet_to_asn_edges.columns = ['source', 'target', 'type']

                edges_list.append(subnet_to_asn_edges)
                log("subnet-asn edges added")
                
        if subnets is None and asns is not None:
            ip_to_asn = resolutions.loc[:, ['ip_node', 'ip']].join(asns.set_index('ip'), on='ip')
            ip_to_asn_edges = ip_to_asn.loc[:, ['ip_node', 'asn_node']].drop_duplicates().dropna()
            ip_to_asn_edges['type'] = 'in_asn'
            ip_to_asn_edges.columns = ['source', 'target', 'type']

            edges_list.append(ip_to_asn_edges)
            log("ip-asn edges added")
            
        if apexes is not None:
            domain_to_apex_edges = resolutions.loc[:, ['domain_node', 'apex']].join(apexes.set_index('apex'), on='apex')
            domain_to_apex_edges = domain_to_apex_edges.loc[:, ['domain_node', 'apex_node']].drop_duplicates().dropna()
            domain_to_apex_edges['type'] = ['subdomain'] * len(domain_to_apex_edges)
            domain_to_apex_edges.columns = ['source', 'target', 'type']

            edges_list.append(domain_to_apex_edges)
            log("domain-apex edges added")

        if apex_edge:
            domain_to_subdomain = resolutions.loc[:, ['domain_node', 'domain', 'apex']].drop_duplicates().dropna()
            domain_to_subdomain = domain_to_subdomain.join(domain_to_subdomain.set_index('apex'), how='outer', on='apex', rsuffix='_right')
            domain_to_subdomain = domain_to_subdomain[~(domain_to_subdomain.domain == domain_to_subdomain.domain_right)]
            domain_to_subdomain = domain_to_subdomain[(domain_to_subdomain.domain == domain_to_subdomain.apex)]

            domain_to_subdomain = domain_to_subdomain.loc[:, ['domain_node', 'domain_node_right']].dropna()
            domain_to_subdomain = pd.DataFrame(np.sort(domain_to_subdomain.values, axis=1), columns=domain_to_subdomain.columns).drop_duplicates()
            domain_to_subdomain['type'] = ['apex'] * len(domain_to_subdomain)
            domain_to_subdomain.columns = ['source', 'target', 'type']

            edges_list.append(domain_to_subdomain)
            log("domain-domain subdomain edges added")

    if similar_domains is not None:
        similar_edges = similar_domains.loc[:, ['domain_node', 'domain_node2']].drop_duplicates().dropna()
        similar_edges = pd.DataFrame(np.sort(similar_edges.values, axis=1), columns=similar_edges.columns).drop_duplicates()
        similar_edges['type'] = ['similar'] * len(similar_edges)
        similar_edges.columns = ['source', 'target', 'type']
        
        edges_list.append(similar_edges)
        log("domain-domain similarity edges added")
        
    return pd.concat(edges_list).reset_index(drop=True)


def generate_graph_components():
    pass