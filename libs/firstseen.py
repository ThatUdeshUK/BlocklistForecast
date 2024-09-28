from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np
import tldextract as te

# import psycopg2
# import sqlalchemy as sa

from tqdm import tqdm


def get_table_name(year, month, from_date, to_date):
    return f'fqdn_{year}_{month:02}_{from_date:02}_{to_date:02}'


def list_files(data_dir, from_date, to_date, string_filter=""):
    directory = [join(data_dir, f) for f in listdir(data_dir)]
    files = [f for f in directory if isfile(f) and 'filtered_attacker_apexes.csv' in f and string_filter in f]
    files = sorted(files, key=lambda x: x.split("/")[-1].split("_"))[:][max(0, from_date - 1): to_date]
    return files


def list_future_files(data_dir, from_date, to_date=None, str_filter='filtered_attacker_apexes.csv'):
    year, month, day = from_date
    
    def check_date(x):
        fy, fm, fd = map(int, x.split("/")[-1].split("-")[:3])
        
        start = fy > year or (fy == year and fm > month) or (fy == year and fm == month and fd >= day)
        if to_date is not None:
            ey, em, ed = to_date
            end = fy < ey or (fy == ey and fm < em) or (fy == ey and fm == em and fd <= ed)
            return start and end
        return start
    
    directory = [join(data_dir, f) for f in listdir(data_dir)]
    files = [f for f in directory if isfile(f) and str_filter in f]
    files = sorted(files, key=lambda x: x.split("/")[-1].split("_"))
    files = list(filter(check_date, files))
    return files


def list_expansion_files(data_dir, from_date, to_date, string_filter=""):
    directory = [join(data_dir, f) for f in listdir(data_dir)]
    files = [f for f in directory if isfile(f) and 'filtered_attacker_apexes.pdns.csv' in f and string_filter in f]
    files = sorted(files, key=lambda x: x.split("/")[-1].split("_"))[:][max(0, from_date - 1): to_date]
    return files


def preprocess_pdns(pdns):
    pdns.drop_duplicates(subset=['domain', 'ip'], inplace=True)
    pdns.dropna(inplace=True)
    pdns.reset_index(inplace=True, drop=True)
    pdns['apex'] = pdns['domain'].apply(lambda x: te.extract(x).registered_domain)
    return pdns


def get_expansion(resolutions, attacker_owned):
    """ Collect the subset of expansion for the attacker-owned apexes from whole PDNS expansion
    
    :param resolutions: Dataset (DataFrame) of domain-ip resolutions from the PDNS expansion
    :param attacker_owned: Dataset (DataFrame) with attacker-owned apexes 
    :returns: Expansion for the attacker-owned apexes (DataFrame) as domain-ip resolutions
    """
    ips_for_apexes = resolutions[resolutions.domain.isin(attacker_owned['apex'].values)]
    domains_for_ips = resolutions[resolutions.ip.isin(ips_for_apexes['ip'].values)]
    expansion = pd.DataFrame(resolutions[resolutions.domain.isin(domains_for_ips['domain'].values)])

    return expansion.drop_duplicates(subset=['domain', 'ip'])


def create_table(conn, table, verbose=True):
    conn.execute(sa.text(f'DROP TABLE IF EXISTS {table}'))

    conn.execute(
        sa.text(
            f"""CREATE TABLE {table} (
            fqdn          VARCHAR(255)    NOT NULL,
            ip            VARCHAR(16)    NOT NULL,
            first_seen    DATE     NOT NULL,
            last_seen     DATE,
            PRIMARY KEY (fqdn, ip)
            )"""
        )
    )
    if verbose:
        print("Table created successfully")

    
def upsert_fqdns(conn, table, df):
    conn.execute(sa.text("DROP TABLE IF EXISTS temp_table"))

    conn.execute(
        sa.text(
            """CREATE TEMPORARY TABLE temp_table (
            fqdn          VARCHAR(255)    NOT NULL,
            ip            VARCHAR(16)    NOT NULL,
            first_seen    DATE     NOT NULL,
            last_seen     DATE,
            PRIMARY KEY (fqdn, ip)
            )"""
        )
    )
    df.to_sql("temp_table", conn, if_exists="append", index=False)

    conn.execute(
        sa.text(f"""\
            INSERT INTO {table} (fqdn, ip, first_seen, last_seen) 
            SELECT fqdn, ip, first_seen, last_seen FROM temp_table
            ON CONFLICT (fqdn, ip) DO
                UPDATE SET last_seen = EXCLUDED.last_seen
            """
        )
    )

    
def store_firstseen(conn, table, pdns, files):
    for file in tqdm(files):
        apexes = pd.read_csv(file, names=['apex'])

        expansion = get_expansion(pdns, apexes)

        fqdns = expansion.loc[:, ['domain', 'ip']].drop_duplicates()
        fqdns.columns = ['fqdn', 'ip']

        firstseen = '-'.join(file.split('/')[-1].split('-')[:3])
        fqdns['first_seen'] = [firstseen] * len(fqdns)
        fqdns['last_seen'] = [firstseen] * len(fqdns)

        upsert_fqdns(conn, table, fqdns)

        
def extract_firstseen(year, month, from_date, to_date, paths, string_filter="", verbose=False):
    expanded_path = paths['expanded_path']
    vt_data_dir = paths['vt_data_dir']
    
    engine = sa.create_engine('postgresql+psycopg2://postgres:qcri2020%40@0.0.0.0:5433/VT_Forecast', pool_recycle=3600);
    conn = engine.connect()
    
    table = get_table_name(year, month, from_date, to_date)
    create_table(conn, table)
    
    files = list_files(vt_data_dir, from_date, to_date, string_filter)
    print(files)
    
    expanded = pd.read_csv(expanded_path) #, names=['domain', 'ip'], sep=' ')
    print(expanded.head())
    pdns = preprocess_pdns(expanded)

    store_firstseen(conn, table, pdns, files)
    
    conn.close()
    

def store_expansions(pdns, files, out_dir):
    for file in tqdm(files):
        apexes = pd.read_csv(file, names=['apex'])

        expansion = get_expansion(pdns, apexes)

        fqdns = expansion.loc[:, ['domain', 'ip']].drop_duplicates()
        fqdns.columns = ['fqdn', 'ip']

        file_path = file.split('/')[-1][:-4] + '.pdns.csv'
        
        fqdns.to_csv(join(out_dir, file_path))
    
