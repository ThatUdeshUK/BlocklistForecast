import sys
import argparse
import pandas as pd
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
from domain_tools import get_fqdn, get_2ld
import json


def get_agg_name(files):
    files = sorted(files)
    first = files[0].split('/')[-1].split("_")[0]
    last = files[-1].split('/')[-1].split("_")[0]
    
    return first + "-" + last.split("-")[-1]


def run(inp_dir, out_dir):
    print('Aggregate malicious, public removed, and alexa remove apexes\n')
    
    query = "mal_firstseen_malicious_no_public_no_alexa"
    
    files = [join(inp_dir, f) for f in listdir(inp_dir) if isfile(join(inp_dir, f)) and (query in f) ]
    
    df = pd.DataFrame()
    for file in tqdm(files):
        inp_df = pd.read_csv(file)
        df = pd.concat([inp_df, df])
        
    df['fqdn'] = df['url'].apply(get_fqdn)
    df['apex'] = df['fqdn'].apply(get_2ld)
    
    name = get_agg_name(files)

    apex_path = out_dir + name + "-filtered_apexes.csv"
    pd.DataFrame(df['apex'].dropna().unique()).to_csv(apex_path, header=False, index=False)
    print('\nWriting the output to', apex_path)
    

def main(arguments):
    data_dir = '../data/'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inp', help="Malicious URL data file", default=data_dir)
    parser.add_argument('-o', '--out', help="Output directory", default=data_dir)

    args = parser.parse_args(arguments)

    inp_path = args.inp
    out_dir = args.out

    run(inp_path, out_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
