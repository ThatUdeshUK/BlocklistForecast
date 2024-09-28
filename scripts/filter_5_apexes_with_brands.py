import sys
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm


def run(inp_path, brands_path, who_path, out_dir):
    print('Tagging domain apexes if they contain brand names\n')
    
    brands = []
    with open(brands_path, 'r') as file:
        brands = [line.strip() for line in tqdm(file)]
            
    who_df = pd.read_csv(who_path, header=None)
    who_df.columns = ['apex']
    
    apex_df = pd.read_csv(inp_path, header=None)
    apex_df.columns = ['apex']

    def apply_tag(row):
        apex = row['apex']
        for brand in brands:
            if brand in apex:
                return apex, True
        return apex, False
    
    apex_df = apex_df.apply(apply_tag, axis=1, result_type='expand')
    apex_df.columns = ['apex', 'has_brand']
    
    branded_df = apex_df[apex_df.has_brand == True]
    branded_path = out_dir + inp_path.split("/")[-1][:13] + "-filtered_branded.csv"
    pd.DataFrame(branded_df['apex'].dropna().unique()).to_csv(branded_path, header=False, index=False)
    
    filtered = pd.concat([who_df['apex'], branded_df['apex']])
    filtered = filtered.unique()
    
    print("No of filtered apexes:", len(filtered))
    output_path = out_dir + inp_path.split("/")[-1][:13] + "-filtered_attacker_apexes.csv"
    print("Writing the output to:", output_path)
    pd.DataFrame(filtered).to_csv(output_path, header=False, index=False)
   
            
def main(arguments):
    apexes_file = '../../../Data/vt/2020-10-13-20-filtered_apexes.csv'
    whois_pdns_file = '../data/2020-10-13-20-filtered_whois_pdns_apexes.csv'
    brands_file = '../../../Data/brands.txt'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inp', help="Filtered apexes file", default=apexes_file)
    parser.add_argument('-b', '--brands', help="WHOIS JSON file", default=brands_file)
    parser.add_argument('-w', '--who', help="WHOIS and PDNs filtered file", default=whois_pdns_file)
    parser.add_argument('-o', '--out', help="Output directory", default='../data/')

    args = parser.parse_args(arguments)

    inp_path = args.inp
    brands_path = args.brands
    who_path = args.who
    out_dir = args.out

    run(inp_path, brands_path, who_path, out_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
