import sys
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from dateutil import parser
from datetime import datetime


def extract_creation_expiry(df):
    def extract_dates(row):
        creation = None
        expiry = None

        for latest_whois in row['data'][:2]:
            if latest_whois['attributes']['whois_map'] == {}:
                continue

            creation_keys = ["Creation Date", "Creation date", "record created", "Registered", "Registered on", "Registration date"]
            for key in creation_keys:
                if key in latest_whois['attributes']['whois_map']:
                    creation = latest_whois['attributes']['whois_map'][key].split(" | ")[0][:19]
                    if creation:
                        try:
                            creation = parser.parse(creation)
                            if creation < datetime(datetime.today().year, 1, 1):
                                creation = None
                        except:
                            creation = None                            
                    break

            expiry_keys = ["Registry Expiry Date", "Expiration Time", "Expiry Date", "Expiry date", "expires"]
            for key in expiry_keys:
                if key in latest_whois['attributes']['whois_map']:
                    expiry = latest_whois['attributes']['whois_map'][key].split(" | ")[0]
                    if expiry and expiry != 'null':
                        try:
                            expiry = parser.parse(expiry)
                            if expiry > datetime(datetime.today().year + 1, 12, 31):
                                creation = None
                                expiry = None
                        except:
                            expiry = None
                    break

            if creation or expiry:
                break

        return row['apex'], creation, expiry
                        
    whois_date_df = df.apply(extract_dates, axis=1, result_type='expand')
    whois_date_df.columns = ['apex', 'creation', 'expiry']
    return whois_date_df;


def run(inp_path, who_path, pdns_path, out_dir):
    print('Filtering older registrations from VirusTotal URLs\n')
    
    whois_data = []
    with open(who_path, 'r') as whois_file:
        for line in tqdm(whois_file):
            apex_data = json.loads(line)
            
            if 'links' in apex_data and 'data' in apex_data:
                apex = apex_data['links']['self'].split('/')[6]
                
                apex_whois = apex_data['data']
                apex_count = apex_data['meta']['count']
                
                if apex_count > 0:
                    whois_data.append({
                        'apex': apex,
                        'data': apex_whois,
                        'count': apex_count
                    })
    
    
    whois_df = pd.DataFrame(whois_data)
    whois_df = extract_creation_expiry(whois_df)
    whois_df = whois_df[whois_df.creation.notnull()]
        
    whois_path = out_dir + inp_path.split("/")[-1][:13] + "-filtered_whois.csv"
    pd.DataFrame(whois_df['apex'].dropna().unique()).to_csv(whois_path, header=False, index=False)
        
        
    pdns_df = pd.read_csv(pdns_path, sep=' ', header=None)
    pdns_df.columns = ['apex', 'firstseen', 'lastseen', 'duration']
    pdns_df = pdns_df[pdns_df.duration < 365]

    pdns_path = out_dir + inp_path.split("/")[-1][:13] + "-filtered_pdns.csv"
    pd.DataFrame(pdns_df['apex'].dropna().unique()).to_csv(pdns_path, header=False, index=False)
    
    
    filtered = set(whois_df['apex'].to_list())
    filtered.update(set(pdns_df['apex']))
    print("No of apexes found:", len(filtered)) 
    
    output_path = out_dir + inp_path.split("/")[-1][:13] + "-filtered_whois_pdns_apexes.csv"
    print("Writing the output to:", output_path)
    pd.DataFrame(filtered).to_csv(output_path, header=False, index=False)
   
            
def main(arguments):
    apexes_file = '../../../Data/vt/2020-10-13-20-filtered_apexes.csv'
    whois_file = '../../../Data/vt/2020-10-13-20-filtered_apexes.whois.json'
    pdns_file = '../../../Data/pdns/2020-10-13-20-filtered_apexes_nowhois.new.duration.csv'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inp', help="Filtered apexes file", default=apexes_file)
    parser.add_argument('-w', '--who', help="WHOIS JSON file", default=whois_file)
    parser.add_argument('-p', '--pdns', help="No WHOIS duration CSV file", default=pdns_file)
    parser.add_argument('-o', '--out', help="Output directory", default='../data/')

    args = parser.parse_args(arguments)

    inp_path = args.inp
    who_path = args.who
    pdns_path = args.pdns
    out_dir = args.out

    run(inp_path, who_path, pdns_path, out_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
