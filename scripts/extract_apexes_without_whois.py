import sys
import argparse
import json
import pandas as pd
from tqdm import tqdm
from dateutil import parser


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
                        except:
                            expiry = None
                    break

            if creation or expiry:
                break

        return row['apex'], creation, expiry
                        
    whois_date_df = df.apply(extract_dates, axis=1, result_type='expand')
    whois_date_df.columns = ['apex', 'creation', 'expiry']
    return whois_date_df;


def run(inp_path, who_path, out_dir):
    print('Filtering apexes without WHOIS from VirusTotal URLs\n')
    
    df = pd.read_csv(inp_path, header=None)
    print('Input URLs count: ', len(df))
    apex_list = df[0].to_list()
    whois_fetched_list = []
    
    no_whois_data = {}
    whois_data = []
    with open(who_path, 'r') as whois_file:
        for line in tqdm(whois_file):
            apex_data = json.loads(line)
            
            if 'links' in apex_data and 'data' in apex_data:
                apex = apex_data['links']['self'].split('/')[6]
                whois_fetched_list.append(apex)
                
                apex_whois = apex_data['data']
                apex_count = apex_data['meta']['count']
                
                if apex_count <= 0:
                    no_whois_data[apex] = {
                        'data': apex_whois,
                        'count': apex_count
                    }
                else:
                    whois_data.append({
                        'apex': apex,
                        'data': apex_whois,
                        'count': apex_count
                    })
                    
        
    whois_df = pd.DataFrame(whois_data)
    whois_df = extract_creation_expiry(whois_df)
    
    creation_not_null_df = whois_df[whois_df.creation.notnull()]
    print(creation_not_null_df)
    
    creation_null_df = whois_df[whois_df.creation.isnull()]
    print(creation_null_df)
                    
    print("Apex list length:", len(apex_list))
    print("WHOIS fetched length:", len(whois_fetched_list))

    apex_whois_not_fetched = set(apex_list) - set(whois_fetched_list)
    print("Apexes where WHOIS not fetched:", len(apex_whois_not_fetched))
    
    filtered_apexes = set(no_whois_data.keys())
    filtered_apexes.update(apex_whois_not_fetched)
    filtered_apexes.update(creation_null_df['apex'])
    print("Apexes without WHOIS:", len(filtered_apexes))
    
    filtered_df = pd.DataFrame(filtered_apexes)
    
#     out_path = out_dir + inp_path.split('/')[-1][:-4] + '_nowhois.csv'
#     filtered_df.to_csv(out_path, index=False, header=False)
#     print('\nWriting the output to', out_path)
    

def main(arguments):
    apexes_file = '../data/2020-10-13-20-filtered_apexes.csv'
    whois_file = '../../../Data/vt/2020-10-13-20-filtered_apexes.whois.json'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inp', help="Filtered apexes file", default=apexes_file)
    parser.add_argument('-w', '--who', help="WHOIS JSON file", default=whois_file)
    parser.add_argument('-o', '--out', help="Output directory", default='../data/')

    args = parser.parse_args(arguments)

    inp_path = args.inp
    who_path = args.who
    out_dir = args.out

    run(inp_path, who_path, out_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
