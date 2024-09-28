import sys
import argparse
import pandas as pd
import tldextract as te


public_urls = None


def remove_public(row):
    url = row['url']
    extracted = te.extract(url)
    
    if extracted.registered_domain in public_urls:
        return None
    else:
        return row


def run(inp_path, pub_path, out_dir):
    print('Remove public URLs from malicious VirusTotal URLs\n')
    
    df = pd.read_csv(inp_path)
    print('Input URLs count: ', len(df))
        
    global public_urls
    public_df = pd.read_csv(pub_path, header=None)
    public_urls = public_df[0].values
    print('Public domain count: ', len(public_urls))
    
    filtered_df = df.apply(remove_public, axis=1).dropna()
    print('Filtered URLs count: ', len(filtered_df))
    
    out_path = out_dir + inp_path.split('/')[-1][:-4] + '_no_public.csv'
    filtered_df.to_csv(out_path, index=False)
    print('\nWriting the output to', out_path)
    

def main(arguments):
    default_inp = '../data/2020-10-13_mal_firstseen_malicious.csv'
    default_pub = '../../../Data/public/pub_domains_20201029'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inp', help="Malicious URL data file", default=default_inp)
    parser.add_argument('-p', '--pub', help="Public URL data file", default=default_pub)
    parser.add_argument('-o', '--out', help="Output directory", default='../data/')

    args = parser.parse_args(arguments)

    inp_path = args.inp
    pub_path = args.pub
    out_dir = args.out

    run(inp_path, pub_path, out_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
