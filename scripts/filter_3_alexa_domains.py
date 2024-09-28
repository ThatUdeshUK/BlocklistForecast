import sys
from os import listdir
from os.path import isfile, join
import argparse
import pandas as pd
import tldextract as te
from tqdm import tqdm

alexa_urls = None


def get_alexa(alx_dir):
    """
    Get intersection of URLs from all the Alexa files (thus the URLs persisted over a month)

    Args:
        alx_dir: Directory of Alexa files

    Returns:
        List of URLs
    """
    directory = [join(alx_dir, f) for f in listdir(alx_dir)]
    files = [f for f in directory if isfile(f)]

    first_file = files[0]
    df = pd.read_csv(first_file, header=None, index_col=0, sep='\t')

    for file in tqdm(files[1:]):
        df2 = pd.read_csv(file, header=None, index_col=0, sep='\t')
        df = pd.merge(df, df2, how='inner', on=[1])

    return df[1].values


def remove_alexa(row):
    url = row['url']
    extracted = te.extract(url)

    if extracted.registered_domain in alexa_urls:
        return None
    else:
        return row


def run(inp_path, alx_dir, out_dir):
    print('Remove Alexa URLs from malicious VirusTotal URLs\n')

    df = pd.read_csv(inp_path)
    print('Input URLs count: ', len(df))

    global alexa_urls
    alexa_urls = get_alexa(alx_dir)
    print('Alexa domain count: ', len(alexa_urls))

    filtered_df = df.apply(remove_alexa, axis=1).dropna()
    print('Filtered URLs count: ', len(filtered_df))

    out_path = out_dir + inp_path.split('/')[-1][:-4] + '_no_alexa.csv'
    filtered_df.to_csv(out_path, index=False)
    print('\nWriting the output to', out_path)


def main(arguments):
    default_inp = '../data/2020-10-13_mal_firstseen_malicious_no_public.csv'
    default_alx = '../../../Data/alexa'

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inp', help="Malicious URL data file", default=default_inp)
    parser.add_argument('-a', '--alx', help="Alexa URL directory", default=default_alx)
    parser.add_argument('-o', '--out', help="Output directory", default='../data/')

    args = parser.parse_args(arguments)

    inp_path = args.inp
    alx_dir = args.alx
    out_dir = args.out

    run(inp_path, alx_dir, out_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
