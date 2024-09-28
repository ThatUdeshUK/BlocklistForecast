import sys
import argparse
import pandas as pd
from tqdm import tqdm


def run(inp_path, out_dir):
    print('Filtering malicious URLs from VirusTotal URLs\n')
    
    df = pd.read_csv(inp_path)
    print('Input URLs count: ', len(df))
    
    filtered_df = df[df.positives >= 5]
    print('Filtered URLs count: ', len(filtered_df))
    
#     out_path = out_dir + inp_path.split('/')[-1][:-4] + '_malicious.csv'
#     filtered_df.to_csv(out_path, index=False)
#     print('\nWriting the output to', out_path)
    

def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inp', help="VT data file", default='../../../Data/vt/2020-10-13_mal_firstseen.csv')
    parser.add_argument('-o', '--out', help="Output directory", default='../data/')

    args = parser.parse_args(arguments)

    inp_path = args.inp
    out_dir = args.out

    run(inp_path, out_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
