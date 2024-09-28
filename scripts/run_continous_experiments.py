import os
import sys

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from VT_Forecast.libs import continous as cont

from tqdm import tqdm
from pprint import pprint


def main(arguments):
    year = 2021
    month = 3
    
    train_sizes = [6, 7, 8, 9, 10]
    test_size = 5
    
    paths = {
        'public_path': '../../../Data/public/pub_domains_20201029',
        'asn_path': '../../../Data/maxmind/geo_20210103_GeoIPASNum.dat',
        'alexa_path': '../data/alexa/2021-03-01_2021-03-31-alexa_domains.csv',
        'attacker_owned_path': '../data/longitudinal/2021-03-01_2021-03-31-filtered_attacker_apexes.csv',
        'embedding_path': '../data/continous/new_embeddings',
        'similarity_path': '../data/cos_sim_domains_0_8.21_mar.csv',
    }
    
    splits = [[(i, i - 1 + train_size + test_size) for i in range(1, 30, test_size)[:3]] for train_size in train_sizes]
    
    print("Experiment Started!")
    pprint(splits)
    print("")
        
    for split in tqdm(splits):
        for trial in tqdm(split):
            print("Running:", trial)
            from_date, to_date = trial
            cont.extract_continous_embeddings(year, month, from_date, to_date, test_size, paths, verbose=False)
    
    print("Experiment Complete!")
    
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

