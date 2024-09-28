import os
import sys

from tqdm import tqdm
from pprint import pprint

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from VT_Forecast.libs import firstseen


def main(arguments):
    year = 2021
    month = 3
    string_filter = f"{year}-{month:02}"
    train_sizes = [5, 6, 7, 8, 9, 10]
    test_size = 5
    
    paths = {
        'expanded_path': '../data/expanded/2021-03-01_2021-03-31-filtered_attacker_apexes.pdns_all.csv',
        'vt_data_dir': '../data/continous',
    }
    
    splits = [[(i, i - 1 + train_size + test_size) for i in range(1, 30, (5))[:3]] for train_size in train_sizes]
    # splits = [[(i, i - 1 + train_size + test_size) for i in range(1, 30, test_size)[:3]] for train_size in train_sizes][:1]

    print("Extraction Started!")
    pprint(splits)
    print("")
        
    for split in tqdm(splits):
        for trial in tqdm(split):
            print("Running:", trial)
            from_date, to_date = trial
            firstseen.extract_firstseen(year, month, from_date, to_date, paths, string_filter, verbose=False)
    
    print("Extraction Complete!")
    
    
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

