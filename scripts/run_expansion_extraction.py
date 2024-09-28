import os
import sys

import numpy as np
import pandas as pd

import calendar
import datetime

import psycopg2
import sqlalchemy as sa

from tqdm import tqdm
from pprint import pprint

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from VT_Forecast.libs import firstseen


def main(arguments):
    months = [
        (2020, 10),
        (2020, 11),
#         (2020, 12),
#         (2021, 1), 
#         (2021, 2), 
#         (2021, 3), 
#         (2021, 4),
    ]

    train_sizes = 7
    test_size = 3
    
    expanded_path = '../data/expanded/'
    vt_data_dir = '../data/continous' 
    out_dir = '../data/continous/expansions' 
    
    print("Extraction Started!")
    month_data = []
    for year, month in tqdm(months):
        from_day = datetime.datetime(year, month, 1)
        to_day = datetime.datetime(year, month, calendar.monthrange(year, month)[1])
        
        tag = from_day.strftime('%Y-%m')
        
        ex_path = expanded_path + f"{from_day.strftime('%Y-%m-%d')}_{to_day.strftime('%Y-%m-%d')}-filtered_attacker_apexes.pdns_all.csv"
        file_paths = firstseen.list_files(vt_data_dir, from_day.day, to_day.day, tag)
        
        print(ex_path)
        print(file_paths)

        expanded = pd.read_csv(ex_path)
        print(expanded.head())
        pdns = firstseen.preprocess_pdns(expanded)

        firstseen.store_expansions(pdns, file_paths, out_dir)
    print("Extraction Complete!")
    
    
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

