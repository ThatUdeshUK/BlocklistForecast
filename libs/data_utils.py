import pandas as pd
from tqdm import tqdm

from os import listdir
from os.path import isfile, join


def load_path_or_df(dataset, header=0):
    """
    Generalizer to load a DataFrame from a dataset or path to the dataset 

    :param dataset: Dataset (DataFrame) or path to the dataset (str)
    :returns: DataFrame
    """
    if type(dataset) is str:
        return pd.read_csv(dataset, header=header)
    elif type(dataset) is pd.DataFrame:
        return dataset
    else:
        raise TypeError('dataset should be either pd.DataFrame or str')
        

def load_path_func_or_df(dataset, func):
    """
    Generalizer to load a DataFrame from a dataset or from calling function using path to the dataset 

    :param dataset: Dataset (DataFrame) or path to the dataset (str)
    :param func: Function to be called if path (str) is given as dataset
    :returns: DataFrame
    """
    if type(dataset) is str:
        return func(dataset)
    elif type(dataset) is pd.DataFrame:
        return dataset
    else:
        raise TypeError('dataset should be either pd.DataFrame or str')


def concat_alexa(alexa_data_dir):
    """
    Concatinate daily Alexa Top domains in a directory into a single dataset

    :param alexa_data_dir: Path (str) to the directory with daily Alexa Top domain datasets
    :returns: DataFrame with concatined Alexa Top domains
    """
    directory = [join(alexa_data_dir, f) for f in listdir(alexa_data_dir)]
    files = [f for f in directory if isfile(f)]
    
    first_file = files[0]
    df1 = pd.read_csv(first_file, header=None, index_col=0, sep='\t')

    for file in tqdm(files[1:]):
        df2 = pd.read_csv(file, header=None, index_col=0, sep='\t')
        df1 = pd.merge(df1, df2, how='inner', on=[1])

    alexa = df1
    alexa.columns = ['domain']

    return alexa


def concat_vt(vt_data_dir, file_range=(None, None)):
    """
    Concatinate VirusTotal firstseen data in a directory into a single dataset

    :param vt_data_dir: Path (str) to the directory with VirusTotal firstseen datasets
    :returns: Tuple containing dataset (DataFrame) with concatined VirusTotal data, date (str) of the first file and date (str) of the last file
    """
    directory = [join(vt_data_dir, f) for f in listdir(vt_data_dir)]
    
    files = [f for f in directory if isfile(f) and 'mal_firstseen.csv' in f]
    files = sorted(files, key=lambda x: x.split("/")[-1].split("_"))[file_range[0]:file_range[1]]

    first_date = files[0].split("/")[-1][:10]
    last_date = files[-1].split("/")[-1][:10]
    print(f"Concatinating files from {first_date} to {last_date}")
    
    first_file = files[0]
    df1 = pd.read_csv(first_file)

    for file in tqdm(files[1:]):
        df2 = pd.read_csv(file)
        df1 = pd.concat([df1, df2])

    return df1, first_date, last_date
