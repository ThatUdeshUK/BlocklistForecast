import pandas as pd

from dateutil import parser
from datetime import datetime, timedelta

from pandarallel import pandarallel
pandarallel.initialize()

from libs.domain_utils import get_fqdn, get_2ld
from libs.data_utils import load_path_or_df, load_path_func_or_df


def log(msg, verbose=True):
    if verbose:
        print(msg)


def preprocess_vt(dataset):
    """
    Preproccess VirusTotal feed data to extract only required fields, and add derived fields

    :param dataset: Dataset (DataFrame) or path to the dataset (str) with VirusTotal feed data
    :returns: DataFrame with VirusTotal URLs
    """
    df = load_path_or_df(dataset)     
    df = pd.DataFrame(df.loc[:, ['url', 'positives']])

    df['apex'] = df['url'].parallel_apply(get_2ld)
    df['fqdn'] = df['url'].parallel_apply(get_fqdn)
    return df


def preprocess_whois(whois_path):
    """
    Preproccess WHOIS data and produce a DataFrame with creation and expiration dates of apexes

    :param whois_path: Path to the WHOIS file
    :returns: DataFrame with creation and expiration dates of apexes
    """
    whos_df = pd.read_json(whois_path, lines=True).dropna(subset=['data', 'links'])
    
    whos_df['apex'] = whos_df['links'].parallel_apply(lambda x: x['self'].split('/')[6])
    whos_df['data'] = whos_df['data'].parallel_apply(lambda x: x if len(x) > 0 else None)
    whos_df = pd.DataFrame(whos_df.loc[:, ['apex', 'data']]).dropna()

    def extract_dates(row):
        date_type_keys = {
            'creation': ["Creation Date", "Creation date", "record created", "Registered", "Registered on", "Registration date"],
            'expiry': ["Registry Expiry Date", "Expiration Time", "Expiry Date", "Expiry date", "expires"]
        }

        dates = {k:None for k in date_type_keys.keys()}

        for latest_whois in row['data'][:2]:
            if latest_whois['attributes']['whois_map'] == {}:
                continue

            for date_type, keys in date_type_keys.items():
                for key in keys:
                    if key in latest_whois['attributes']['whois_map']:
                        dates[date_type] = latest_whois['attributes']['whois_map'][key].split(" | ")[0][:19]
                        if dates[date_type]:
                            try:
                                dates[date_type] = parser.parse(dates[date_type])
                            except:
                                dates[date_type] = None    
                        break

            if dates['creation']:
                break

        return dates['creation'], dates['expiry']

    dates_series = whos_df.parallel_apply(extract_dates, axis=1, result_type="expand")
    result = whos_df.join(dates_series).reset_index(drop=True).iloc[:, [0, 2, 3]]
    result.columns = ['apex', 'creation', 'expiry']
    return result


def preprocess_pdns_duration(pdns_path):
    """
    Preproccess pDNS duration data and produce a DataFrame

    :param pdns_path: Path to the pDNS durations file
    :returns: DataFrame with pDNS durations of apexes
    """
    pdns_df = pd.read_csv(pdns_path, sep=' ', header=None)
    pdns_df.columns = ['apex', 'firstseen', 'lastseen', 'duration']
    return pdns_df


def preprocess_brands(brands_path):
    """
    Preproccess file with brands list and produce a DataFrame

    :param pdns_path: Path to the brands file
    :returns: DataFrame with brands
    """
    brands = []
    with open(brands_path, 'r') as file:
        brands = [line.strip() for line in file]
    return pd.DataFrame(brands, columns=['brand'])


def malicious_apexes(df, n=5, with_fqdn=False):
    """
    Extract malicious Apexes (apexes from URLs with positives greater than n) from VirusTotal URLs

    :param df: DataFrame with VirusTotal URLs
    :param n: Positives threshold to be malicious
    :returns: DataFrame with malicious apexes
    """
    ext_keys = ['apex'] if not with_fqdn else ['apex', 'fqdn'] 
    return pd.DataFrame(df[df.positives >= n]).loc[:, ext_keys].drop_duplicates().dropna().reset_index(drop=True)


def __substract_domains(df, domains):
    """
    Remove a apexes in domain list from the dataset

    :param df: DataFrame with VirusTotal apexes
    :param domain: Iterable of domains to be removed from the dataset
    :returns: DataFrame without given domains
    """
    def remove(row):
        apex = row['apex']

        if apex in domains:
            row['apex'] = None
            return row
        else:
            return row

    return df.parallel_apply(remove, axis=1).dropna().reset_index(drop=True)


def remove_public(df, public):
    """
    Remove a given set of public domains from VirusTotal apexes

    :param df: DataFrame with VirusTotal apexes
    :param public: Dataset (DataFrame) or path to the dataset (str) with pubilc domains 
    :returns: DataFrame without public domains URLs
    """
    public_df = load_path_or_df(public, header=None)     
    public_domains = public_df.iloc[:, 0].values

    return __substract_domains(df, public_domains)


def remove_alexa(df, alexa):
    """
    Remove a given set of Alexa domains from VirusTotal apexes

    :param df: DataFrame with VirusTotal apexes
    :param alexa: Dataset (DataFrame) or path to the dataset (str) with alexa domains 
    :returns: DataFrame without alexa domains URLs
    """
    alexa_df = load_path_or_df(alexa, header=None)     
    alexa_domains = alexa_df.iloc[:, 0].values

    return __substract_domains(df, alexa_domains)


def matching_whois_duration(df, whois, current_date=datetime.today()):
    """
    Apply WHOIS duration constraint to filter apexes with WHOIS creation within past year and expiration within next two years from the given current date

    :param df: DataFrame with first phase filtered apexes
    :param whos: WHOIS date dataset (DataFrame) preprosessed using `preprocess_whois` or path to the data file (str)
    :param current_date: Date (datetime) for WHOIS records to be checked with
    :returns: DataFrame with apexes matching WHOIS duration constraints
    """
    whois = load_path_func_or_df(whois, preprocess_whois)
    def whois_constraint(row):
        creation_rule = row['creation'] > current_date - timedelta(days=365)

        expiration_rule = True
        if row['expiry']:
            expiration_rule = (row['expiry'] < current_date + timedelta(days=2*365))

        return creation_rule and expiration_rule

    return df[df.apex.isin(whois[whois.apply(whois_constraint, axis=1)]['apex'].values)].reset_index(drop=True)


def matching_pdns_duration(df, pdns_duration):
    """
    Apply pDNS duration constraint to filter apexes with pDNS duration less than a year

    :param df: DataFrame with first phase filtered apexes
    :param pdns_duration: pDNS duration dataset (DataFrame) preprosessed using `preprocess_pdns_duration` or path to the data file (str)
    :returns: DataFrame with apexes matching pDNS duration constraints
    """
    pdns_duration = load_path_func_or_df(pdns_duration, preprocess_pdns_duration)
    pdns_filtered_df = pdns_duration[pdns_duration.duration < 365]

    return df[df.apex.isin(pdns_filtered_df['apex'].values)].reset_index(drop=True)


def match_brands(df, brands):
    """
    Extract VirusTotal apexes containing given brands

    :param df: DataFrame with first phase filtered apexes
    :param brands: Brand list dataset (DataFrame) preprosessed using `preprocess_brands` or path to the data file (str)
    :returns: DataFrame with apexes containing brands
    """
    brands = load_path_func_or_df(brands, preprocess_brands)
    def brand_filter(apex):
        for brand in brands['brand'].values:
            if brand in apex:
                return True
        return False


    return df[df['apex'].parallel_apply(brand_filter)]


def run_first_phase(dataset, public, alexa, with_fqdn=False, verbose=False):
    """
    Run the first phase of attacker-owned domain filtration. Output of this should be used as the input for the WHOIS and pDNS duration collection.
    
    Phase include:
    * Preprocessing
    * Malicious URL filtration
    * Public doamin removal
    * Alexa Top domain removal

    :param dataset: Dataset (DataFrame) or path to the dataset (str) with VirusTotal feed data
    :param public: Dataset (DataFrame) or path to the dataset (str) with pubilc domains 
    :param alexa: Dataset (DataFrame) or path to the dataset (str) with alexa domains 
    :returns: DataFrame with first phase filtered apexes
    """
    log("Step 1/4: Preprocess VT URLs", verbose)
    df = preprocess_vt(dataset)
    log(f"Step 1/4 [Results]: URLs - {df.shape[0]}\n", verbose)
    
    log("Step 2/4: Extract malicious apexes", verbose)
    df = malicious_apexes(df, with_fqdn=with_fqdn)
    log(f"Step 2/4 [Results]: Apexes - {df.shape[0]}\n", verbose)
    
    log("Step 3/4: Remove public domains", verbose)
    df = remove_public(df, public=public)
    log(f"Step 3/4 [Results]: Apexes - {df.shape[0]}\n", verbose)
    
    log("Step 4/4: Remove alexa domains", verbose)
    df = remove_alexa(df, alexa=alexa)
    log(f"Step 4/4 [Results]: Apexes - {df.shape[0]}\n", verbose)

    return df


def run_second_phase(df, whois, pdns_duration, brands, current_date=datetime.today(), verbose=True):
    """
    Run the second phase of attacker-owned domain filtration. This is the finalized output from the attacker-owned domain filtration.
    
    Phase include:
    * WHOIS duration constraint
    * pDNS duration constraint
    * Availability of brands in apexes

    :param df: DataFrame with first phase filtered apexes
    :param whois: WHOIS date dataset (DataFrame) preprosessed using `preprocess_whois` or path to the data file (str)
    :param pdns_duration: pDNS duration dataset (DataFrame) preprosessed using `preprocess_pdns_duration` or path to the data file (str)
    :param brands: Brand list dataset (DataFrame) preprosessed using `preprocess_brands` or path to the data file (str)
    :param current_date: Date (datetime) for WHOIS records to be checked with
    :returns: DataFrame with attacker-owned apexes
    """
    log("Step 1/3: Extract apexes matching WHOIS duration constraint", verbose)
    whois_df = matching_whois_duration(df, whois, current_date=current_date)
    log(f"Step 1/3 [Results]: Apexes matching WHOIS duration - {whois_df.shape[0]}\n", verbose)

    log("Step 2/3: Extract apexes matching pDNS duration constraint", verbose)
    pdns_df = matching_pdns_duration(df, pdns_duration)
    log(f"Step 2/3 [Results]: Apexes matching pDNS duration - {pdns_df.shape[0]}\n", verbose)

    log("Step 3/3: Extract apexes containing brands", verbose)
    brands_df = match_brands(df, brands)
    log(f"Step 3/3 [Results]: Apexes containing brands - {brands_df.shape[0]}\n", verbose)
    
    return pd.DataFrame(pd.concat([whois_df['apex'], pdns_df['apex'], brands_df['apex']]).unique(), columns=['apex'])
    
    
