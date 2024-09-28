import re
import pandas as pd
from scipy.stats import entropy
from collections import Counter
import nltk

def shannon_entropy(string):
    bases = Counter([tmp_base for tmp_base in string])
    # define distribution
    dist = [x/sum(bases.values()) for x in bases.values()]
 
    # use scipy to calculate entropy
    entropy_value = entropy(dist, base=2)
 
    return entropy_value


def get_ngram_freq_dist(domain, n):
    ngrams = nltk.ngrams(domain, n)

    fdist = nltk.FreqDist(ngrams)
    return pd.DataFrame(fdist.items()).iloc[:, 1]


def get_features(domain, tlds):
    features = dict()

    features["length"] = len(domain)

    # Entropy
    # Higher entropy is kind of suspicious
    features["entropy"] = shannon_entropy(domain)

    # Deeply nested subdomains (ie. www.paypal.com.security.accountupdate.gq)
    features["num_subdomains"] = domain.count('.') - 1
    
    # Subdomain length
    subdomains = domain.split('/')[-1].split('.')[:-2]
    features["subdomain_len"] = len('.'.join(subdomains))
    
    # Whether domain contains 'www' or not
    features["has_www"] = 1 if 'www' in domain else 0
    
    # Has valid TLD
    features["valid_tlds"] = 1 if domain.split('.')[-1].lower() in tlds else 0
    
    # Has subdomain with only one character
    if len(subdomains) == 0:
        features['has_single_subdomain'] = 0
    features['has_single_subdomain'] = 1 if any(map(lambda x: len(x) == 1, subdomains)) else 0
    
    # Has TLDs as subdomains
    if len(subdomains) == 0:
        features['has_tld_subdomain'] = 0
    features['has_tld_subdomain'] = 1 if any(map(lambda x: x.lower() in tlds, subdomains)) else 0

    # Ratio of digit exclusive subdomains
    if len(subdomains) == 0:
        features['digit_ex_subdomains_ratio'] = 0
    else:
        features['digit_ex_subdomains_ratio'] = len(list(filter(lambda x: re.search('\d', x) == None, subdomains))) / len(subdomains)
    
    # Ratio of of underscores
    features['underscore_ratio'] = domain.count('_') / len(domain)
    
    # Has IP address
    features['has_ip'] = 1 if re.search("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}", domain) else 0
    
    
    # Contains digits
    features['has_digits'] = 1 if re.search('\d', domain) else 0
    
    # Vowel ratio
    features['vowel_ratio'] = len(re.findall('[aeiou]', domain)) / len(domain)
    
    # Digit ratio
    features['digit_ratio'] = len(re.findall('\d', domain)) / len(domain)
    
    # Alphabet cardinality
    features['alphabet_cardinality'] = len(set(domain))
    
    # Repeated character ratio
    features['repeated_char_ratio'] = len(re.findall(r'(\w)\1+', domain)) / len(domain)
    
    # Consecutive consonants ratio
    features['consec_consonants_ratio'] = len(re.findall('[^aeiou][^aeiou]', domain)) / len(domain)
    
    # Consecutive digit ratio
    features['consec_digit_ratio'] = len(re.findall('\d{2}', domain)) / len(domain)
    
    # n-gram freq distribution
    for n in range(1, 4):
        ngrams = get_ngram_freq_dist(domain, n)
        features[str(n) + '-gram-mean'] = ngrams.mean()
        features[str(n) + '-gram-std'] = ngrams.std()
        features[str(n) + '-gram-median'] = ngrams.median()
        features[str(n) + '-gram-max'] = ngrams.max()
        features[str(n) + '-gram-min'] = ngrams.min()
        features[str(n) + '-gram-lower-q'] = ngrams.quantile(0.25)
        features[str(n) + '-gram-upper-q'] = ngrams.quantile(0.75)
    
    return features
