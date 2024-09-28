'''
Note: This code is adapted from github phishing_catcher project
'''
import re
from scipy.stats import entropy
from Levenshtein import distance
import sys
from collections import Counter

from libs.features.lexical import brandseg
from libs.features.lexical.confusables import unconfuse
from libs.features.lexical.suspicious import tlds, brands, popular_keywords

bseg = brandseg.BrandSeg()


def shannon_entropy(string):
    bases = Counter([tmp_base for tmp_base in string])
    # define distribution
    dist = [x/sum(bases.values()) for x in bases.values()]
 
    # use scipy to calculate entropy
    entropy_value = entropy(dist, base=2)
 
    return entropy_value


def get_features(domain):
    features = dict()

    #segment the brand
    res = bseg.segment_domain(domain)
    sub_words = res[0]
    dom_words = res[1]
    all_words = sub_words + dom_words
    tld = res[2]

    # Suspicious TLD
    features["suspicious_tld"] = 0
    for t in tlds:
        if t == tld:
            features["suspicious_tld"] = 1
            break

    # Remove initial '*.' for wildcard certificates bug
    if domain.startswith('*.'):
        domain = domain[2:]

    features["length"] = len(domain)

    # Entropy
    # Higher entropy is kind of suspicious
    features["entropy"] = shannon_entropy(domain)

    # IDN characters
    domain = unconfuse(domain)

    # Contains embedded TLD/ FAKE TLD
    features["fake_tld"] = 0
    #exclude tld
    for word in all_words:
        if word in ['com', 'net', 'org', 'edu', 'mil', 'gov']:
            features["fake_tld"] += 1

    # No. of popular brand names appearing in domain name
    features["brand"] = 0
    for br in brands:
        for word in all_words:
            if br in word:
                features["brand"] += 1

    # Appearance of popular keywords
    features["pop_keywords"] = 0
    for word in popular_keywords:
        if word in all_words:
            features["pop_keywords"] += 1

    # Testing int for keywords
    # Let's go for Levenshtein distance less than 2
    features["similar"] = 0
    for br in brands:
        # Removing too generic keywords (ie. mail.domain.com)
        for word in [w for w in all_words if w not in ['email', 'mail', 'cloud']]:
            if distance(str(word), str(br)) <= 2:
                features["similar"] += 1

    # Lots of '-' (ie. www.paypal-datacenter.com-acccount-alert.com)
    features["is_idn"] = 0
    if 'xn--' in domain:
        features["is_idn"] = 1
        features["minus"] = domain.count('-') - 2
    else:
        features["minus"] = domain.count('-')

    # Deeply nested subdomains (ie. www.paypal.com.security.accountupdate.gq)
    features["num_subdomains"] = domain.count('.') - 1

    return features
