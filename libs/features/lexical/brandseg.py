from wordsegment import load, segment
import tldextract
import re

class BrandSeg:
    def __init__(self):
        load()

    def segment_domain(self, domain):
        is_wild_card = False
        if domain.startswith('*'):
            is_wild_card = True
        ext = tldextract.extract(domain)
        subdomain_list = list()
        tld = ext.suffix
        if len(ext.subdomain) > 0:
            subdomain_list = self.segment_part(ext.subdomain)
        apex_list = self.segment_part(ext.domain)
        return [subdomain_list, apex_list, tld, is_wild_card]

    def segment_part(self, part):
        words = re.split("\W+", part)
        words_list = list()
        for word in words:
            words_list.append(segment(word))
        final_list = list()
        for value in words_list:
            for word in value:
                final_list.append(word)
        return final_list
