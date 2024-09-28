def get_tlds(path):
    tlds = []
    
    with open(path, 'r') as f:
        tld = f.readline()
        while tld:
            tld = f.readline()
            tlds.append(tld.strip().lower())
    
    return tlds
