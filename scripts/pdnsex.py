'''
Description: Expanding on domain or IP based on domain-ip resolution graph
Author: @nabeelxy 
'''

import pdns
import sys
import os
from tqdm import tqdm
from datetime import datetime

def get_firstseen_for_domains(domain_file, outfile, lastseen):
  df = open(domain_file)
  of = open(outfile, "w+")
  for domain in df:
    domain = domain.strip()
    entries = pdns.pdns_domain(domain, None, None, None, lastseen)
    if entries != None:
      firstseen = pdns.get_first_seen_date(entries)
      of.write("{}###{}\n".format(domain, firstseen))
  of.close()
  df.close() 

def get_firstseen_lastseen_for_domains(domain_file, outfile):
  df = open(domain_file)
  of = open(outfile, "w+")
  for domain in df:
    domain = domain.strip()
    entries = pdns.pdns_domain(domain, None, None, None, None)
    if entries != None:
      [firstseen, lastseen] = pdns.get_first_last_seen_date(entries)
      of.write("{} {} {} {}\n".format(domain, firstseen, lastseen, (lastseen - firstseen)/86400))
  of.close()
  df.close() 

#Given a list of domains, get all IPs seen after a certain time
def get_ips_for_domains(domain_file, outfile, lastseen, limit):
  df = open(domain_file)
  of = open(outfile, "w+")
  for domain in df:
    domain = domain.strip()
    entries = pdns.pdns_domain(domain, None, None, None, lastseen, limit)
    if entries != None:
      for entry in entries:
        for ip in entry["rdata"]:
          of.write("{} {}\n".format(domain, ip))
  of.close()
  df.close() 

# get the ips for a given set of domains for time period between st (start time) and et (end time)
# include_outside captures those IPs that were seen before the start time and after the end time,
# not within the time interval (it may or may not have appeared within the time interval. The default
# value is false to exlude by default.
def get_ips_for_time_int(domain_file, outfile, st, et, include_outside = False, limit = 1000):
  df = open(domain_file)
  of = open(outfile, "w+")
  for domain in df:
    domain = domain.strip()
    entries = pdns.pdns_domain(domain, None, None, None, None, limit)
    if entries != None:
      for entry in entries:
        time_first = entry['time_first']
        time_last = entry['time_last']
        #The api returns only if the last seen is within that week. In other words,
        #if the last seen advnaces, it does not output the domain-ip pair when
        #time fenced with last seen. In this, we get all domain-ip pairs and
        #filter those that are likely to be seen on the given time frame. Note
        #that this is only an estimation. Some domain-ip pairs may or may not have
        #been seen on the given time interval.

        #conditions
        #first seen is within the interval
        #last seen is within the interval
        #while not seen in the interval, but seen before and after + include check
        if (time_first >= st and time_first <= et) or \
           (time_last >= st and time_last <= et) or \
           (include_outside and time_first < st and time_last > et): 
          for ip in entry["rdata"]:
            of.write("{} {}\n".format(domain, ip))
  of.close()
  df.close() 

def get_ips_for_time_int2(domain, st, et, include_outside = False, limit = 1000):
  new_entries = None
  entries = pdns.pdns_domain(domain, None, None, None, None, limit)
  if entries != None:
    for entry in entries:
      time_first = entry['time_first']
      time_last = entry['time_last']
      if (time_first >= st and time_first <= et) or \
         (time_last >= st and time_last <= et) or \
         (include_outside and time_first < st and time_last > et): 
        if new_entries == None:
          new_entries = list()
        new_entries.append(entry)
  return new_entries

#given a twold, get the likely seen authoritative NS entries for the twold
#given a twold, get the likely seen authoritative NS entries for the twold
def get_ns_for_time_int(twold, st, et, include_outside = False, limit = 100):
    entries = pdns.pdns_ns(twold, None, None, None, None, limit)
    new_entries = None
    #TODO - refactor this code
    if entries != None:
        for entry in entries:
            time_first = entry['time_first']
            time_last = entry['time_last']
            if (time_first >= st and time_first <= et) or \
                (time_last >= st and time_last <= et) or \
                (include_outside and time_first < st and time_last > et): 
                if new_entries == None:
                    new_entries = list()
                new_entries.append(entry)
    return new_entries

#given an IP, get the domains likely seen resolving to that IP during a given time period
def get_domains_for_time_int(ip, st, et, include_outside = False, limit = 5000):
    entries = pdns.pdns_ip(ip, None, None, None, None, limit)
    new_entries = None
    if entries != None:
        for entry in entries:
            time_first = entry['time_first']
            time_last = entry['time_last']
            if (time_first >= st and time_first <= et) or \
                (time_last >= st and time_last <= et) or \
                (include_outside and time_first < st and time_last > et): 
                if new_entries == None:
                    new_entries = list()
                new_entries.append(entry)
    return new_entries

#seen after a certain timestamp
def get_domains_for_ips(ip_file, outfile, lastseen, limit = None):
  ipf = open(ip_file)
  of = open(outfile, "w+")
  for ip in ipf:
    ip = ip.strip()
    entries =  pdns.pdns_ip(ip, None, None, None, lastseen, limit)
    if entries != None:
      for entry in entries:
        of.write("{} {}\n".format(entry['rrname'], ip))
  of.close()
  ipf.close()

def get_set_(inputf, index):
  ds = set()
  with open(inputf) as inf:
    for line in inf:
      ds.add(line.strip().split()[index].strip("."))
  return ds

def get_domainset(inputf):
  return get_set_(inputf, 0)

def get_ipset(inputf):
  return get_set_(inputf, 1)        

def get_domains_for_time_int_set(ipset, outf, st, et, outside, limit):
  of = open(outf, "w+")
  for ip in tqdm(ipset):
    ip = ip.strip()
    entries = get_domains_for_time_int(ip, st, et, outside, limit)
    if entries != None:
      for entry in entries:
        of.write("{} {}\n".format(entry['rrname'].strip("."),ip))
  of.close()

def get_ips_for_time_int2_set(domainset, outf, st, et, outside, limit):
  of = open(outf, "w+")
  for domain in tqdm(domainset):
    domain = domain.strip()
    entries = get_ips_for_time_int2(domain, st, et, outside, limit)  
    if entries != None:
      for entry in entries:
        for ip in entry['rdata']:
          of.write("{} {}\n".format(domain,ip))
  of.close()

# turn 1 -> apex to ip :_1
# turn 2 -> ip to domain :_2
# turn 3 -> domain to ip :_3
def expansion_generic(exp_type, start_time, end_time, limt, level, infile, outfile):
  ipset = set()
  domainset = set()
  turn = 0
  if exp_type == "ip":
    with open(infile) as inf:
      for ip in inf:
        ipset.add(ip.strip())
    get_domains_for_time_int_set(ipset, outfile+"_1", start_time, end_time, True, limit)
  elif exp_type == "domain":
    turn = 1
    with open(infile) as inf:
      for domain in inf:
        domainset.add(domain.strip())
    get_ips_for_time_int2_set(domainset, outfile + "_1", start_time, end_time, True, limit)
  else:
    print("Invalid expansion type: {}".format(exp_type))
    return
  
  if level == 1:
    return 

  prev_domainset = domainset
  prev_ipset = ipset
  print("Level: {} #IPs: {}".format(1, len(ipset)))
  print("Level: {} #Domains: {}".format(1, len(domainset)))

  for l in range(2, level+1):
    if l%2 == turn: #domain lookup
      domainset = get_domainset(outfile + "_" + str(l - 1))
      if prev_domainset != None:
        domainset_temp = domainset - prev_domainset
        prev_domainset = prev_domainset | domainset
        domainset = domainset_temp
      print("Level: {} #Domains: {}".format(l, len(domainset)))
      if len(domainset) == 0:
        print("Done")
        return 
      get_ips_for_time_int2_set(domainset, outfile + "_" + str(l), start_time, end_time, True, limit)
    else: #ip lookup
      ipset = get_ipset(outfile + "_" + str(l - 1))
      if prev_ipset != None:
        ipset_temp = ipset - prev_ipset
        prev_ipset = prev_ipset | ipset
        ipset = ipset_temp
      print("Level: {} #IPs: {}".format(l, len(ipset)))
      if len(ipset) == 0:
        print("Done")
        return
      get_domains_for_time_int_set(ipset, outfile + "_" + str(l), start_time, end_time, True, limit)

#Get the duration of a given list of domains
def get_duration(inputf, outputf):
  inf = open(inputf)
  outf = open(outputf, "w+")

  for domain in inf:
    domain = domain.strip()
    entries = pdns.pdns_domain(domain, None, None, None, "20180101 00:00", 5000)
    if entries != None:
      [firstseen, lastseen] = pdns.get_first_last_seen_date(entries)
      outf.write("{} {} {} {}\n".format(domain, firstseen, lastseen, float(lastseen - firstseen)/86400))
  outf.close()
 
#===================================================================================================
if len(sys.argv) < 4:
  print("Usage: {} [domain_lookup|ip_lookup|duration|firstseen|firstseen_lastseen|domain_lookup_int|ip_expansion|domain_expansion] input_file out_file")
  exit()

infile = sys.argv[2].strip()
outfile = sys.argv[3].strip()
method = sys.argv[1].strip()

if method == "domain_lookup":
  if len(sys.argv) < 5:
    print("Missing 4th parameter: lastseen_after")
    exit()
  lastseen = sys.argv[4]
  if lastseen == "None":
    lastseen = None
  limit = None
  if len(sys.argv) > 5:
    limit = int(sys.argv[5])
  get_ips_for_domains(infile, outfile, lastseen, limit)
elif method == "ip_lookup":
  if len(sys.argv) < 5:
    print("Missing 4th parameter: lastseen_after")
    exit()
  lastseen = sys.argv[4]
  if lastseen == "None":
    lastseen = None
  limit = None
  if len(sys.argv) > 5:
    limit = int(sys.argv[5])
  get_domains_for_ips(infile, outfile, lastseen, limit)
elif method == "firstseen":
  if len(sys.argv) < 5:
    print("Missing 4th parameter: lastseen_after")
    exit()
  lastseen = sys.argv[4]
  if lastseen == "None":
    lastseen = None
  get_firstseen_for_domains(infile, outfile, lastseen)
elif method == "firstseen_lastseen":
  get_firstseen_lastseen_for_domains(infile, outfile)
elif method == "domain_lookup_int":
  if len(sys.argv) < 6:
    print("Missing 4th and 5th parameter: start_time(YYYYMMDD HH:MM) end_time(YYYYMMDD HH:MM)")
    exit()
  os.environ["TZ"] = "UTC"
  start_time = int(datetime.strptime(sys.argv[4], '%Y%m%d %H:%M').strftime('%s'))
  end_time = int(datetime.strptime(sys.argv[5], '%Y%m%d %H:%M').strftime('%s'))
  #the last parameter includes those IPs seen before and after the interval (not within)
  get_ips_for_time_int(infile, outfile, start_time, end_time, True)
elif method == "ip_expansion":
  if len(sys.argv) < 8:
    print("Missing 4th, 5th, 6th and 7th parameters: start_time(YYYYMMDD HH:MM) end_time(YYYYMMDD HH:MM) limit level")
    exit()
  os.environ["TZ"] = "UTC"
  start_time = int(datetime.strptime(sys.argv[4], '%Y%m%d %H:%M').strftime('%s'))
  end_time = int(datetime.strptime(sys.argv[5], '%Y%m%d %H:%M').strftime('%s'))
  limit = int(sys.argv[6])
  level = int(sys.argv[7])
  if level < 1:
    print("Level is {}. But it must be greater than or equal to 1".format(level))
    exit()
  expansion_generic("ip", start_time, end_time, limit, level, infile, outfile)
elif method == "domain_expansion":
  if len(sys.argv) < 8:
    print("Missing 4th, 5th, 6th and 7th parameters: start_time(YYYYMMDD HH:MM) end_time(YYYYMMDD HH:MM) limit level")
    exit()
  os.environ["TZ"] = "UTC"
  start_time = int(datetime.strptime(sys.argv[4], '%Y%m%d %H:%M').strftime('%s'))
  end_time = int(datetime.strptime(sys.argv[5], '%Y%m%d %H:%M').strftime('%s'))
  limit = int(sys.argv[6])
  level = int(sys.argv[7])
  print(f"Domain Expansion: {start_time}, {end_time}, {limit}, {level}")
  if level < 1:
    print("Level is {}. But it must be greater than or equal to 1".format(level))
    exit()
  expansion_generic("domain", start_time, end_time, limit, level, infile, outfile)
elif method == "duration":
  get_duration(infile, outfile)
else:
  print("Unsupported operation")

