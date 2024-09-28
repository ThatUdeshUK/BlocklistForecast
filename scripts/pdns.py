'''
Description: DNSDB API wrapper
Author: @nabeelxy
'''

import os, json
import urllib
import sys
import contextlib
import random

if sys.version_info[0] < 3:
    from urllib2 import Request, urlopen, HTTPError
else:
    from urllib.request import urlopen
    from urllib.error import HTTPError
from datetime import datetime

#TODO:
#Now we have two more servers: 10.4.8.28 and 10.4.8.181  - add them 
FARSIGHT_LOCAL_SERVER = ["10.4.8.82", "10.4.8.28", "10.4.8.181"]
SERVER_MAX_INDEX = 2
FARSIGHT_API_KEY = "82827cf79576078112b80646585a583b081edcaba91bf6f7898c6ee1725febf7"

# Description:
# Farsight DNS access utility functions
# Farsight query parameters
# time_first_before - show records before the first seen datetime [UTC]
# time_first_after - show records after the first seen datetime [UTC]
# time_last_before - show records before the last seen datetime [UTC]
# time_last_after - show records after the last seen datetime [UTC]
# limit - limit the number of responses
# Reference: https://api.dnsdb.info/

# Build the get query parameters for query request to the farsight server
def build_query_params(first_before = None, first_after = None, last_before = None, last_after = None, limit = None):
  os.environ["TZ"] = "UTC"
  query_str = ""
  if (first_before != None):
    #TODO: change to UTC epoch time
    epoch_first_before = datetime.strptime(first_before, '%Y%m%d %H:%M').strftime('%s')
    if len(query_str) > 0:
      query_str += "&"
    query_str += "time_first_before="
    query_str += epoch_first_before

  if (first_after != None):
    epoch_first_after = datetime.strptime(first_after, '%Y%m%d %H:%M').strftime('%s')
    if len(query_str) > 0:
      query_str += "&"
    query_str += "time_first_after="
    query_str += epoch_first_after

  if (last_before != None):
    epoch_last_before = datetime.strptime(last_before, '%Y%m%d %H:%M').strftime('%s')
    if len(query_str) > 0:
      query_str += "&"
    query_str += "time_last_before="
    query_str += epoch_last_before

  if (last_after != None):
    epoch_last_after = datetime.strptime(last_after, '%Y%m%d %H:%M').strftime('%s')
    if len(query_str) > 0:
      query_str += "&"
    query_str += "time_last_after="
    query_str += epoch_last_after

  if (limit != None):
    if len(query_str) > 0:
      query_str += "&"
    query_str += "limit="
    query_str += str(limit)

  return query_str

# Get resolved IP for a given domain (only A records)
# Update: 20190924 - updated to return different types of resource records - A, AAA, SOA, MX, NS
# Dates in YYYYMMDD HH:MM format. Note that no validation is performed. 
# Caller must validate the datetime inputs before
# invoking this function
def pdns_domain(domain, first_before = None, first_after = None, last_before = None, last_after = None, limit = None, 
                rtype = "A", server = FARSIGHT_LOCAL_SERVER[random.randint(0, SERVER_MAX_INDEX)]):
  query_str = build_query_params(first_before, first_after, last_before, last_after, limit)
  if len(query_str) > 0:
    query_str = "?" + query_str
    
  request = urllib.request.Request("http://{}/lookup/rrset/name/{}/{}{}".format(server, domain, rtype, query_str), 
                    headers={
                      "Accept":'application/json',
                      "X-API-Key": FARSIGHT_API_KEY
                    })
  ip_objects = None
  try:
    with contextlib.closing(urlopen(request)) as response:
        #response = urlopen(request)
        ip_objects = [json.loads(obj) for obj in response]
  except HTTPError:
    #handle 404's
    pass
  return ip_objects
    
# Get the domains that a given IP resolves to
def pdns_ip(ip, first_before = None, first_after = None, last_before = None, last_after = None, 
            limit = None, server = FARSIGHT_LOCAL_SERVER[random.randint(0, SERVER_MAX_INDEX)]):
  query_str = build_query_params(first_before, first_after, last_before, last_after, limit)
  if len(query_str) > 0:
    query_str = "?" + query_str

  request = urllib.request.Request("http://{}/lookup/rdata/ip/{}{}".format(server, ip, query_str), 
                    headers={
                      "Accept":'application/json',
                      "X-API-Key": FARSIGHT_API_KEY
                    })
  ip_objects = None
  try:
    #response = urlopen(request)
    with contextlib.closing(urlopen(request)) as response:
        ip_objects = [json.loads(obj) for obj in response]
  except HTTPError:
    pass
  return ip_objects

def get_first_seen_date(mappings):
  first_seen = None
  first_seen_date = None
  if mappings == None:
   return None
  for entry in mappings:
    if first_seen == None:
      first_seen = entry["time_first"]
    elif first_seen > entry["time_first"]:
      first_seen = entry["time_first"]

  if first_seen != None:
    first_seen_date = datetime.utcfromtimestamp(first_seen)
  return first_seen_date

def get_first_last_seen_date(mappings):
  first_seen = None
  last_seen = None
  if mappings == None:
   return None
  for entry in mappings:
    if first_seen == None:
      first_seen = entry["time_first"]
    elif first_seen > entry["time_first"]:
      first_seen = entry["time_first"]
    if last_seen == None:
      last_seen = entry["time_last"]
    elif last_seen < entry["time_last"]:
      last_seen = entry["time_last"]
  return [first_seen, last_seen]

def pdns_ns(twold, first_before = None, first_after = None, last_before = None, last_after = None, 
            limit = None, server = FARSIGHT_LOCAL_SERVER[random.randint(0, SERVER_MAX_INDEX)]):
  query_str = build_query_params(first_before, first_after, last_before, last_after, limit)
  if len(query_str) > 0:
    query_str = "?" + query_str
  #print("http://{}/lookup/rdata/{}/ns{}".format(FARSIGHT_LOCAL_SERVER, twold, query_str))
  request = urllib.request.Request("http://{}/lookup/rrset/name/{}/ns{}".format(server, twold, query_str),
                    headers={
                      "Accept":'application/json',
                      "X-API-Key": FARSIGHT_API_KEY
                    })
  pdns_objects = None
  try:
    #response = urlopen(request)
    with contextlib.closing(urlopen(request)) as response:
        pdns_objects = [json.loads(obj) for obj in response]
  except HTTPError:
    pass
  return pdns_objects

