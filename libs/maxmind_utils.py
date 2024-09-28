from pygeoip import GeoIP, MEMORY_CACHE


class MyGeo:
    def __init__(self, maxmind_city_file):
        self.gi = GeoIP(maxmind_city_file, MEMORY_CACHE)

    def get_country(self, ip):
        ret = self.gi.record_by_addr(ip)
        if ret != None:
            return ret["country_name"]
        return None

    def get_lat_lon(self, ip):
        ret = self.gi.record_by_addr(ip)
        if ret != None:
            return [ret["latitude"], ret["longitude"]]
        return None


class MyASN:
    def __init__(self, asn_file):
        self.gi = GeoIP(asn_file, MEMORY_CACHE)

    def get_asn(self, ip):
        as_str = self.gi.org_by_addr(ip)
        if as_str != None:
            arr = as_str.split()
            if len(arr) > 0:
                return arr[0].strip()
        return as_str
