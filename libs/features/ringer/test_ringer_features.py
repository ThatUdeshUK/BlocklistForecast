import ringer_features as rf
import json

def print_features(domain, features):
    print("{}: {}".format(domain, json.dumps(features, indent=4)))

print_features("www.paypal.com.security.accountupdate.gq", rf.get_features("www.paypal.com.security.accountupdate.gq"))
print_features("myapple.com", rf.get_features("myapple.com"))
print_features("a.myapp1e.com", rf.get_features("a.myapp1e.com"))
print_features("app1e.com", rf.get_features("app1e.com"))
print_features("app11e22.com", rf.get_features("app11e22.com"))
print_features("app1e.not_valid", rf.get_features("app1e.not_valid"))
print_features("apple.com.go.ly", rf.get_features("apple.com.go.ly"))
print_features("account-applesd.repeatbat.com", rf.get_features("account-applesd.repeatbat.com"))
print_features("www.qcri.org", rf.get_features("www.qcri.org"))
print_features("www1.qcri.org", rf.get_features("www1.qcri.org"))
print_features("192_168_1_1.qcri.org", rf.get_features("192_168_1_1.qcri.org"))


