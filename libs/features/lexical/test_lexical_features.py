import lexical_features as lf
import brandseg

bs = brandseg.BrandSeg()

print("{}: {}".format("www.paypal.com.security.accountupdate.gq", lf.get_features("www.paypal.com.security.accountupdate.gq", bs)))
print("{}: {}".format("myapple.com", lf.get_features("myapple.com", bs)))
print("{}: {}".format("myapp1e.com", lf.get_features("myapp1e.com", bs)))
print("{}: {}".format("app1e.com", lf.get_features("app1e.com", bs)))
print("{}: {}".format("account-applesd.repeatbat.com", lf.get_features("account-applesd.repeatbat.com", bs)))
print("{}: {}".format("www.qcri.org", lf.get_features("www.qcri.org", bs)))
