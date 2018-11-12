# @author: charnix@pku.edu.cn

import numpy as np
import pandas as pd
from collections import OrderedDict
import hashlib

class FeatureCache(object):
    """Cache Feature
    """
    def __init__(self):
        # TODO: set cache size to avoid oom
        self.cache_map = OrderedDict()

    def put(self, key, val):
        assert key is not None and val is not None
        hash_key = self.key_hash(key)
        # print('put', hash_key)
        self.cache_map[hash_key] = val

    def get(self, key):
        hash_key = self.key_hash(key)
        # print('get', hash_key)
        if hash_key in self.cache_map:
            return self.cache_map[hash_key]
        else:
            return None

    def check(self, key):
        hash_key = self.key_hash(key)
        return hash_key in self.cache_map

    def key_hash(self, key):
        m5 = hashlib.md5()
        m5.update(key.encode('utf-8'))
        return m5.hexdigest()


xcache = FeatureCache()



def unit_test():
    ll = {
        '$(a) + $(b) ': np.zeros(10),
        'Resi($x)' : np.ones(5),
    }
    for k,v in ll.items():
        xcache.put(k,v)
        print(xcache.check(k))
    for k,v in ll.items():
        print(k, xcache.get(k), v)
    print('unit_test done!')

if __name__ == "__main__":
    unit_test()
        
