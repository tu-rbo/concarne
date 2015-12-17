# -*- coding: utf-8 -*-

__all__ = ["isfunction", "isiterable", "all_elements_equal_len"]

import collections
import numpy as np

def isfunction(f):
    return hasattr(f, '__call__')
    
def isiterable(lst):
    return isinstance(lst, collections.Iterable)
    
def all_elements_equal_len(lst):
    return (not np.isnan(reduce(lambda x,y: x if x==y else np.nan, map(len, lst))))    
