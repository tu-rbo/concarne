# -*- coding: utf-8 -*-

__all__ = ["isfunction", "isiterable", "all_elements_equal_len", "generate_timestamp"]

import collections
import numpy as np
import datetime

def isfunction(f):
    return hasattr(f, '__call__')
    
def isiterable(lst):
    return isinstance(lst, collections.Iterable)
    
def all_elements_equal_len(lst):
    return (not np.isnan(reduce(lambda x,y: x if x==y else np.nan, map(len, lst))))    

def generate_timestamp():
  return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
