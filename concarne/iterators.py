# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np

class _list(list):
    pass


class _dict(dict):
    def __contains__(self, key):
        return True


def dict_slice(arr, sl):
    if isinstance(arr, dict):
        ret = OrderedDict()
        for k, v in arr.items():
            ret[k] = v[sl]
        return ret
    else:
        return arr[sl]


class SimpleBatchIterator(object):
    """
        A simple iterator class, accepts three numpy arrays, 
        inputs X, targets y and contexts C (context is optional).
        
        Assumes that all numpy arrays are of equal length.
        
        Inspired by the BatchIterator class used in nolearn.
        https://github.com/dnouri/nolearn
    """
    
    def __init__(self, batch_size, shuffle=True):
        """
        batch_size - size of every minibatch 
        shuffle    - whether to shuffle data (default is true)
        """
        
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, X, y, C=None):
        self.elem_dict = OrderedDict()
        self.elem_dict['X'] = X
        self.elem_dict['y'] = y
        if C is not None:
            self.elem_dict['C'] = C
            
        # make sure all items have equal length
        assert (not np.isnan(reduce(lambda x,y: x if x==y else np.nan, map(len, self.elem_dict.values()))))
        
        return self

    def __iter__(self):
        bs = self.batch_size
        indices = range(len(self.elem_dict.values()[0]))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range((self.n_samples + bs - 1) // bs):
            sl = indices[slice(i * bs, (i + 1) * bs)]
            belem_dict = dict_slice(self.elem_dict, sl)
            yield belem_dict.values()

    @property
    def n_samples(self):
        X = self.elem_dict.values()[0]
        if isinstance(X, dict):
            return len(list(X.values())[0])
        else:
            return len(X)

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ('elem_dict',):
            if attr in state:
                del state[attr]
        return state
        

class DualContextBatchIterator(SimpleBatchIterator):
    def __call__(self, X, y, CX, Cy):
        self.elem_dict = OrderedDict()
        self.elem_dict['X'] = X
        self.elem_dict['y'] = y
        self.elem_dict['CX'] = CX
        self.elem_dict['Cy'] = Cy
            
        # make sure all items have equal length
        assert (not np.isnan(reduce(lambda x,y: x if x==y else np.nan, map(len, self.elem_dict.values()))))
        
        return self