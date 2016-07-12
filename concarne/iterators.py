# -*- coding: utf-8 -*-

from .utils import all_elements_equal_len

from collections import OrderedDict

import numpy as np

__all__ = ["AlignedBatchIterator", ]

# ---------------------------------------------------------------------------

def dict_slice(arr, sl):
    """
    Helper method to slice all arrays contained in a dictionary.
    """
    if isinstance(arr, dict):
        ret = OrderedDict()
        for k, v in arr.items():
            ret[k] = v[sl]
        return ret
    else:
        return arr[sl]

def list_slice(arr, sl):
    """
    Helper method to slice all arrays contained in a list.
    """
    if isinstance(arr, list) or isinstance(arr, tuple):
        ret = []
        for v in arr:
            ret.append(v[sl])
        return ret
    else:
        return arr[sl]

# ---------------------------------------------------------------------------

class AlignedBatchIterator(object):
    """
        A simple iterator class, accepts an arbitrary number of numpy arrays.
        
        Assumes that all numpy arrays are of equal length.
        
        Inspired by the BatchIterator class used in nolearn.
        https://github.com/dnouri/nolearn
        Copyright (c) 2012-2015 Daniel Nouri
        
        Parameters
        ----------
        batch_size : int
            Size of mini batch
        shuffle : bool, optional
            Shuffle data before iterating
    """
    
    def __init__(self, batch_size, shuffle=True):
        """
        batch_size - size of every minibatch 
        shuffle    - whether to shuffle data (default is true)
        """
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.elem_list = None

    def __call__(self, *args):
        """
        Note for developers:
        The __call__ magic function puts all passed arguments into a list
        elem_list which is used for iteration.
        
        It also checks whether all args contain the same number of elements.
        """
        self.elem_list = args

        # make sure some arguments are passed
        assert (len(args) > 0)
        
        # make sure all items have equal length
        assert (all_elements_equal_len(self.elem_list))
        
        return self

    def __iter__(self):
        bs = self.batch_size
        indices = range(len(self.elem_list[0]))
        if self.shuffle:
            np.random.shuffle(list(indices))
        for i in range((self.n_samples + bs - 1) // bs):
            sl = indices[slice(i * bs, (i + 1) * bs)]
            belem_dict = list_slice(self.elem_list, sl)
            yield belem_dict

    @property
    def n_samples(self):
        X = self.elem_list[0]
        if isinstance(X, dict):
            return len(list(X.values())[0])
        else:
            return len(X)

    @property
    def num_inputs(self):
        return len(self.elem_list)

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ('elem_list',):
            if attr in state:
                del state[attr]
        return state
        
