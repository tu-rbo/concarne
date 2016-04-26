from __future__ import absolute_import

"""
A set of helpers to use nolearn.lasagne functionality.

The most important function is `pattern2neuralnet` which duck-types any Pattern object
such that it behaves similarly to a nolearn.lasagne.NeuralNet object.

Supported `handlers`:
- PrintLayerInfo

Supported `visualize` methods:
- plot_conv_weights
- plot_conv_activity

"""

import nolearn.lasagne
from nolearn.lasagne.base import Layers

import nolearn.lasagne.handlers
import nolearn.lasagne.visualize

        
def pattern2neuralnet(pattern, verbose=0):
    pattern.layers = []
    pattern.layers_ = Layers()
    pattern.verbose = verbose
    
    for fn_name in ['phi', 'psi', 'beta']:
        if fn_name not in pattern.__dict__ or pattern.__dict__[fn_name] is None:
            continue
            
        fn = pattern.__dict__[fn_name]
        for i,l in enumerate(pattern._get_all_function_layers(fn)):
            name = l.name
            dct = l.__dict__
            if name is None:
                name = ""
            prefix = "%s%d" % (fn_name, i)
            if prefix not in name:
                name = prefix + ("_" + name if name != "" else name)
            dct['name'] = name
            pattern.layers_[name] = l    
            pattern.layers.append (( l.__class__, dct))
    
    return pattern

class PrintLayerInfo(nolearn.lasagne.handlers.PrintLayerInfo):
    def __call__(self, pattern, train_history=None):
        nolearn.lasagne.handlers.PrintLayerInfo.__call__(self, pattern2neuralnet(pattern), train_history=None)
