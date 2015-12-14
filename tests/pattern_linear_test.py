# -*- coding: utf-8 -*-

#!/usr/bin/env python

from __future__ import print_function

import concarne
import concarne.patterns

import lasagne
import theano
import theano.tensor as T

import numpy as np


def build_linear_simple(input_layer, n_out, nonlinearity=None, name=None):
    network = lasagne.layers.DenseLayer(input_layer, n_out, nonlinearity=nonlinearity, b=None, name=name)
    return network    


class TestDirectData:
    """
      Setup class generating some simplistic direct data
    """
    def setup(self):
        self.X = np.array( [[0,1,2,3,4], [1, 1, 1, 1, 1]]).T
        self.Y = np.array( [0, 0, 1, 1, 1] )
        self.C = np.array( [[0,1,2,3,4]] ).T
        
        self.n = self.X.shape[1]
        self.m = self.C.shape[1]
        
        self.num_classes = 1 #2
    

class TestDirectPattern(TestDirectData):
    """
      Test the direct pattern with linear models
    """

    def test_pattern(self):
        input_var = T.matrix('inputs')
        target_var = T.ivector('targets')
        context_var = T.matrix('contexts')
        
        input_layer = lasagne.layers.InputLayer(shape=(None, self.n),
                                            input_var=input_var)
        phi = build_linear_simple( input_layer, self.m, name="phi")
        psi = build_linear_simple( phi, self.num_classes, 
            #nonlinearity=lasagne.nonlinearities.softmax,
            nonlinearity=None,
            name="psi")
    
        dp = concarne.patterns.DirectPattern(phi=phi, psi=psi, 
                                             target_var=target_var, 
                                             context_var=context_var,
                                             target_loss=lasagne.objectives.squared_error(
                                                             lasagne.layers.get_output(phi, input_var),
                                                             #phi.get_output_for(input_var), 
                                                             context_var)
                                             )
                                      
        assert (phi.W.get_value(borrow=True).shape == (2,1))
        phi.W.set_value ( np.array([[1,0]]).T )
        assert (phi.W.get_value(borrow=True).shape == (2,1))

        #assert (psi.W.get_value(borrow=True).shape == (2,1))
        assert (psi.W.get_value(borrow=True).shape == (1,1))
        psi.W.set_value ( np.array([[1,]]) )
        
        test_prediction = lasagne.layers.get_output(dp, deterministic=True)
        test_fn = theano.function([input_var], test_prediction)
        
        X_hat = test_fn(self.X)
    
        assert ( np.all(self.C == X_hat) )



    
if __name__ == "__main__":
    td = TestDirectPattern()
    td.setup()
    td.test_pattern()
    
